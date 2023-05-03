"""
DEER
Training script

Author: Wen 2022
"""
import os
import sys
import time
import torch
import random
import ruamel.yaml
import numpy as np
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

from deep_evidential_emotion_regression import *
from utils import *


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def dataio_prep(hparams):
    label_dic = np.load(hparams['label_file'],allow_pickle=True).item()

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("fea_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(fea_path):
        sig = sb.dataio.dataio.read_audio(fea_path)
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("seg_name")
    @sb.utils.data_pipeline.provides("label","utt_name")
    def label_pipeline(seg_name):
        label = label_dic[seg_name] # N_rater+1,n_dim
        yield torch.from_numpy(label).float()
        yield seg_name

    # Define datasets:
    datasets = {}
    hparams["dataloader_options"]["shuffle"] = False
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "label","utt_name"],
        )
    return datasets


# Brain class for DEER training
class DEER_Brain(sb.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        outputs = self.modules.SSLModel(wavs, lens)[1:]
        # print(outputs.shape) #[12, B, T, 768]#

        if len(outputs.shape)>3: 
            outputs=outputs.permute(1,2,0,3)    
        else:
            outputs=outputs.transpose(0,1)
        # print(outputs.shape) #[B, T, 12, 768]

        pred = self.modules.model(outputs)
        if self.hparams.output_dim == 1:
            return pred.squeeze(-1)
        else:
            return pred

    def check_nan(self,label_ref):
        label_ref = label_ref.transpose(0,1)
        for i, row in enumerate(label_ref):
            if len(set(row.tolist())) == 1: # avoid nan when computing PCC
                label_ref[i]+=torch.rand(label_ref[i].size()).to(self.device)*1e-3
        return label_ref.transpose(0,1)

    def compute_objectives(self, predictions, batch, stage):
        if stage == sb.Stage.TRAIN:
            label = batch.label#.data
            # Pad label to the same number of raters for batch operation by repeating the averaged label. 
            # Padded labels will be masked when computing the loss.
            max_rater = max([x.shape[0] for x in label])
            label_padded = []
            label_mask = []
            for x in label:
                label_mask.append(torch.tensor([1.0]*(x.shape[0]-1)+[0.0]*(max_rater-x.shape[0])))
                if x.shape[0]!= max_rater:
                    ref = x[-1,:].unsqueeze(0)
                    tmp = ref.expand(max_rater-x.shape[0],-1)
                    x=torch.cat([x,tmp])
                label_padded.append(x)                
            label=torch.stack(label_padded)
            label_mask = torch.stack(label_mask).to(self.device)
            # B, num_rater, output_dim = label.size()
            if self.hparams.output_dim == 1:
                label_ref = self.check_nan(torch.stack([x[-1,self.hparams.output_idx] for x in label]).unsqueeze(-1).to(self.device))#.squeeze(-1)
                label = label[:,:-1,self.hparams.output_idx].unsqueeze(-1)
            else:
                label_ref = self.check_nan(torch.stack([x[-1,:] for x in label]).to(self.device))
                label = label[:,:-1,:]
            
            assert(label.shape[0] == label_mask.shape[0] and label.shape[1] == label_mask.shape[1]),(label.shape,label_mask.shape)
        else:
            # during validation, batchsize=1，label set to padded format
            label = batch.label.data
            # B, num_rater, output_dim = label.size()
            if self.hparams.output_dim == 1:
                label_ref = label[:,-1,self.hparams.output_idx].unsqueeze(-1).to(self.device)
                label = label[:,:,self.hparams.output_idx].unsqueeze(-1)
            else:
                label_ref = label[:,-1,:].to(self.device)
                label = label[:,:-1,:]
            label_mask = torch.ones(label.shape[0],label.shape[1]).to(self.device)

        loss = DEER_loss(label=label, 
                         label_ref=label_ref, 
                         label_mask=label_mask,
                         evidential_output=predictions, 
                         coeff_reg=self.hparams.coeff_DEER, 
                         coeff_ref=self.hparams.coeff_ref,
                         avg_label=self.hparams.avg_label,
                         ref_only=self.hparams.ref_only,
                        )

        loss = torch.sum(loss)
        self.deer_loss += loss

        mu, v, alpha, beta = torch.split(predictions, int(predictions.shape[-1]/4), dim=-1)
        pred_std = torch.sqrt(beta * (1 + v) / (v * (alpha-1)))
        
        if self.hparams.output_dim == 1:
            if stage == sb.Stage.TRAIN:
                loss_CCC = 1.0 - CCC_loss(targets=label_ref.squeeze(-1),predictions=mu.squeeze(-1))
            self.CCC_metric.append(batch.id, targets=label_ref.squeeze(-1), predictions=mu.squeeze(-1), stds = pred_std)
        else:
            if stage == sb.Stage.TRAIN:
                loss_all = torch.stack([1.0 - CCC_loss(targets=label_ref[:,i],predictions=mu[:,i]) for i in range(self.hparams.output_dim)])
                loss_CCC = sum(loss_all)
            for i in range(self.hparams.output_dim):
                self.CCC_metrics[i].append(batch.id, targets=label_ref[:,i], predictions=mu[:,i], stds = pred_std[:,i])

        if stage == sb.Stage.TEST:
            utt_name = batch.utt_name
            assert len(utt_name)==len(pred_std)==len(mu)==len(label_ref),[len(utt_name),len(pred_std),len(mu),len(label_ref)]
            for i in range(len(utt_name)):
                self.test_outcome[utt_name[i]]=(label_ref[i].detach().cpu().numpy(),mu[i].detach().cpu().numpy(),v[i].detach().cpu().numpy(),alpha[i].detach().cpu().numpy(),beta[i].detach().cpu().numpy())

        if stage != sb.Stage.TRAIN:
            return loss
        self.CCC_loss += loss_CCC   # CCC loss used for mcdp and ensemble
        return loss


    def on_stage_start(self, stage, epoch=None):
        self.start_time = time.time()
        if self.hparams.output_dim == 1:
            self.CCC_metric = self.hparams.error_stats_std() 
        else:
            self.CCC_metrics = {i: self.hparams.error_stats_std() for i in range(self.hparams.output_dim)}
        
        self.deer_loss = 0.0
        self.CCC_loss = 0.0
        if stage == sb.Stage.TEST:
            self.test_outcome = {}
    
    def fit_batch(self, batch):
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # gradient clipping & early stop if loss is not finite
            self.check_gradients(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.detach()


    def on_stage_end(self, stage, stage_loss, epoch=None):
        self.elapse_time = time.time() - self.start_time
        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
            self.train_ccc_loss = self.CCC_loss
            self.train_deer_loss = self.deer_loss
            self.train_duration = self.elapse_time
            if self.hparams.output_dim == 1:
                self.train_CCC = self.CCC_metric.summarize()
            else:
                self.train_CCC = {i: self.CCC_metrics[i].summarize() for i in range(self.hparams.output_dim)}

        # Summarize the statistics from the stage for record-keeping.
        else:
            if self.hparams.output_dim == 1:
                stats = {
                    "loss": stage_loss,
                    'CCC': self.CCC_metric.summarize()[0],
                    'std': self.CCC_metric.summarize()[1],
                    'elapse': self.elapse_time,
                }
            else:
                stats = {
                    "loss": stage_loss,
                    'CCC': sum([self.CCC_metrics[i].summarize()[0] for i in range(self.hparams.output_dim)])/self.hparams.output_dim,
                    'std': sum([self.CCC_metrics[i].summarize()[1] for i in range(self.hparams.output_dim)])/self.hparams.output_dim,
                    "CCC-each": {i: self.CCC_metrics[i].summarize() for i in range(self.hparams.output_dim)},
                    'elapse': self.elapse_time,
                }

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            self.old_lr = old_lr
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": self.old_lr},
                train_stats={"loss": self.train_loss,
                            "loss-CCC": self.train_ccc_loss,
                            "loss-DEER": self.train_deer_loss,
                            "CCC": self.train_CCC,
                            "elapse": self.train_duration},
                valid_stats=stats,
            )

            self.checkpointer.save_and_keep_only(meta=stats, max_keys=["CCC"], num_to_keep=2)


        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
            np.save(os.path.join(self.hparams.output_folder,'test_outcome-E{}.npy'.format(self.hparams.epoch_counter.current)),self.test_outcome)


if __name__ == "__main__":
    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    if '--device' not in sys.argv[1:]:
        run_opts['device']= 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load hyperparameters file with command-line overrides.
    ruamel_yaml = ruamel.yaml.YAML()
    overrides = ruamel_yaml.load(overrides)
    overrides.update({'device': run_opts['device']})
    
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    set_seed(hparams['seed'])

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)

    hparams["SSLModel"] = hparams["SSLModel"].to(run_opts['device'])
    # freeze the feature extractor part when unfreezing
    if not hparams["freeze_SSL"] and hparams["freeze_SSL_conv"]:
        hparams["SSLModel"].model.feature_extractor._freeze_parameters()
    
    # Initialize the Brain object to prepare for mask training.
    DEER_brain = DEER_Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer_DEER"],
    )

    # Group batch by length
    lengths = [datasets["train"].data[x]['duration'] for x in datasets["train"].data_ids]
    generator = torch.Generator()
    generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
    train_sampler=LengthGroupedSampler(hparams['batch_size'],dataset=datasets["train"],lengths=lengths,generator=generator,longest_first=hparams['group_longest_first'])    #,mega_batch_mult=max(lengths)


    DEER_brain.fit(
        epoch_counter=DEER_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs={'sampler': train_sampler, 
                            **hparams["dataloader_options"]},
        valid_loader_kwargs={'batch_size':1},
    )

    test_stats = DEER_brain.evaluate(
    test_set=datasets["test"],
    max_key="CCC",
    test_loader_kwargs={'batch_size':1},
    )
