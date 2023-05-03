'''
DEER
Utils

Author: Wen 2022
'''

import torch
import torch.nn as nn
from torch.utils import data
import functools
import speechbrain as sb

def concordance_correlation_coefficient(targets, predictions):
    # Modified from https://github.com/stylianos-kampakis/supervisedPCA-Python/blob/master/Untitled.py
    tmp = torch.stack((targets,predictions),dim=0)
    cor=torch.corrcoef(tmp)[0][1]
    
    mean_true=torch.mean(targets)
    mean_pred=torch.mean(predictions)
    
    var_true=torch.var(targets)
    var_pred=torch.var(predictions)
    
    sd_true=torch.std(targets)
    sd_pred=torch.std(predictions)
    
    numerator=2*cor*sd_true*sd_pred
    denominator=var_true+var_pred+(mean_true-mean_pred)**2

    return numerator/denominator

def CCC_loss(predictions, targets, length=None):
    loss = functools.partial(concordance_correlation_coefficient)
    return sb.nnet.losses.compute_masked_loss(
        loss, predictions, targets, length
    )

class MetricStats_std:
    """
    Storing and summarizing predictions and std.
    """
    def __init__(self, metric, n_jobs=1, batch_eval=True):
        self.metric = metric
        self.clear()

    def clear(self):
        self.preds = []
        self.targets = []
        self.ids = []
        self.stds = []
        self.summary = {}

    def append(self, ids, targets, predictions,stds):
        self.ids.extend(ids)
        self.stds.extend(stds)
        self.preds.extend(predictions.detach())
        self.targets.extend(targets.detach())

    def summarize(self, field=None):
        scores = self.metric(predictions=torch.stack(self.preds),targets=torch.stack(self.targets)).detach()
        avg_stds = sum(self.stds)/len(self.stds)
        return scores.item(),avg_stds.item()

    
# adapted from huggingface
def get_length_grouped_indices(lengths, batch_size, mega_batch_mult=None, generator=None,longest_first=False):
    if mega_batch_mult is None:
        mega_batch_mult = min(len(lengths) // (batch_size * 4), 50)
        if mega_batch_mult == 0:
            mega_batch_mult = 1

    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = mega_batch_mult * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    if longest_first:
        megabatches = [list(sorted(megabatch, key=lambda i: lengths[i], reverse=True)) for megabatch in megabatches]
    else:
        megabatches = [list(sorted(megabatch, key=lambda i: lengths[i])) for megabatch in megabatches]

    megabatch_maximums = [lengths[megabatch[0]] for megabatch in megabatches]
    max_idx = torch.argmax(torch.tensor(megabatch_maximums)).item()
    megabatches[0][0], megabatches[max_idx][0] = megabatches[max_idx][0], megabatches[0][0]

    return [i for megabatch in megabatches for i in megabatch]

class LengthGroupedSampler(data.Sampler):
    def __init__(self,batch_size: int,dataset= None,lengths= None,mega_batch_mult=None,generator=None,longest_first=False):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")
        self.batch_size = batch_size
        self.lengths = lengths
        self.generator = generator
        self.mega_batch_mult=mega_batch_mult
        self.longest_first=longest_first

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size,mega_batch_mult=self.mega_batch_mult,generator=self.generator,longest_first=self.longest_first)
        return iter(indices)
