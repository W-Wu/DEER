'''
DEER
Code for scoring

Author: Wen 2022
'''

import numpy as np
import torch
from torch.distributions import Normal
import sys
from deep_evidential_emotion_regression import NIG_NLL


def reject_option(y_true,y_pred_mean,y_pred_std,idx = 0, rej_percentage=0.25):
    assert(len(y_true)==len(y_pred_mean)==len(y_pred_std))
    rmse_all = []
    rmse_remain = []
    for idx in range(len(y_true[0])):
        tmp_true = y_true[:,idx]
        tmp_mean = y_pred_mean[:,idx]
        tmp_std = y_pred_std[:,idx]
        rmse_all.append(RMSE(y_true=tmp_true, y_pred=tmp_mean))

        ind = np.argsort(tmp_std)   #ascending order
        sorted_mean=np.take_along_axis(tmp_mean,ind,axis=0)
        sorted_true=np.take_along_axis(tmp_true,ind,axis=0)

        num_remaining = int((1.0-rej_percentage) * len(sorted_true))
        remaining_mean=sorted_mean[:num_remaining]
        remaining_true=sorted_true[:num_remaining]
        rmse_remain.append(RMSE(y_true=remaining_true, y_pred=remaining_mean))

    return rmse_all,rmse_remain

def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient.
    Modified from https://github.com/stylianos-kampakis/supervisedPCA-Python/blob/master/Untitled.py
    """
    cor=np.corrcoef(y_true,y_pred)[0][1]
    
    mean_true=np.mean(y_true)
    mean_pred=np.mean(y_pred)
    
    var_true=np.var(y_true)
    var_pred=np.var(y_pred)
    
    sd_true=np.std(y_true)
    sd_pred=np.std(y_pred)
    
    numerator=2*cor*sd_true*sd_pred
    
    denominator=var_true+var_pred+(mean_true-mean_pred)**2

    return numerator/denominator

def RMSE(y_true, y_pred):
    mse = (y_pred-y_true)**2
    return np.sqrt(np.mean(mse,axis=0))

def compute_NLL_NIG(y_true_ref,y_true_all,gamma, v, alpha, beta):
    nll_ref = NIG_NLL(torch.from_numpy(y_true_ref),torch.from_numpy(gamma), torch.from_numpy(v), torch.from_numpy(alpha), torch.from_numpy(beta))
    nll_all = 0
    for i in range(len(gamma)):
        tmp = 0
        for x in y_true_all[i]:
            tmp += NIG_NLL(torch.from_numpy(x),torch.from_numpy(gamma[i]), torch.from_numpy(v[i]), torch.from_numpy(alpha[i]), torch.from_numpy(beta[i]))
        nll_all+= tmp/len(y_true_all[i])
    nll_all/=len(gamma)
    return nll_ref.mean(axis=0),nll_all#.mean(axis=0)

def process_target_label():
    dat = np.load(label_path,allow_pickle=True).item()
    target_std_dic = {k:np.std(v,axis=0) for k,v in dat.items()}
    target_mean_dic = {k:np.mean(v,axis=0) for k,v in dat.items()}
    target_all_dic = {k:v[:-1] for k,v in dat.items()}
    return target_mean_dic,target_std_dic,target_all_dic

def process_DEER_outcome(rpn_res_path,idx=-1):
    rpn_res = np.load(rpn_res_path,allow_pickle=True).item()
    target_mean=[]
    target_std=[]
    target_all=[]
    DEER_mu = []
    DEER_v = []
    DEER_alpha = []
    DEER_beta = []
    for k,val in rpn_res.items():
        true, mu, v, alpha, beta = val
        if len(true) < 3 and idx<0:
            print(len(true),idx)
            print("Please specify dimension")
            exit()
        elif len(true) < 3:
            target_all.append(np.array(target_all_dic[k][:,idx]).reshape(-1,1))
        else:
            target_all.append(np.array(target_all_dic[k]))
        target_mean.append(true)
        target_std.append(target_std_dic[k])
        DEER_mu.append(mu)
        DEER_v.append(v)
        DEER_alpha.append(alpha)
        DEER_beta.append(beta)

    target_mean=np.array(target_mean)
    target_std=np.array(target_std)
    DEER_mu=np.array(DEER_mu)
    DEER_v=np.array(DEER_v)
    DEER_alpha=np.array(DEER_alpha)
    DEER_beta=np.array(DEER_beta)
    return target_mean,target_std,target_all, DEER_mu,DEER_v,DEER_alpha,DEER_beta


res_path =  sys.argv[1]
idx_dic={0:'v',1:'a',2:'d'}
label_path = 'msp-data/msp-label.npy'
target_mean_dic,target_std_dic,target_all_dic = process_target_label()
target_mean,target_std,target_all, DEER_mu,DEER_v,DEER_alpha,DEER_beta  = process_DEER_outcome(res_path,idx=0)

epistemic = np.sqrt(DEER_beta / (DEER_v * (DEER_alpha - 1)))
pred_std = np.sqrt(DEER_beta * (1 + DEER_v) / (DEER_v * (DEER_alpha-1)))
aleatoric = np.sqrt(DEER_beta /  (DEER_alpha - 1))

ccc=[]
for idx in range(len(target_mean[0])):
    tmp_true = target_mean[:,idx]
    tmp_mean = DEER_mu[:,idx]
    ccc.append(concordance_correlation_coefficient(y_true=tmp_true, y_pred=tmp_mean))
print("CCC-mean\t{}\t{}\t{}".format(ccc[0],ccc[1],ccc[2]))

nll_NIG_ref, nll_NIG_all = compute_NLL_NIG(target_mean,target_all,DEER_mu,DEER_v,DEER_alpha,DEER_beta)
print("NLL_NIG_ref\t{}\t{}\t{}".format(nll_NIG_ref[0],nll_NIG_ref[1],nll_NIG_ref[2]))
print("NLL_NIG_all\t{}\t{}\t{}".format(nll_NIG_all[0],nll_NIG_all[1],nll_NIG_all[2]))

for percentage in range(0,100,10):
    rmse_all,rmse_remain = reject_option(target_mean,DEER_mu,pred_std,idx = 0, rej_percentage=percentage/100)
    print("rej RMSE(std)-{}%\t{}\t{}\t{}".format(percentage,rmse_remain[0],rmse_remain[1],rmse_remain[2]))
