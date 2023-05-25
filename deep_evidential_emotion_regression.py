'''
Code for deep evidential emotion regression (DEER)

Author: Wen 2022
'''

import torch
import numpy as np
from utils import *

def NIG_NLL(y, gamma, v, alpha, beta, reduce=False):
    # Adapted from https://github.com/hxu296/torch-evidental-deep-learning
    twoBlambda = 2*beta*(1+v)
    nll = 0.5*torch.log(np.pi/v)  \
        - alpha*torch.log(twoBlambda)  \
        + (alpha+0.5) * torch.log(v*(y-gamma)**2 + twoBlambda)  \
        + torch.lgamma(alpha)  \
        - torch.lgamma(alpha+0.5)
    return torch.mean(nll) if reduce else nll

class DenseNormalGamma(nn.Module):
    # Adapted from https://github.com/hxu296/torch-evidental-deep-learning
    def __init__(self, units_in, units_out):
        super(DenseNormalGamma, self).__init__()
        self.units_in = int(units_in)
        self.units_out = int(units_out)
        self.linear = nn.Linear(units_in, 4 * units_out)

    def evidence(self, x):
        softplus = nn.Softplus(beta=1)
        return softplus(x)

    def forward(self, x):
        output = self.linear(x)
        mu, logv, logalpha, logbeta = torch.split(output, self.units_out, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return torch.cat(tensors=(mu, v, alpha, beta), dim=-1)

    def compute_output_shape(self):
        return (self.units_in, 4 * self.units_out)
   

def NIG_Reg_phi(y, gamma, v, alpha, beta, reduce=False):
    error = torch.abs(y-gamma)
    evi = 1/(beta * (1 + v) / (v * (alpha-1)))    #pred_std
    reg = error*evi

    return torch.mean(reg) if reduce else reg


def DEER_loss(label, label_ref, label_mask,evidential_output, avg_rater=True, coeff_reg=1.0,ref_only=False,coeff_ref=0.0):
    B,num_rater,output_dim=label.size()
    label_var = torch.var(label,dim=1)
    gamma, v, alpha, beta = torch.split(evidential_output, int(evidential_output.shape[-1]/4), dim=-1)
    aleatoric = beta /  (alpha - 1)

    loss_reg = NIG_Reg_phi(label_ref, gamma, v, alpha, beta) + NIG_Reg_phi(label_var, aleatoric, v, alpha, beta)

    loss_nll_ref = NIG_NLL(label_ref, gamma, v, alpha, beta)

    if ref_only:
        return loss_nll_ref + coeff_reg * loss_reg

    loss_nll_all = 0
    for r_idx in range(num_rater):
        if len(label.shape)<=2:
            loss_r = NIG_NLL(label[:,r_idx], gamma, v, alpha, beta)
        else:
            loss_r = NIG_NLL(label[:,r_idx,:], gamma, v, alpha, beta)
        loss_nll_all += loss_r*label_mask[:,r_idx].unsqueeze(-1)
        
    if avg_rater:
        loss_nll_all /=torch.sum(label_mask,-1,keepdim=True)  

    return loss_nll_all + coeff_reg * loss_reg + coeff_ref * loss_nll_ref


