"""
DEER
Code for downstream Transformer model

deep_evidential_emotion_regression.py
"""
import torch
import torch.nn as nn
import math

from deep_evidential_emotion_regression import *

class PositionalEncoding(nn.Module):
    # Code adapt from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(self, d_model, dropout= 0.1, max_len = 1800):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2.0) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel_DEER(nn.Module):
    def __init__(self,input_dim=768,output_dim=3,num_pretrain_layers=12,
                d_model=256, nhead=4, num_encoder_layers=4,dim_feedforward=256,dp = 0.1,device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.layer_weights=nn.Parameter(torch.ones(num_pretrain_layers) /num_pretrain_layers)
        self.fc_embed=nn.Linear(input_dim,d_model)
        self.pos_encoder = PositionalEncoding(d_model, dp)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=dp)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.out_params = DenseNormalGamma(d_model, output_dim)
        self.output_dim = output_dim
        self.device = device

    def forward(self, hidden_states,src_key_padding_mask=None):
        # PaddedData.data: torch.Size([B, T, 12, 768])
        norm_weights=nn.functional.softmax(self.layer_weights, dim=-1)
        src=(hidden_states*norm_weights.view(1, 1, -1, 1)).sum(dim=2)      # torch.Size([B, T, 768])

        src = src.permute(1,0,2)        # torch.Size([T, B, 768])
        src = self.fc_embed(src)
        src = self.pos_encoder(src)    
        output = self.transformer_encoder(src,src_key_padding_mask=src_key_padding_mask)        # torch.Size([T, B, 256])

        # mean pooling
        output = torch.mean(output,dim=0)        # torch.Size([B, 256])
        params = self.out_params(output)        # (torch.Size([B, 3],torch.Size([B, 3],torch.Size([B, 3])
        return params

