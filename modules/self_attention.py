from torch import nn
import numpy as np
import torch
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, masked=False, attention_dropout=0.1, resid_dropout=0.1):
        super().__init__()
        self.dk = output_dim
        # initialize matrices WQ, WK, WV with random values
        self.WQ = nn.Parameter(nn.init.xavier_uniform_(torch.randn(input_dim, output_dim)))
        self.WK = nn.Parameter(nn.init.xavier_uniform_(torch.randn(input_dim, output_dim)))
        self.WV = nn.Parameter(nn.init.xavier_uniform_(torch.randn(input_dim, output_dim)))
        self.softmax = nn.Softmax(dim=1)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)
        self.masked = masked


        

    def __call__(self, x, y=None):
        # assert self.encoder_decoder == (y is not None)
        q = x @ self.WQ
        k = x @ self.WK
        v = x @ self.WV
        scores = q @ k.transpose(1, 2) / np.sqrt(self.dk)
        if self.masked:
            mask = torch.tril(torch.ones(x.size(1), x.size(1))).to(x.device)
            scores = scores.masked_fill(mask==0, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        scores = self.attention_dropout(scores)
        y = attention @ v
        # attention = self.resid_dropout(y)
        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, masked=False, prob_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        # self.WO = nn.Parameter(torch.randn(num_heads * output_dim, output_dim))
        self.WO = nn.Linear(num_heads * output_dim, output_dim)
        self.attentions = nn.ModuleList([SelfAttention(input_dim, output_dim, masked=masked) for _ in range(num_heads)])
        self.dropout = nn.Dropout(p=prob_dropout)
        self.masked = masked

    def __call__(self, x):
        outputs = torch.cat([attention(x) for attention in self.attentions], dim=2)
        return self.dropout(self.WO(outputs))
