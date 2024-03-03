from torch import nn
import numpy as np
import torch
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, masked=False, dropout=0):
        super().__init__()
        self.dk = output_dim
        # initialize matrices WQ, WK, WV with random values
        self.WQ = nn.Linear(input_dim, output_dim)
        self.WK = nn.Linear(input_dim, output_dim)
        self.WV = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.masked = masked


    def __call__(self, x, y=None):
        q = self.WQ(x)
        k = self.WK(x)
        v = self.WV(x)
        scores = q @ k.transpose(1, 2) / np.sqrt(self.dk)
        if self.masked:
            mask = torch.tril(torch.ones(x.size(1), x.size(1))).to(x.device)
            scores = scores.masked_fill(mask==0, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        y = attention @ v
        return y

