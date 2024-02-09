from torch import nn
import numpy as np
import torch


class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim, masked=False):
        super().__init__()
        self.dk = output_dim
        # self.WQ = nn.Linear(input_dim, output_dim)
        # self.WK = nn.Linear(input_dim, output_dim)
        # self.WV = nn.Linear(input_dim, output_dim)
        # initialize matrices WQ, WK, WV with random values
        self.WQ = nn.Parameter(torch.randn(input_dim, output_dim))
        self.WK = nn.Parameter(torch.randn(input_dim, output_dim))
        self.WV = nn.Parameter(torch.randn(input_dim, output_dim))
        self.softmax = nn.Softmax(dim=1)
        self.masked = masked
        if self.masked:
            self.WQ = nn.Parameter(torch.tril(self.WQ))
            self.WK = nn.Parameter(torch.tril(self.WK))
            self.WV = nn.Parameter(torch.tril(self.WV))
            # self.dk = ?
        # self.encoder_decoder = encoder_decoder

    def __call__(self, x, y=None):
        # assert self.encoder_decoder == (y is not None)
        q = x @ self.WQ
        k = x @ self.WK
        v = x @ self.WV
        return self.softmax(q @ torch.transpose(k, 1, 2) / np.sqrt(self.dk)) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, masked=False):
        super().__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.WO = nn.Parameter(torch.randn(num_heads * output_dim, output_dim))
        self.attention = SelfAttention(input_dim, output_dim, masked=masked)
        self.masked = masked

    def __call__(self, x):
        outputs = torch.cat([self.attention(x) for _ in range(self.num_heads)], dim=2)
        return outputs @ self.WO
