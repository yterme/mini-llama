import math

import torch
from torch import nn


class PositionalEmbedding(nn.Module):

    def __init__(self, dim, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        pos_enc = self.pe[:seq_len, :].unsqueeze(0)
        x = x + pos_enc
        return x


class RotaryPositionalEmbeddings(nn.Module):

    def __init__(self, dim: int, max_len=5000):
        super().__init__()
        self.dim = dim

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        theta = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        theta_repeat = torch.cat([theta, theta], dim=0)
        sin_pe = torch.sin(position * theta_repeat)[None, None, :, :]  # .to("cuda")
        cos_pe = torch.cos(position * theta_repeat)[None, None, :, :]  # .to("cuda")
        self.register_buffer("sin_pe", sin_pe)
        self.register_buffer("cos_pe", cos_pe)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(2)
        x1, x2 = x[..., : self.dim], x[..., self.dim :]

        other_half = torch.cat([-x1[..., self.dim // 2 :], x1[..., : self.dim // 2]], dim=-1)
        x1 = x1 * self.cos_pe[:, :, :seq_len] + other_half * self.sin_pe[:, :, :seq_len]
        return torch.cat((x1, x2), dim=-1).type_as(x)
