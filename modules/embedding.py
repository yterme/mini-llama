import math

import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        pos_enc = self.pe[:seq_len, :].unsqueeze(0)
        x = x + pos_enc
        return x


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, max_len=5000):
        super().__init__()
        self.d = d

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        theta = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        theta_repeat = torch.cat([theta, theta], dim=0)
        self.sin_pe = torch.sin(position * theta_repeat)[None, None, :, :].to("cuda")
        self.cos_pe = torch.cos(position * theta_repeat)[None, None, :, :].to("cuda")

    def forward(self, x: torch.Tensor):
        seq_len = x.size(2)
        x1, x2 = x[..., : self.d], x[..., self.d :]

        other_half = torch.cat([-x1[..., self.d // 2 :], x1[..., : self.d // 2]], dim=-1)
        x1 = x1 * self.cos_pe[:, :, :seq_len] + other_half * self.sin_pe[:, :, :seq_len]
        return torch.cat((x1, x2), dim=-1)