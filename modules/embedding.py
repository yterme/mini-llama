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


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # As written in the paragraph 3.2.2 of the paper
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)
    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

class RotaryPositionalEmbeddings(nn.Module):

    def __init__(self, dim: int, max_len=2 * 4096, theta=10_000):
        super().__init__()
        self.dim = dim

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        theta = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

        theta_repeat = torch.cat([theta, theta], dim=0)
        sin_pe = torch.sin(position * theta_repeat)[None, None, :, :]
        cos_pe = torch.cos(position * theta_repeat)[None, None, :, :]
        self.register_buffer("sin_pe", sin_pe)
        self.register_buffer("cos_pe", cos_pe)

    def forward(self, x: torch.Tensor):
        x1, x2 = x[..., : self.dim], x[..., self.dim :]
        other_half = torch.cat([-x1[..., self.dim // 2 :], x1[..., : self.dim // 2]], dim=-1)
        seq_len = x.size(2)
        x1 = x1 * self.cos_pe[:, :, :seq_len] + other_half * self.sin_pe[:, :, :seq_len]
        return torch.cat((x1, x2), dim=-1).type_as(x)
