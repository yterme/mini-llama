import torch
from torch import nn
from .multi_head import MultiHeadAttention, RotaryPEMultiHeadAttention
import math


class RMSNorm(nn.Module):
    def __init__(self, d) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x / (torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + 1e-6) * self.scale


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, norm="rms", dropout=0.0) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = {"rms": RMSNorm(embed_dim), "layer": nn.LayerNorm(embed_dim)}[norm]
        self.norm2 = {"rms": RMSNorm(embed_dim), "layer": nn.LayerNorm(embed_dim)}[norm]
        self.linear = nn.Linear(embed_dim, embed_dim)

    def __call__(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.linear(self.norm2(x))
        return x


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        )


class SwiGLU(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.swish = torch.nn.SiLU()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.swish(self.linear1(x)) * self.linear2(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0,
        norm="rms",
        activation="swiglu",
        num_query_heads_per_key=None,
        rope=False,
    ) -> None:
        super().__init__()
        if rope:
            mha_class = RotaryPEMultiHeadAttention
        else:
            mha_class = MultiHeadAttention

        self.attention = mha_class(
            embed_dim,
            num_heads,
            is_causal=True,
            dropout=dropout,
            num_query_heads_per_key=num_query_heads_per_key,
        )
        self.norm1 = {"rms": RMSNorm(embed_dim), "layer": nn.LayerNorm(embed_dim)}[norm]
        self.norm2 = {"rms": RMSNorm(embed_dim), "layer": nn.LayerNorm(embed_dim)}[norm]

        # MLP
        hidden_dim = 4 * embed_dim
        if activation == "swiglu":
            self.activation_unit = SwiGLU(embed_dim, hidden_dim)
        else:
            self.fc = nn.Linear(embed_dim, hidden_dim)
            self.activation = {"gelu": NewGELU(), "relu": nn.ReLU()}[activation]
            self.activation_unit = lambda x: self.activation(self.fc(x))
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = lambda x: self.dropout(self.proj(self.activation_unit(x)))

    def __call__(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
