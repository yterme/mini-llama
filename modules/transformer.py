import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .multi_head import MultiHeadAttention, RotaryPEMultiHeadAttention


from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from modules.embedding import PositionalEmbedding


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-6
    norm: str = "rms"
    activation: str = "swiglu"
    proba_dropout: float = 0.01
    use_rope_embeddings: bool = True
    n_query_heads_per_key: Optional[int] = None

    max_batch_size: int = 32
    max_seq_len: int = 1024


class Transformer(nn.Module):

    def __init__(
        self,
        params: ModelArgs,
    ):
        super().__init__()
        # self.max_seq_len = params.max_seq_len
        # self.pad_token = params.pad_token
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        if params.use_rope_embeddings:
            # identity - embeddings are computed in the multi head attention layer
            self.pos_embedding = nn.Identity()
        else:
            self.pos_embedding = PositionalEmbedding(params.dim, params.max_seq_len)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    dim=params.dim,
                    n_heads=params.n_heads,
                    norm=params.norm,
                    p_dropout=params.proba_dropout,
                    multiple_of=params.multiple_of,
                    use_rope_embeddings=params.use_rope_embeddings,
                    n_query_heads_per_key=params.n_query_heads_per_key,
                )
                for _ in range(params.n_layers)
            ]
        )
        self.norm = {
            "rms": RMSNorm(params.dim),
            "layer": nn.LayerNorm(params.dim),
        }[params.norm]

        self.dropout = nn.Dropout(p=params.proba_dropout)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

    def forward(self, x):
        x = self.tok_embeddings(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.output(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(
        self,
        dim,
        n_heads,
        norm="rms",
        p_dropout=0,
        multiple_of: Optional[int] = None,
        n_query_heads_per_key=None,
        use_rope_embeddings=False,
    ) -> None:
        super().__init__()
        if use_rope_embeddings:
            mha_class = RotaryPEMultiHeadAttention
        else:
            mha_class = MultiHeadAttention

        self.attention = mha_class(
            dim,
            n_heads,
            is_causal=True,
            dropout=p_dropout,
            n_query_heads_per_key=n_query_heads_per_key,
        )
        self.attention_norm = {"rms": RMSNorm(dim), "layer": nn.LayerNorm(dim)}[norm]
        self.ffn_norm = {"rms": RMSNorm(dim), "layer": nn.LayerNorm(dim)}[norm]

        hidden_dim = 4 * dim
        self.feed_forward = FeedForward(dim, hidden_dim, multiple_of=multiple_of)
        self.dropout = nn.Dropout(p_dropout)

    def __call__(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if multiple_of is not None:
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):

    def __init__(self, d, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x / (torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True)) + self.eps) * self.weight


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
