import math
from typing import Optional

from torch import nn
import torch

from modules.embedding import RotaryPositionalEmbeddings
from modules.self_attention import SelfAttention


class MultiHeadAttention_Slow(nn.Module):
    def __init__(self, embed_dim, num_heads, is_causal=True, dropout=0):
        super().__init__()
        self.num_heads = num_heads
        self.output_dim = embed_dim
        self.WO = nn.Linear(embed_dim, embed_dim)
        head_dim = embed_dim // num_heads
        self.attentions = nn.ModuleList(
            [
                SelfAttention(embed_dim, head_dim, is_causal=is_causal, dropout=dropout)
                for _ in range(num_heads)
            ]
        )
        self.dropout = nn.Dropout(p=dropout)
        self.is_causal = is_causal

    def __call__(self, x):
        outputs = torch.cat([attention(x) for attention in self.attentions], dim=2)
        return self.dropout(self.WO(outputs))


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        is_causal: Optional[bool] = True,
        num_query_heads_per_key: Optional[int] = None,
        use_efficient: bool = True,
    ):
        super().__init__()
        assert is_causal
        assert d_model % num_heads == 0
        self.is_causal = is_causal
        self.use_efficient = use_efficient
        self.d_model = d_model
        self.num_q_heads = num_heads
        self.d_k = d_model // num_heads
        if num_query_heads_per_key is None:
            # regular multi head attention
            self.num_query_heads_per_key = 1
        else:
            # grouped-query attention
            # set to 1 for traditional multi head attention
            assert num_heads % num_query_heads_per_key == 0
            self.num_query_heads_per_key = num_query_heads_per_key
        # self.linear_q = nn.Linear(d_model, d_model)
        # self.linear_k = nn.Linear(d_model, self.d_k * self.num_query_heads_per_key)
        # self.linear_v = nn.Linear(d_model, self.d_k * self.num_query_heads_per_key)
        self.QKV_sizes = [
            d_model,
            d_model // self.num_query_heads_per_key,
            d_model // self.num_query_heads_per_key,
        ]
        self.linear_qkv = nn.Linear(d_model, sum(self.QKV_sizes))
        self.p_dropout = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.linear_proj = nn.Linear(d_model, d_model)

    def compute_QK(self, Q, K):
        return Q, K

    def forward(self, x, mask=None):
        batch_size, seq_len = x.size(0), x.size(1)

        # Linear projections
        # Q = self.linear_q(x)
        # K = self.linear_k(x)
        # V = self.linear_v(x)
        Q, K, V = self.linear_qkv(x).split(self.QKV_sizes, dim=-1)
        # Split and transpose
        Q = Q.view(batch_size, seq_len, self.num_q_heads, self.d_k).transpose(1, 2)
        num_kv_heads = self.num_q_heads // self.num_query_heads_per_key
        K = K.view(batch_size, seq_len, num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, num_kv_heads, self.d_k).transpose(1, 2)
        if self.num_query_heads_per_key > 1:
            K = K.repeat_interleave(self.num_query_heads_per_key, dim=1)
            V = V.repeat_interleave(self.num_query_heads_per_key, dim=1)
        Q, K = self.compute_QK(Q, K)
        if self.use_efficient:
            y = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, is_causal=self.is_causal, dropout_p=self.p_dropout
            )
        else:
            # Causal mask
            if mask is None:
                mask = (
                    torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
                )  # (1, 1, seq_len, seq_len)
            else:
                mask = mask.unsqueeze(1).unsqueeze(1)  # Broadcast mask to match dimensions
            mask = mask.to(x.device)
            # Scaled dot-product attention
            scores = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
            scores = scores.masked_fill(mask == 0, -float("inf"))  # Apply causal mask
            attention = self.softmax(scores)
            attention = self.dropout(attention)
            y = attention @ V

        # Concatenate and project
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.linear_proj(y)  # Final projection
        return output


class RotaryPEMultiHeadAttention(MultiHeadAttention):
    def __init__(self, d_model: int, num_heads: int, rope_percentage: float = 0.5, **kwargs):
        super().__init__(d_model=d_model, num_heads=num_heads, **kwargs)
        d_rope = int(self.d_k * rope_percentage)
        # not implemented for efficient attention
        assert not kwargs.get("use_efficient", False)
        self.rotary_pe = RotaryPositionalEmbeddings(d_rope)

    def compute_QK(self, query: torch.Tensor, key: torch.Tensor):
        Q = self.rotary_pe(query)
        K = self.rotary_pe(key)
        return Q, K
