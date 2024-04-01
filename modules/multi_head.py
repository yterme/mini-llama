import math
from typing import Optional

from torch import nn
import torch

from modules.embedding import RotaryPositionalEmbeddings
from modules.self_attention import SelfAttention


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.1,
        is_causal: Optional[bool] = True,
        n_query_heads_per_key: Optional[int] = None,
        use_efficient: bool = True,
    ):
        super().__init__()
        assert is_causal
        assert dim % num_heads == 0
        self.is_causal = is_causal
        self.use_efficient = use_efficient
        self.dim = dim
        self.n_q_heads = num_heads
        self.d_kv = dim // num_heads
        if n_query_heads_per_key is None:
            # regular multi head attention
            self.n_query_heads_per_kv = 1
        else:
            # grouped-query attention
            # set to 1 for traditional multi head attention
            assert num_heads % n_query_heads_per_key == 0
            self.n_query_heads_per_kv = n_query_heads_per_key
        self.n_kv_heads = self.n_q_heads // self.n_query_heads_per_kv
        self.wq = nn.Linear(dim, self.d_kv * self.n_q_heads, bias=False)
        self.wk = nn.Linear(dim, self.d_kv * self.n_kv_heads, bias=False)
        self.wv = nn.Linear(dim, self.d_kv * self.n_kv_heads, bias=False)
        self.p_dropout = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.wo = nn.Linear(dim, dim, bias=False)

    def compute_QK(self, Q, K):
        return Q, K

    def forward(self, x, mask=None):
        batch_size, seq_len = x.size(0), x.size(1)

        # Linear projections
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        # Split and transpose
        Q = Q.view(batch_size, seq_len, self.n_q_heads, self.d_kv).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_kv_heads, self.d_kv).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_kv_heads, self.d_kv).transpose(1, 2)
        if self.n_query_heads_per_kv > 1:
            K = K.repeat_interleave(self.n_query_heads_per_kv, dim=1)
            V = V.repeat_interleave(self.n_query_heads_per_kv, dim=1)
        Q, K = self.compute_QK(Q, K)
        if self.use_efficient:
            y = torch.nn.functional.scaled_dot_product_attention(
                Q, K, V, is_causal=self.is_causal, dropout_p=self.p_dropout
            )
        else:
            # Causal mask
            if mask is None:
                mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
            else:
                mask = mask.unsqueeze(1).unsqueeze(1)  # Broadcast mask to match dimensions
            mask = mask.to(x.device)
            # Scaled dot-product attention
            scores = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_kv))
            scores = scores.masked_fill(mask == 0, -float("inf"))  # Apply causal mask
            attention = self.softmax(scores)
            attention = self.dropout(attention)
            y = attention @ V

        # Concatenate and project
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.wo(y)  # Final projection
        return output


class RotaryPEMultiHeadAttention(MultiHeadAttention):

    def __init__(self, dim: int, num_heads: int, rope_percentage: float = 0.5, **kwargs):
        super().__init__(dim=dim, num_heads=num_heads, **kwargs)
        d_rope = int(self.d_kv * rope_percentage)
        # not implemented for efficient attention
        assert not kwargs.get("use_efficient", False)
        self.rotary_pe = RotaryPositionalEmbeddings(d_rope)

    def compute_QK(self, query: torch.Tensor, key: torch.Tensor):
        Q = self.rotary_pe(query)
        K = self.rotary_pe(key)
        return Q, K


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
