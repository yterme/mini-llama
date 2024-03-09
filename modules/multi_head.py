import math
from typing import Optional

from torch import nn
import torch

from modules.self_attention import SelfAttention


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
        x1, x2 = x[..., :self.d], x[..., self.d:]

        other_half = torch.cat([-x1[..., self.d//2:], x1[..., :self.d//2]], dim=-1)
        x1 = x1 * self.cos_pe[:, :, :seq_len] +  other_half * self.sin_pe[:, :, :seq_len]
        return torch.cat((x1, x2), dim=-1)
    


class MultiHeadAttention_Slow(nn.Module):
    def __init__(self, embed_dim, num_heads, masked=True, dropout=0):
        super().__init__()
        self.num_heads = num_heads
        self.output_dim = embed_dim
        self.WO = nn.Linear(embed_dim, embed_dim)
        head_dim = embed_dim // num_heads
        self.attentions = nn.ModuleList([SelfAttention(embed_dim, head_dim, masked=masked, dropout=dropout) for _ in range(num_heads)])
        self.dropout = nn.Dropout(p=dropout)
        self.masked = masked

    def __call__(self, x):
        outputs = torch.cat([attention(x) for attention in self.attentions], dim=2)
        return self.dropout(self.WO(outputs))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, masked: Optional[int]=True, num_query_heads_per_key: Optional[int] = None):
        super().__init__()
        assert masked
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        if num_query_heads_per_key is None:
            # multi-query attention
            self.num_query_heads_per_key = num_heads
        else:
            # grouped-query attention
            # set to 1 for traditional multi head attention
            assert num_heads % num_query_heads_per_key == 0
            self.num_query_heads_per_key = num_query_heads_per_key
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, self.d_k * self.num_query_heads_per_key)
        self.linear_v = nn.Linear(d_model, self.d_k * self.num_query_heads_per_key)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.linear_proj = nn.Linear(d_model, d_model)

    def compute_scores(self, Q, K):
        return (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))

    def forward(self, x, mask=None):
        batch_size, seq_len = x.size(0), x.size(1)

        # Linear projections
        Q = self.linear_q(x)
        K = self.linear_k(x)
        V = self.linear_v(x)

        # Split and transpose
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        num_key_value_heads = self.num_heads // self.num_query_heads_per_key
        K = K.view(batch_size, seq_len, num_key_value_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, num_key_value_heads, self.d_k).transpose(1, 2)
        if num_key_value_heads > 1:
            K = K.repeat_interleave(self.num_query_heads_per_key, dim=1)
            V = V.repeat_interleave(self.num_query_heads_per_key, dim=1)


        # Causal mask
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        else:
            mask = mask.unsqueeze(1).unsqueeze(1)  # Broadcast mask to match dimensions

        mask = mask.to(x.device)
        # Scaled dot-product attention
        scores = self.compute_scores(Q, K)
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
        self.query_rotary_pe = RotaryPositionalEmbeddings(d_rope)
        self.key_rotary_pe = RotaryPositionalEmbeddings(d_rope)

    def compute_scores(self, query: torch.Tensor, key: torch.Tensor):
        Q = self.query_rotary_pe(query)
        K = self.key_rotary_pe(key)
        return (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
