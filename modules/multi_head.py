import math

from torch import nn
import torch

from modules.self_attention import SelfAttention


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
    def __init__(self, d_model, num_heads, dropout=0.1, masked=True):
        super().__init__()
        assert masked
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.linear_proj = nn.Linear(d_model, d_model)

    def forward(self, query, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)

        # Linear projections
        Q = self.linear_q(query)
        K = self.linear_k(query)
        V = self.linear_v(query)

        # Split and transpose
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Causal mask
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        else:
            mask = mask.unsqueeze(1).unsqueeze(1)  # Broadcast mask to match dimensions

        mask = mask.to(query.device)
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