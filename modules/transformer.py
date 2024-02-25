import torch
from torch import nn
from .self_attention import MultiHeadAttention
import math

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
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
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))



class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, proba_dropout=0.1) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, embed_dim, num_heads, masked=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

        # MLP
        self.fc = nn.Linear(embed_dim, 4 * embed_dim)
        # self.activation  = nn.ReLU()
        self.activation  = NewGELU()
        self.proj = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(proba_dropout)
        self.mlpf = lambda x: self.dropout(self.proj(self.activation(self.fc(x))))

    def __call__(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlpf(self.norm2(x))
        return x

    # def generate(self, x, max_length=50):   
    #     for _ in range(max_length):
    #         x = x + self.attention(self.norm1(x))
    #         x = x + self.linear(self.norm2(x)) + self.bias
    #     return x