from torch import nn
import torch
from .self_attention import MultiHeadAttention

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        

    def __call__(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.linear(self.norm2(x)) + self.bias
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, embed_dim, num_heads, masked=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        

    def __call__(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.linear(self.norm2(x)) + self.bias
        return x

    # def generate(self, x, max_length=50):   
    #     for _ in range(max_length):
    #         x = x + self.attention(self.norm1(x))
    #         x = x + self.linear(self.norm2(x)) + self.bias
    #     return x