import torch
from torch import nn
from .self_attention import MultiHeadAttention
import math


class RMSNorm(nn.Module):
    def __init__(self, d) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x / (torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True)) + 1e-8) * self.scale

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, norm = "rms") -> None:
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, embed_dim, num_heads)
        self.norm1 = {
            "rms": RMSNorm(embed_dim),
            "layer": nn.LayerNorm(embed_dim)
        }[norm]
        self.norm2 = {
            "rms": RMSNorm(embed_dim),
            "layer": nn.LayerNorm(embed_dim)
        }[norm]
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

class Swish(nn.Module):
    def __init__(self, beta=1) -> None:
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(x * self.beta)

class SwiGLU(nn.Module):
    def __init__(self, input_dim, output_dim, beta=1) -> None:
        super().__init__()
        self.swish = Swish(beta)
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.swish(self.linear1(x)) * self.linear2(x)

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, proba_dropout=0.1, activation ="swiglu") -> None:
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, embed_dim, num_heads, masked=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Linear(embed_dim, embed_dim)

        # MLP
        interm_embed_dim = 4 * embed_dim
        if activation == "swiglu":
            self.activation_unit = SwiGLU(embed_dim, interm_embed_dim)
        else:
            self.fc = nn.Linear(embed_dim, interm_embed_dim)
            self.activation  = {
                "gelu": NewGELU(),
                "relu": nn.ReLU()
            }[activation]
            self.activation_unit = lambda x: self.activation(self.fc(x))
        self.proj = nn.Linear(interm_embed_dim, embed_dim)
        self.dropout = nn.Dropout(proba_dropout)
        self.mlpf = lambda x: self.dropout(self.proj(self.activation_unit(x)))

    def __call__(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlpf(self.norm2(x))
        return x

    # def generate(self, x, max_length=50):   
    #     for _ in range(max_length):
    #         x = x + self.attention(self.norm1(x))
    #         x = x + self.linear(self.norm2(x)) + self.bias
    #     return x