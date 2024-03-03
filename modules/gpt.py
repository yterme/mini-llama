import math
from typing import Any
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from .transformer import DecoderBlock, RMSNorm
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        pos_enc = self.pe[:seq_len, :].unsqueeze(0)
        x = x + pos_enc
        return x


class GPT(LightningModule):
    def __init__(self, tokenizer, num_layers, num_heads, embed_dim, context_length, gradient_clip, norm="rms", activation="relu", proba_dropout=0.01, **kwargs):
        super().__init__(**kwargs)
        self.gradient_clip = gradient_clip
        self.tokenizer = tokenizer
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.pad_token = self.tokenizer.vocab_size
        vocab_size = tokenizer.vocab_size + 1
    
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = PositionalEncoding(embed_dim, context_length)
        self.layers = nn.ModuleList([DecoderBlock(embed_dim, num_heads, norm=norm, dropout=proba_dropout, activation=activation) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=proba_dropout)
        self.norm = {
            "rms": RMSNorm(embed_dim),
            "layer": nn.LayerNorm(embed_dim)
        }[norm]
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # self.apply(self._init_weights)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_embedding(x)
        # x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.lm_head(x)
        return x

        

    def _predict_probas(self, x):
        assert len(x) <= self.context_length
        x = x + [self.pad_token] * (self.context_length - len(x))
        x = torch.tensor(x).unsqueeze(0).to(self.device)
        y = self.forward(x)[0]
        return torch.softmax(y, dim=1)

    def generate_next_max_likelihood(self, x):
        x = x[-self.context_length:]
        x = self.tokenizer.encode(x)
        i = len(x)
        y = self._predict_probas(x)
        y = torch.argmax(y, dim=1)
        y_next = y[i-1]
        return self.tokenizer.decode(y_next)

    def generate(self, x, max_length=50):
        for _ in range(max_length):
            new_token = self.generate_next_max_likelihood(x)
            x += new_token
        return x

    def compute_loss(self, batch) -> torch.Tensor:
        inputs, target = batch
        output = self.forward(inputs).transpose(1, 2)
        loss = torch.nn.functional.cross_entropy(output, target, ignore_index=self.pad_token)
        # accuracy
        preds = torch.argmax(output, dim=1)
        correct_mask = (preds == target)[target != self.pad_token]
        correct_sum = correct_mask.sum().item()
        total = correct_mask.size(0) 
        acc = correct_sum / total
        return loss, acc

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        opt = self.optimizers()
        opt.zero_grad()
        loss, acc = self.compute_loss(batch)
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.manual_backward(loss)
        opt.step()
    
    def manual_backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        loss.backward(*args, **kwargs)
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)
        return
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self.compute_loss(batch)
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return super().train_dataloader()

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=0.001)
