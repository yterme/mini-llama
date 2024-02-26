import math
from typing import Any
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from .transformer import DecoderBlock
from torch import nn


def generate_next_max_likelihood(model, tokenizer, x):
    x = tokenizer.encode(x)
    i = len(x)
    y = model._predict_probas(x)
    y = torch.argmax(y, dim=1)
    y = tokenizer.decode(y)
    return y[i - 1]


def generate(model, tokenizer, x, max_length=50):
    for _ in range(max_length):
        new_token = generate_next_max_likelihood(model, tokenizer, x)
        x += new_token
    return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.0)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        pos_enc = self.pe[:seq_len, :].unsqueeze(0)
        x = x + pos_enc
        return x


class GPTModel(nn.Module):
    def __init__(self, num_layers, num_heads, embed_dim, context_length, vocab_size, proba_dropout=0.1, gradient_clip=1.0):
        super().__init__()
        self.gradient_clip = gradient_clip
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.pos_embedding = PositionalEncoding(embed_dim, context_length)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([DecoderBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.dropout = nn.Dropout(p=proba_dropout)

    def __call__(self, x):
        x = self.embedding(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln(x)
        x = self.lm_head(x)
        return x
    
    # def _init_weights(self, module):
    #     super()._init_weights(module)
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #     elif isinstance(module, nn.LayerNorm):
    #         torch.nn.init.zeros_(module.bias)
    #         torch.nn.init.ones_(module.weight)

class GPT(LightningModule):
    def __init__(self, num_layers, num_heads, embed_dim, context_length, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.model = GPTModel(num_layers, num_heads, embed_dim, context_length, self.tokenizer.vocab_size + 1)
        self.pad_token = self.tokenizer.vocab_size

    def forward(self, x):
        return self.model(x)

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
        correct = (preds == target).sum().item()
        total = target.size(0) * target.size(1)
        acc = correct / total
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
        # log accuracy in progress bar
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # log loss in progress bar
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return super().train_dataloader()

    def configure_optimizers(self) -> Any:
        # return super().configure_optimizers()
        # adam optimizer
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-5, weight_decay=0.1)
        # cosine annealing learning rate scheduler with linear warmup
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2)
        # return [optimizer], [scheduler]
        return optimizer
