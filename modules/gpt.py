import math
from typing import Any
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from .transformer import DecoderBlock, RMSNorm
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        pos_enc = self.pe[:seq_len, :].unsqueeze(0)
        x = x + pos_enc
        return x


class GPT(LightningModule):
    def __init__(
            self, 
            tokenizer,
            num_layers,
            num_heads,
            d_model,
            context_length,
            gradient_clip,
            norm="rms",
            activation="relu",
            proba_dropout=0.01,
            rope_embeddings=False,
            num_query_heads_per_key=None,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.gradient_clip = gradient_clip
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.context_length = context_length
        self.pad_token = self.tokenizer.vocab_size
        vocab_size = tokenizer.vocab_size + 1
    
        self.d_model = d_model
        self.context_length = context_length
        self.embedding = nn.Embedding(vocab_size, d_model)
        if rope_embeddings:
            # identity - embeddings are computed in the multi head attention layer
            self.pos_embedding = nn.Identity()
        else:
            self.pos_embedding = PositionalEmbedding(d_model, context_length)
        self.layers = nn.ModuleList([
            DecoderBlock(
                d_model, 
                num_heads, 
                norm=norm, 
                dropout=proba_dropout, 
                activation=activation, 
                rope=rope_embeddings,
                num_query_heads_per_key=num_query_heads_per_key
            ) for _ in range(num_layers)])
        self.dropout = nn.Dropout(p=proba_dropout)
        self.norm = {
            "rms": RMSNorm(d_model),
            "layer": nn.LayerNorm(d_model)
        }[norm]
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)
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
