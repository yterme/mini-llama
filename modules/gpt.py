from typing import Any

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from torch import nn

from modules.embedding import PositionalEmbedding
from modules.transformer import DecoderBlock, RMSNorm


def generate_single_greedy(model, tokenizer, x):
    x = tokenizer.encode(x)
    x = x[-model.context_length :]
    i = len(x)
    y = model._predict_probas(x)
    y = torch.argmax(y, dim=1)
    y_next = y[i - 1]
    return tokenizer.decode(y_next)

def generate_greedy(model, tokenizer, x, max_length=50):
    for _ in range(max_length):
        new_token = generate_single_greedy(model, tokenizer, x)
        x += new_token
        if new_token == tokenizer.eos_token:
            break
    return x

class GPT(LightningModule):
    def __init__(
        self,
        num_layers,
        num_heads,
        d_model,
        context_length,
        gradient_clip,
        vocab_size,
        pad_token,
        norm="rms",
        activation="relu",
        proba_dropout=0.01,
        rope_embeddings=False,
        num_query_heads_per_key=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.gradient_clip = gradient_clip
        self.d_model = d_model
        self.pad_token = pad_token
        vocab_size = vocab_size
        self.context_length = context_length
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        if rope_embeddings:
            # identity - embeddings are computed in the multi head attention layer
            self.pos_embedding = nn.Identity()
        else:
            self.pos_embedding = PositionalEmbedding(d_model, context_length)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    num_heads,
                    norm=norm,
                    dropout=proba_dropout,
                    activation=activation,
                    rope=rope_embeddings,
                    num_query_heads_per_key=num_query_heads_per_key,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(p=proba_dropout)
        self.norm = {"rms": RMSNorm(d_model), "layer": nn.LayerNorm(d_model)}[norm]
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.text_embedding(x)
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
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.manual_backward(loss)
        opt.step()

    def manual_backward(self, loss: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        loss.backward(*args, **kwargs)
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)
        return

    def validation_step(self, batch, batch_idx):
        loss, acc = self.compute_loss(batch)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return super().train_dataloader()

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=1e-3)
