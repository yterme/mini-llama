import json
from pathlib import Path
from typing import Any
import json

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from sentencepiece import SentencePieceProcessor

from modules.transformer import Transformer, ModelArgs



class LanguageModel(LightningModule):

    def __init__(
        self,
        model: torch.nn.Module,
        gradient_clip: float = 1.0,
        pad_token: int = 0,
        context_length: int = 1024,
        lr: float = 1e-3,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model"])

        self.model = model

        self.gradient_clip = gradient_clip
        self.pad_token = pad_token
        self.context_length = context_length
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def compute_metrics(self, batch) -> torch.Tensor:
        inputs, target = batch
        output = self.forward(inputs).transpose(1, 2)
        loss = torch.nn.functional.cross_entropy(output, target, ignore_index=self.pad_token)
        # accuracy
        preds = torch.argmax(output, dim=1)
        correct_mask = (preds == target)[target != self.pad_token]
        correct_sum = correct_mask.sum()
        total = correct_mask.size(0)
        acc = correct_sum / total
        return loss, acc

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        opt = self.optimizers()
        opt.zero_grad()
        loss, acc = self.compute_metrics(batch)
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
        loss, acc = self.compute_metrics(batch)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return super().train_dataloader()

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.lr)
