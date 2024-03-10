import math
from typing import Any
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from .transformer import TransformerEncoderBlock
from .tokenizer import ViTTokenizer
from torch import nn


class ViT(LightningModule):
    def __init__(
        self, patch_size, image_shape, num_layers, num_heads, embed_dim, num_classes, num_channels
    ):
        super().__init__()
        self.image_shape = image_shape
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        self.init_tokenizer(patch_size, image_shape, num_channels)
        self.layers = [TransformerEncoderBlock(embed_dim, num_heads) for _ in range(num_layers)]
        self.num_patches = math.ceil(image_shape[0] / patch_size) * math.ceil(
            image_shape[1] / patch_size
        )
        self.mlp_head = nn.Parameter(torch.randn(embed_dim * self.num_patches, num_classes))
        self.mlp_head_bias = nn.Parameter(torch.zeros(num_classes))

    def init_tokenizer(self, patch_size, image_shape, num_channels):
        self.tokenizer = ViTTokenizer(
            embed_dim=self.embed_dim,
            image_shape=image_shape,
            patch_size=patch_size,
            num_channels=num_channels,
        )

    def __call__(self, x):
        assert x.shape[-2:] == self.image_shape
        assert x.shape[1] == self.num_channels
        assert len(x.shape) == 4
        x = self.tokenizer(x)
        # print(x.shape)
        for layer in self.layers:
            x = layer(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], -1)
        # print(x.shape)
        output = x @ self.mlp_head + self.mlp_head_bias
        # print(output.shape)
        return output

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = torch.nn.functional.cross_entropy(output, target.view(-1))
        # accuracy
        preds = torch.argmax(output, dim=1)
        correct = (preds == target).sum().item()
        total = target.size(0)
        acc = correct / total
        # log accuracy in progress bar
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # log loss in progress bar
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = torch.nn.functional.cross_entropy(output, target.view(-1))
        # accuracy
        preds = torch.argmax(output, dim=1)
        correct = (preds == target).sum().item()
        total = target.size(0)
        acc = correct / total
        # log accuracy in progress bar
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # log loss in progress bar
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return super().train_dataloader()

    def configure_optimizers(self) -> Any:
        # return super().configure_optimizers()
        # adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
