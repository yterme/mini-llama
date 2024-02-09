import math
from typing import Any
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from .transformer import DecoderBlock
from .tokenizer import GPTTokenizer
from torch import nn

from transformers import AutoTokenizer


def generate_next_max_likelihood(model, tokenizer, x):
    x = tokenizer.encode(x)
    i = len(x)
    y = model._predict_probas(x)
    y = torch.argmax(y, dim=1)
    y = tokenizer.decode(y)
    return y[i - 1]


def generate(model, tokenizer, x, max_length=50):
    # x = self.tokenizer.encode(x)
    # x = x + self.pos_embedding
    for _ in range(max_length):
        new_token = generate_next_max_likelihood(model, tokenizer, x)
        x += new_token
    # x = x.reshape(x.shape[0], -1)
    # output = x @ self.mlp_head + self.mlp_head_bias
    return x


class GPT2(LightningModule):
    def __init__(self, num_layers, num_heads, embed_dim, context_length):
        super().__init__()
        self.init_tokenizer()
        self.embed_dim = embed_dim
        self.context_length = context_length
        self.layers = [DecoderBlock(embed_dim, num_heads) for _ in range(num_layers)]
        self.context_length = context_length
        self.pos_embedding = nn.Parameter(torch.randn(self.context_length, embed_dim))
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, embed_dim)

    def init_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def __call__(self, x):
        # pad sequence to context length
        y = self.embedding(x)
        y += self.pos_embedding
        # add batch dimension
        # if len(x.shape) == 2:
        #     y = x.unsqueeze(0)
        # print(y.shape)
        for layer in self.layers:
            y = layer(y)
        # print(y.shape)
        # x = x.reshape(x.shape[0], -1)
        # print(x.shape)
        # output = x @ self.mlp_head + self.mlp_head_bias
        # print(output.shape)
        y = torch.softmax(y @ self.embedding.weight.T, dim=2)
        return y

    def _predict_probas(self, x):
        assert len(x) <= self.context_length
        x = x + [0] * (self.context_length - len(x))
        x = torch.tensor(x).unsqueeze(0)
        y = self(x)[0]
        # y = self.tokenizer.decode(y)
        return y

    def generate_next_max_likelihood(self, x):
        x = self.tokenizer.encode(x)
        i = len(x)
        y = self._predict_probas(x)
        y = torch.argmax(y, dim=1)
        y = self.tokenizer.decode(y)
        return y[i - 1]

    def generate(self, x, max_length=50):
        # x = self.tokenizer.encode(x)
        # x = x + self.pos_embedding
        for _ in range(max_length):
            new_token = self.generate_next_max_likelihood(x)
            x += new_token
        # x = x.reshape(x.shape[0], -1)
        # output = x @ self.mlp_head + self.mlp_head_bias
        return x

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs).transpose(1, 2)
        loss = torch.nn.functional.cross_entropy(output, target)
        # accuracy
        preds = torch.argmax(output, dim=1)
        correct = (preds == target).sum().item()
        total = target.size(0)
        acc = correct / total
        # log accuracy in progress bar
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # log loss in progress bar
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs).transpose(1, 2)
        loss = torch.nn.functional.cross_entropy(output, target)
        # accuracy
        preds = torch.argmax(output, dim=1)
        correct = (preds == target).sum().item()
        total = target.size(0)
        acc = correct / total
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
