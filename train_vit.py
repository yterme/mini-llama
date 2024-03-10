import argparse

import torch
from modules.vit import ViT
import torchvision

# import datasets
# import lighting trainer
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    train_dataset = torchvision.datasets.MNIST(
        root=".", download=True, train=True, transform=torchvision.transforms.ToTensor()
    )
    test_dataset = torchvision.datasets.MNIST(
        root=".", download=True, train=False, transform=torchvision.transforms.ToTensor()
    )
    image_shape = (28, 28)
    num_channels = 1
    patch_size = 4
    num_classes = 10
    num_heads = 4
    embed_dim = 128
    num_layers = 6
    vit_model = ViT(
        num_layers=num_layers,
        image_shape=image_shape,
        patch_size=patch_size,
        num_heads=num_heads,
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_channels=num_channels,
    )
    # pytorch lightning model checkpoint
    callbacks = [ModelCheckpoint(monitor="val_acc", save_top_k=1, mode="max")]
    trainer = Trainer(
        accelerator="cpu", check_val_every_n_epoch=1, callbacks=callbacks, max_epochs=10
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    trainer.fit(vit_model, train_dataloader, val_dataloader)
    # save model
    torch.save(vit_model.state_dict(), "vit_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT")
    args = parser.parse_args()
    main(**vars(args))
