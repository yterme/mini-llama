import argparse
from datetime import datetime

import yaml
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import load_dataset
from transformers import AutoTokenizer

from modules.data import ChatDataset, TokenizedDataset, pad_collate
from modules.gpt import GPT


def main(
    dataset: str,
    tokenizer_name: str,
    epochs: int,
    check_val_every_n_epoch: int,
    load_ckpt: str = None,
):
    # load yaml config model_config.yaml
    model_config = yaml.load(open("model_config.yaml", "r"), Loader=yaml.FullLoader)
    batch_size = 12
    num_workers = 6
    context_length = 1024

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    pad_token = tokenizer.vocab_size
    vocab_size = 50304
    # vocab_size = tokenizer.vocab_size + 1

    if load_ckpt is not None:
        # load checkpoint
        gpt_model = GPT.load_from_checkpoint(load_ckpt)
    else:
        gpt_model = GPT(
            vocab_size=vocab_size,
            pad_token=pad_token,
            context_length=context_length,
            **model_config,
        )
    # pytorch lightning model checkpoint
    callbacks = [
        ModelCheckpoint(monitor="val_acc", save_top_k=1, mode="max"),
        ModelCheckpoint(every_n_train_steps=1000)
    ]
    trainer = Trainer(
        accelerator="gpu",
        check_val_every_n_epoch=check_val_every_n_epoch,
        callbacks=callbacks,
        max_epochs=epochs,
    )

    # huggingface tinystories dataset
    if dataset == "chat":
        train_dataset = ChatDataset(
            "data/_chat_cleaned.txt", tokenizer=tokenizer, sequence_length=context_length + 1
        )
    elif dataset in ["tinystories", "french"]:
        if dataset == "tinystories":
            train_dataset = load_dataset("roneneldan/TinyStories", split="train")
            val_dataset = load_dataset("roneneldan/TinyStories", split="validation")
        elif dataset == "french":
            train_dataset = load_dataset("OpenLLM-France/Claire-Dialogue-French-0.1", split="train")
            val_dataset = load_dataset("OpenLLM-France/Claire-Dialogue-French-0.1", split="test")
        train_dataset = TokenizedDataset(
            train_dataset,
            tokenizer,
            sequence_length=context_length + 1,
        )
        val_dataset = TokenizedDataset(
            val_dataset,
            tokenizer,
            sequence_length=context_length + 1,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    collate_fn = lambda x: pad_collate(x, padding_value=pad_token)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    trainer.fit(gpt_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    # model_name: use date and time
    model_name = f"gpt_model_{dataset}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.pth"
    # save model
    torch.save(gpt_model.state_dict(), model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT")
    parser.add_argument("--dataset", type=str, default="tinystories", help="Dataset to use")
    parser.add_argument("--tokenizer-name", type=str, default="gpt2", help="Tokenizer to use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument(
        "--check-val-every-n-epoch",
        type=int,
        default=1,
        help="Check validation set every n epochs",
    )
    parser.add_argument("--load-ckpt", type=str, default=None, help="Path to checkpoint to load")
    args = parser.parse_args()
    main(**vars(args))
