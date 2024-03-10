import argparse
from datetime import datetime

import torch
from modules.data import ChatDataset, TokenizedDataset
from modules.gpt import GPT

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import load_dataset
from transformers import AutoTokenizer


def main(dataset: str = "tinystories"):
    config = dict(
        num_heads = 16,
        d_model = 512,
        num_layers = 6,
        context_length=500,
        activation = "swiglu",
        norm = "rms",
        gradient_clip = 1.0,
        proba_dropout = 0.01,
        num_query_heads_per_key = 4,
        rope_embeddings = True,
    )
    batch_size = 16
    num_workers = 6
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    gpt_model = GPT(tokenizer=tokenizer, **config)
    # pytorch lightning model checkpoint
    callbacks = [ModelCheckpoint(monitor="val_acc", save_top_k=1, mode="max")]
    trainer = Trainer(
        accelerator="gpu", check_val_every_n_epoch=1, callbacks=callbacks, max_epochs=10,
    )

    # huggingface tinystories dataset
    if dataset == "tinystories":
        train_dataset = load_dataset("roneneldan/TinyStories", split="train")
        val_dataset = load_dataset("roneneldan/TinyStories", split="validation")
        train_dataset = TokenizedDataset(
            train_dataset,
            tokenizer,
            sequence_length=config["context_length"],
        )
        val_dataset = TokenizedDataset(
            val_dataset,
            tokenizer,
            sequence_length=config["context_length"],
        )
    elif dataset == "chat":
        train_dataset = ChatDataset(
            "data/_chat_cleaned.txt",
            tokenizer=tokenizer,
            sequence_length=config["context_length"]
        )
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,  num_workers=num_workers)
    trainer.fit(
        gpt_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    # model_name: use date and time
    model_name = f"gpt_model_{dataset}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.pth"
    # save model
    torch.save(gpt_model.state_dict(), model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT")
    parser.add_argument("--dataset", type=str, default="tinystories", help="Dataset to use")
    args = parser.parse_args()
    main(**vars(args))
