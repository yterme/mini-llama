import argparse
from datetime import datetime
import random

import torch
from modules.gpt import GPT

# import datasets
# import lighting trainer
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import load_dataset
from transformers import AutoTokenizer

# # new dataloader returns (sequence, shfited_sequence)
# class TinyStoriesDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         text = self.dataset[idx]["text"]
#         shifted_text =


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        tokenized_text = self.tokenizer.encode(text)
        return tokenized_text


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, sequence_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sequence = self.dataset[idx]["text"]
        tokenized_story = self.tokenizer.encode(sequence)
        # padding
        tokenized_story = tokenized_story + [self.tokenizer.vocab_size] * (
            self.sequence_length + 1 - len(tokenized_story)
        )
        # extract random sequence of length sequence_length
        # start = random.randint(0, len(tokenized_story) - 1 - self.sequence_length)
        start = 0
        # Extract input and target sequences
        input_sequence = torch.tensor(
            tokenized_story[start : start + self.sequence_length]
        )
        target_sequence = torch.tensor(
            tokenized_story[start + 1 : start + self.sequence_length + 1]
        )
        return input_sequence, target_sequence

    # def __getitem__(self, idx):
    #     tokenized_story = self.dataset[idx]["text"]
    #     # tokenized_story = self.tokenizer.encode(text)
    #     # extract random sequence of length sequence_length
    #     start = torch.randint(0, len(tokenized_story) - 1 - self.sequence_length, (1,))
    #     # Extract input and target sequences
    #     input_sequence = torch.tensor(start[start : start + self.sequence_length])
    #     target_sequence = torch.tensor(
    #         start[start + 1 : start + self.sequence_length + 1]
    #     )

    #     return input_sequence, target_sequence


def main():
    # huggingface tinystories dataset
    train_dataset = load_dataset("roneneldan/TinyStories", split="train")
    val_dataset = load_dataset("roneneldan/TinyStories", split="validation")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    num_heads = 4
    embed_dim = 128
    num_layers = 8
    context_length = 150
    gpt_model = GPT(
        num_layers=num_layers,
        num_heads=num_heads,
        embed_dim=embed_dim,
        context_length=context_length,
        tokenizer= tokenizer,
    )
    # gpt2_model.model = gpt2_model.model.to(device="cuda:0")

    # pytorch lightning model checkpoint
    callbacks = [ModelCheckpoint(monitor="val_acc", save_top_k=1, mode="max")]
    trainer = Trainer(
        accelerator="gpu", check_val_every_n_epoch=1, callbacks=callbacks, max_epochs=10,
    )

    train_dataset = TokenizedDataset(
        train_dataset,
        tokenizer,
        sequence_length=context_length,
    )
    val_dataset = TokenizedDataset(
        val_dataset,
        tokenizer,
        sequence_length=context_length,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False,  num_workers=6)
    trainer.fit(
        gpt_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    # save model
    # model_name: use date and time
    model_name = f"gpt_model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
    torch.save(gpt_model.state_dict(), model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT2")
    args = parser.parse_args()
    main(**vars(args))
