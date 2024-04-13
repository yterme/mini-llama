import os
import random
import re
from typing import Optional
import torch


def pad_collate(batch, padding_value):
    batch_pad = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=padding_value
    ).to(torch.int64)
    return batch_pad[:, :-1], batch_pad[:, 1:]


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, sequence_length: Optional[int] = None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sequence = self.dataset[idx]["text"]
        # add start and end tokens
        sequence += self.tokenizer.eos_token
        tokenized_story = self.tokenizer.encode(sequence, return_tensors="pt")[0]
        # extract random sequence of length sequence_length
        if self.sequence_length is not None:
            start = random.randint(0, max(0, len(tokenized_story) - 1 - self.sequence_length))
            tokenized_story = tokenized_story[start : start + self.sequence_length]
        return tokenized_story


class ChatDataset(torch.utils.data.Dataset):
    def __init__(
        self, path, tokenizer, sequence_length, impose_start_line_with_username: bool = True
    ):
        with open(path, "r") as f:
            lines = f.readlines()
        if impose_start_line_with_username:
            usernames = os.environ.get("USERNAMES", "").split(",")
            self.userpattern = re.compile(rf"({'|'.join(usernames)}):")
            self.messages = self.merge_lines_as_messages(lines)
        else:
            self.messages = lines
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.tokenized_messages = self._tokenize_messages()
        self.NEW_MESSAGE_SEP = "\n"

    def merge_lines_as_messages(self, lines):
        # bring lines together
        new_lines = []
        current_line = ""
        for line in lines:
            # if match self.userpattern, then it's a new line
            if self.userpattern.match(line):
                if current_line:
                    new_lines.append(current_line)
                current_line = line
            else:
                current_line += line
        return new_lines

    def _tokenize_messages(self):
        return [
            self.tokenizer.encode(msg, max_length=self.sequence_length, truncation=True)
            for msg in self.messages
        ]

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        # tokenized_story = self.tokenizer.encode(message)
        tokenized_story = self.tokenized_messages[idx]
        i = idx + 1
        while len(tokenized_story) < self.sequence_length + 1 and i < len(self.messages):
            # message = self.lines[i]
            tokenized_story += self.tokenizer.encode(self.NEW_MESSAGE_SEP)
            # tokenized_story += self.tokenizer.encode(message)[: self.sequence_length + 1 - len(tokenized_story)]
            tokenized_story += self.tokenized_messages[i][
                : self.sequence_length + 1 - len(tokenized_story)
            ]
            i += 1
        # padding
        if len(tokenized_story) < self.sequence_length + 1:
            assert i == len(self.messages)
        return torch.tensor(tokenized_story[: self.sequence_length])
