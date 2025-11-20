import os
import random
import re
from typing import Optional
import torch
from modules.constants import CHAT_USERNAMES


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
        tokenized_story = self.tokenizer.encode(sequence)
        # extract random sequence of length sequence_length
        if self.sequence_length is not None:
            start = random.randint(0, max(0, len(tokenized_story) - 1 - self.sequence_length))
            tokenized_story = tokenized_story[start : start + self.sequence_length]
        return torch.tensor(tokenized_story)


def count_messages_in_file(path, impose_start_line_with_username=True):
    """Count number of messages without loading everything into memory."""
    with open(path, 'r') as f:
        lines = f.readlines()
    
    if impose_start_line_with_username:
        # Use predefined usernames from constants
        usernames = CHAT_USERNAMES
        
        if not usernames:
            # No usernames set, treat each line as a message
            return len([l for l in lines if l.strip()])
        
        userpattern = re.compile(rf"({'|'.join(re.escape(u) for u in usernames)}):")
        count = 0
        current_line = ""
        for line in lines:
            if userpattern.match(line):
                if current_line:
                    count += 1
                current_line = line
            else:
                current_line += line
        if current_line:
            count += 1
        return count
    else:
        return len(lines)


class ChatDataset(torch.utils.data.Dataset):
    def __init__(
        self, path, tokenizer, sequence_length, impose_start_line_with_username: bool = True,
        sample_indices: Optional[list] = None
    ):
        """
        Args:
            path: Path to dataset file
            tokenizer: Tokenizer to use
            sequence_length: Context length in TOKENS (e.g., 1024 tokens)
                            Each item returned will have (sequence_length + 1) tokens
                            pad_collate splits this into input (seq_len) and target (seq_len)
            impose_start_line_with_username: Whether to merge lines by username pattern
            sample_indices: If provided, only load and tokenize these specific line indices
                           (for memory-efficient sampling during callback)
        """
        with open(path, "r") as f:
            lines = f.readlines()
        
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.NEW_MESSAGE_SEP = "\n"
        
        # Merge lines into messages
        if impose_start_line_with_username:
            # Use predefined usernames from constants
            usernames = CHAT_USERNAMES
            
            if usernames:
                # Use username pattern if usernames are provided
                self.userpattern = re.compile(rf"({'|'.join(re.escape(u) for u in usernames)}):")
                
                # If sample_indices provided, they are LINE indices (for truly random sampling)
                # So we filter lines first, then merge
                if sample_indices is not None:
                    filtered_lines = [lines[i] for i in sample_indices if i < len(lines)]
                    all_messages = self.merge_lines_as_messages(filtered_lines)
                else:
                    # No sampling, merge all lines into messages
                    all_messages = self.merge_lines_as_messages(lines)
            else:
                # No usernames set, treat each line as a message
                all_messages = lines
                # Filter to sample indices if provided
                if sample_indices is not None:
                    all_messages = [all_messages[i] for i in sample_indices if i < len(all_messages)]
        else:
            # When not using username pattern, treat each line as a message
            all_messages = lines
            # Filter to sample indices if provided (these are MESSAGE indices)
            if sample_indices is not None:
                all_messages = [all_messages[i] for i in sample_indices if i < len(all_messages)]
        
        # Note: all_messages is already filtered by sample_indices if provided,
        # so self.messages is the subset we need (not all lines when sampling)
        self.messages = all_messages
        
        # Concatenate all messages into one long text
        self.full_text = "".join(self.messages)
        
        # Tokenize the ENTIRE concatenated text once
        # This gives us TOKEN indices that we can slide over
        self.full_text_tokens = self.tokenizer.encode(self.full_text)

    def merge_lines_as_messages(self, lines):
        """Merge lines into messages based on username pattern.
        
        Args:
            lines: List of raw lines from file (may be pre-filtered for sampling)
        
        Returns:
            List of merged messages
        """
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
        
        # Don't forget the last message
        if current_line:
            new_lines.append(current_line)
        
        return new_lines

    def __len__(self):
        # Return number of possible token windows
        # We need sequence_length + 1 tokens total (for input + target in pad_collate)
        if len(self.full_text_tokens) < self.sequence_length + 1:
            return 1
        return len(self.full_text_tokens) - self.sequence_length

    def __getitem__(self, idx):
        # Get a sliding window of (sequence_length + 1) tokens
        # pad_collate will split this into input[:-1] and target[1:]
        start_token = idx
        end_token = start_token + self.sequence_length + 1
        
        # Extract token window
        token_window = self.full_text_tokens[start_token:end_token]
        
        # Pad if we're at the end of the text
        if len(token_window) < self.sequence_length + 1:
            pad_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            token_window = token_window + [pad_token] * (self.sequence_length + 1 - len(token_window))
        
        return torch.tensor(token_window, dtype=torch.int64)
