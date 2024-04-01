import torch

import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor

from modules.transformer import Transformer, ModelArgs


class Llama:

    def __init__(
        self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(
        checkpoints_dir: str,
        tokenizer_path: str,
        load_model: bool,
        max_seq_len: int,
        max_batch_size: int,
        # device: str,
    ):
        prev_time = time.time()
        if load_model:
            checkpoints = [
                x for x in sorted(Path(checkpoints_dir).glob("*.pth")) if "copy" not in x.stem
            ]
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            if "rope.freqs" in checkpoint:
                del checkpoint["rope.freqs"]
                torch.save(checkpoint, ckpt_path)
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        from accelerate import init_empty_weights, load_checkpoint_and_dispatch

        with init_empty_weights():
            model = Transformer(model_args)
        if load_model:
            model = load_checkpoint_and_dispatch(
                model, checkpoint=str(ckpt_path), device_map="auto", dtype="bfloat16"
            )
        return Llama(model, tokenizer, model_args)

    def generate_single_greedy(self, tokens):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.tensor(tokens).unsqueeze(0).to(device)
        logits = self.model(x)
        probas = torch.softmax(logits[:, -1], dim=-1)
        next_token = torch.argmax(probas, dim=-1).tolist()[0]
        return next_token

    def generate_greedy(self, x, max_length=50):
        token_ids = self.tokenizer.encode(x)
        for _ in range(max_length):
            new_token_id = self.generate_single_greedy(token_ids)
            token_ids.append(new_token_id)
            if new_token_id == self.tokenizer.eos_id():
                break
        return self.tokenizer.decode(token_ids)
