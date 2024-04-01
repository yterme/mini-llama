import argparse
import torch

import torch
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor

from modules.transformer import Transformer, ModelArgs

from accelerate import init_empty_weights, load_checkpoint_and_dispatch


class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def build(
        checkpoint_dir: str,
        tokenizer_path: str,
    ):
        checkpoints = [
            x for x in sorted(Path(checkpoint_dir).glob("*.pth")) if "copy" not in x.stem
        ]
        assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoint_dir}"
        ckpt_path = checkpoints[0]
        # checkpoint = torch.load(ckpt_path, map_location="cpu")
        # if "rope.freqs" in checkpoint:
        #     del checkpoint["rope.freqs"]
        #     torch.save(checkpoint, ckpt_path)
        with open(Path(checkpoint_dir, "params.json"), "r") as f:
            params = json.loads(f.read())

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)

        model_args = ModelArgs(**params)
        model_args.vocab_size = tokenizer.vocab_size()

        # use huggingface accelerate to load the model as bfloat16
        with init_empty_weights():
            model = Transformer(model_args)
        model = load_checkpoint_and_dispatch(
            model, checkpoint=str(ckpt_path), device_map="auto", dtype="bfloat16"
        )
        return LLaMA(model, tokenizer)

    def generate_single_greedy(self, tokens):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.tensor(tokens).unsqueeze(0).to(device)
        logits = self.model(x)
        probas = torch.softmax(logits[:, -1], dim=-1)
        next_token = torch.argmax(probas, dim=-1).tolist()[0]
        return next_token

    def generate_greedy(self, x, max_length=50):
        token_ids = self.tokenizer.encode(x)
        token_ids.insert(0, self.tokenizer.bos_id())
        for _ in range(max_length):
            new_token_id = self.generate_single_greedy(token_ids)
            token_ids.append(new_token_id)
            if new_token_id == self.tokenizer.eos_id():
                break
        return self.tokenizer.decode(token_ids)


def main(checkpoint_dir, tokenizer_path):
    llama = LLaMA.build(
        checkpoint_dir=checkpoint_dir,
        tokenizer_path=tokenizer_path,
    )

    sentences = [
        "Who was the third president of the United States?\n",
        "What is the capital of France?\n",
        "How long did the Hundred Years' War last?\n",
    ]
    for sentence in sentences:
        print(llama.generate_greedy(sentence))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="llama-2-7b-chat")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer.model")
    args = parser.parse_args()
    main(**vars(args))
