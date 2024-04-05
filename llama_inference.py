import argparse
import torch

import torch
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor

from modules.transformer import Transformer, ModelArgs

from accelerate import init_empty_weights, load_checkpoint_and_dispatch


def build_llama_model(
    checkpoint_dir: str,
    vocab_size: int,
) -> Transformer:
    checkpoints = [
        x for x in sorted(Path(checkpoint_dir).glob("*.pth"))
    ]
    assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoint_dir}"
    ckpt_path = checkpoints[0]

    with open(Path(checkpoint_dir, "params.json"), "r") as f:
        params = json.loads(f.read())

    model_args = ModelArgs(**params)
    model_args.vocab_size = vocab_size

    # use huggingface accelerate to load the model as bfloat16 on a 16GB GPU
    with init_empty_weights():
        model = Transformer(model_args)
    print(f"Loading model from {ckpt_path}")
    model = load_checkpoint_and_dispatch(
        model, checkpoint=str(ckpt_path), device_map="auto", dtype="bfloat16"
    )
    return model

def generate_single_greedy(model, tokens):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(tokens).unsqueeze(0).to(device)
    logits = model(x)
    probas = torch.softmax(logits[:, -1], dim=-1)
    next_token = torch.argmax(probas, dim=-1).tolist()[0]
    return next_token

def generate_greedy(model, tokenizer, x, max_length=50):
    token_ids = tokenizer.encode(x)
    token_ids.insert(0, tokenizer.bos_id())
    for _ in range(max_length):
        new_token_id = generate_single_greedy(model, token_ids)
        token_ids.append(new_token_id)
        if new_token_id == tokenizer.eos_id():
            break
    return tokenizer.decode(token_ids)


def main(checkpoint_dir, tokenizer_path):
    tokenizer = SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    llama = build_llama_model(
        checkpoint_dir=checkpoint_dir,
        vocab_size=tokenizer.vocab_size(),
    )

    system_prompt = """
    Answer in a precise and concise manner.
    """
    sentences = [
        "Who was the third president of the United States?",
        "What is the capital of France?",
        "Why is the sky blue?",
    ]
    for sentence in sentences:
        prompt = f"{system_prompt} /n {sentence}"
        print(generate_greedy(llama, tokenizer, prompt, max_length=50))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="llama-2-7b-chat")
    parser.add_argument("--tokenizer-path", type=str, default="tokenizer.model")
    args = parser.parse_args()
    main(**vars(args))
