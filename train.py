import argparse
from datetime import datetime
import os

import yaml
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import load_dataset
from transformers import AutoTokenizer
from dotenv import load_dotenv

# Set CUDA memory optimizations and Tensor Core precision
if torch.cuda.is_available():
    # Use medium precision for Tensor Cores (better performance, slightly less precision)
    torch.set_float32_matmul_precision('medium')
    torch.cuda.set_per_process_memory_fraction(0.9)
    print(f"🔧 CUDA optimizations enabled, Tensor Core precision set to 'medium'")

from modules.data import ChatDataset, TokenizedDataset, pad_collate
from modules.gpt import GPT
from modules.tinyllama_utils import load_tinyllama_weights
from modules.croissantllm_utils import load_croissantllm_weights
from modules.training_callbacks import TextGenerationCallback


# Load environment variables
load_dotenv()


def main(
    dataset: str,
    tokenizer_name: str,
    epochs: int,
    check_val_every_n_epoch: int,
    load_ckpt: str = None,
    load_tinyllama: str = None,
    load_croissantllm: str = None,
    config_file: str = None,
    learning_rate: float = 1e-5,
):
    # load yaml config - use appropriate config based on model type
    if config_file is None:
        if load_tinyllama:
            config_file = "tinyllama_config.yaml"
        elif load_croissantllm:
            config_file = "croissantllm_config.yaml"
        else:
            config_file = "model_config.yaml"
    
    print(f"Using config file: {config_file}")
    model_config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    
    batch_size = 1
    context_length = 1024 
    num_workers = 2
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    
    # Ensure pad token is set (some tokenizers don't have one by default)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"⚠️  Setting pad_token_id to eos_token_id ({tokenizer.eos_token_id})")
        else:
            tokenizer.pad_token_id = 0
            print(f"⚠️  Setting pad_token_id to 0")
    
    # Use proper pad token ID, not vocab_size
    if tokenizer.pad_token_id is not None:
        pad_token = tokenizer.pad_token_id
    else:
        pad_token = tokenizer.eos_token_id
    
    # Use appropriate vocab size based on which pre-trained model we're loading
    if load_tinyllama:
        from transformers import AutoConfig
        tinyllama_config = AutoConfig.from_pretrained(load_tinyllama)
        vocab_size = tinyllama_config.vocab_size
        print(f"Using TinyLlama vocab_size: {vocab_size}")
        
        # Ensure tokenizer vocab size matches or is smaller
        if tokenizer.vocab_size > vocab_size:
            print(f"Warning: Tokenizer vocab_size ({tokenizer.vocab_size}) > model vocab_size ({vocab_size})")
            print("This may cause indexing errors. Consider using a compatible tokenizer.")
    elif load_croissantllm:
        from transformers import AutoConfig
        croissantllm_config = AutoConfig.from_pretrained(load_croissantllm)
        vocab_size = croissantllm_config.vocab_size
        print(f"Using CroissantLLM vocab_size: {vocab_size}")
        
        # Ensure tokenizer vocab size matches or is smaller
        if tokenizer.vocab_size > vocab_size:
            print(f"Warning: Tokenizer vocab_size ({tokenizer.vocab_size}) > model vocab_size ({vocab_size})")
            print("This may cause indexing errors. Consider using a compatible tokenizer.")
    else:
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
            learning_rate=learning_rate,
            **model_config,
        )
    # Load pre-trained weights if specified
    if load_tinyllama is not None:
        gpt_model = load_tinyllama_weights(gpt_model, load_tinyllama)
    elif load_croissantllm is not None:
        success = load_croissantllm_weights(gpt_model, load_croissantllm)
        if success:
            print(f"✅ Successfully loaded CroissantLLM weights from {load_croissantllm}")
        else:
            print(f"⚠️ Some CroissantLLM weights failed to load from {load_croissantllm}")
    
    # Clear any cached memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 Cleared CUDA cache before training")
        # Enable memory efficient attention if available
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("📌 Enabled TF32 for memory efficiency")
        except:
            pass
        # Fix memory fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("📌 Enabled expandable memory segments to reduce fragmentation")
    # pytorch lightning model checkpoint and text generation callbacks
    callbacks = [
        ModelCheckpoint(monitor="val_acc", save_top_k=1, mode="max"),
        ModelCheckpoint(every_n_train_steps=1000),
    ]
    
    if dataset == "chat":
        callbacks.append(TextGenerationCallback(
            tokenizer, 
            dataset_path="data/_chat_cleaned_train.txt",
            generation_frequency="step", 
            interval=2000,
            dataset_seq_len=context_length,  # Use same context length as training
            num_samples=5,
            random_samples=True,
            sample_seed=42,
        ))
    trainer_kwargs = {
        "accelerator": "gpu",
        "check_val_every_n_epoch": check_val_every_n_epoch,
        "callbacks": callbacks,
        "max_epochs": epochs,
        "gradient_clip_val": 1.0,  
        "log_every_n_steps": 10,
        "enable_progress_bar": True,
    }
    
    if load_croissantllm:
        trainer_kwargs.update({
            "accumulate_grad_batches": 8,  
            "precision": "16-mixed", 
        })
    
    trainer = Trainer(**trainer_kwargs)

    # huggingface tinystories dataset
    if dataset == "chat":
        train_dataset = ChatDataset(
            "data/_chat_cleaned_train.txt", tokenizer=tokenizer, sequence_length=context_length
        )
        val_dataset = ChatDataset(
            "data/_chat_cleaned_val.txt", tokenizer=tokenizer, sequence_length=context_length
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
    model_name = f"gpt_model_{dataset}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.pth"
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
    parser.add_argument("--load-tinyllama", type=str, default=None, help="Load TinyLlama weights (e.g., 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')")
    parser.add_argument("--load-croissantllm", type=str, default=None, help="Load CroissantLLM weights (e.g., 'croissantllm/CroissantLLMBase')")
    parser.add_argument("--config-file", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate for fine-tuning (default: 1e-5)")
    args = parser.parse_args()
    main(**vars(args))
