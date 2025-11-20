#!/usr/bin/env python3

import torch
from pytorch_lightning.callbacks import Callback
import random
from typing import Optional
from modules.data import ChatDataset


class TextGenerationCallback(Callback):
    """Callback to generate text samples during training to monitor progress."""
    
    def __init__(
        self,
        tokenizer,
        dataset_path: Optional[str] = None,
        max_length: int = 256,
        generation_frequency: str = "epoch",
        interval: int = 500,
        use_chatdataset: bool = True,
        dataset_seq_len: int = 128,
        impose_start_line_with_username: bool = True,
        num_samples: int = 5,
        random_samples: bool = True,
        sample_seed: int = 42,
    ):
        """
        Args:
            tokenizer: The tokenizer to use for encoding/decoding
            dataset_path: Path to dataset file to sample from. If None, will use default prompts
            max_length: Maximum length of generated text
            generation_frequency: When to generate ("epoch" or "step")
            interval: Steps interval for generation (when frequency="step")
            use_chatdataset: If True, load via ChatDataset; else load from raw file
            dataset_seq_len: Sequence length for ChatDataset
            impose_start_line_with_username: Whether to merge lines by username pattern (ChatDataset)
            num_samples: Number of samples to draw (default 5)
            random_samples: If True, sample randomly; else take first N samples
            sample_seed: Seed for reproducible random sampling
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.generation_frequency = generation_frequency
        self.interval = interval
        self.dataset_path = dataset_path
        self.fixed_samples = []
        self.samples_loaded = False

        # Sample configuration
        self.use_chatdataset = use_chatdataset
        self.dataset_seq_len = dataset_seq_len
        self.impose_start_line_with_username = impose_start_line_with_username
        self.num_samples = num_samples
        self.random_samples = random_samples
        self.sample_seed = sample_seed
        
        # Store dataset reference and sampled indices
        self.dataset = None
        self.sample_indices = []

        # Load samples from dataset
        self._load_fixed_samples()
    
    def _load_fixed_samples(self):
        """Load full dataset and sample random token positions."""
        if self.dataset_path and not self.samples_loaded:
            try:
                # Load FULL dataset to get all tokens
                # We'll sample random positions from it
                self.dataset = ChatDataset(
                    self.dataset_path,
                    self.tokenizer,
                    sequence_length=self.dataset_seq_len,
                    impose_start_line_with_username=self.impose_start_line_with_username,
                    sample_indices=None,  # Load full dataset
                )
                
                self.full_tokens = self.dataset.full_text_tokens
                num_possible_windows = max(1, len(self.full_tokens) - self.dataset_seq_len)
                
                # Sample random window indices
                if self.random_samples:
                    random.seed(self.sample_seed)
                    n_to_take = min(self.num_samples, num_possible_windows)
                    self.sample_indices = sorted(random.sample(range(num_possible_windows), n_to_take))
                    sampling_mode = f"random windows (seed={self.sample_seed})"
                else:
                    # Take evenly spaced windows
                    n_to_take = min(self.num_samples, num_possible_windows)
                    step = max(1, num_possible_windows // n_to_take)
                    self.sample_indices = list(range(0, num_possible_windows, step))[:n_to_take]
                    sampling_mode = f"evenly-spaced windows (step={step})"

                self.samples_loaded = True
                print(f"✅ Sampled {len(self.sample_indices)} window indices from dataset (mode={sampling_mode})")
                print(f"✅ Total tokens available: {len(self.full_tokens)}, total windows: {num_possible_windows}")
                
            except Exception as e:
                error_msg = f"❌ Failed to load samples from {self.dataset_path}: {e}"
                print(error_msg)
                raise RuntimeError(error_msg)
        elif not self.dataset_path:
            raise ValueError("dataset_path must be provided to TextGenerationCallback - no default prompts available")
            
    def generate_text(self, model, prompt, max_new_tokens=None, temperature=0.8):
        """Generate text using the GPT model.
        
        Args:
            model: GPT model to use for generation
            prompt: Input prompt text
            max_new_tokens: Maximum NEW tokens to generate (if None, use self.max_length)
            temperature: Sampling temperature
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_length
            
        device = next(model.parameters()).device
        
        # Tokenize input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        
        # Use the model's generate method
        generated_ids = model.generate(
            input_ids,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch."""
        if self.generation_frequency == "epoch":
            self._generate_and_log(trainer, pl_module)
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called at the end of each training step."""
        if self.generation_frequency == "step" and batch_idx % self.interval == 0:  # Every interval steps
            self._generate_and_log(trainer, pl_module)
    
    def _generate_and_log(self, trainer, pl_module):
        """Generate text and log it using token windows from randomly sampled positions."""
        # ANSI color codes
        BOLD = "\033[1m"
        CYAN = "\033[96m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        
        step_info = f"Step {trainer.global_step}" if self.generation_frequency == "step" else ""
        print(f"\n🎭 === TEXT GENERATION EVALUATION (Epoch {trainer.current_epoch + 1} {step_info}) ===")
        
        # Check we have tokens and sample indices
        if not hasattr(self, 'full_tokens') or self.full_tokens is None:
            print("⚠️ Dataset tokens not loaded, skipping text generation")
            return
        
        if not hasattr(self, 'sample_indices') or not self.sample_indices:
            print("⚠️ Sample indices not loaded, skipping text generation")
            return
        
        # Generate from each sampled window position
        for sample_idx, window_idx in enumerate(self.sample_indices):
            try:
                # Extract token window: [window_idx, window_idx + context_length]
                start_token = window_idx
                end_token = min(start_token + self.dataset_seq_len, len(self.full_tokens))
                token_window = self.full_tokens[start_token:end_token]
                
                # Split window: 70% as prompt input, 30% as target output
                split_point = int(len(token_window) * 0.7)
                prompt_tokens_list = token_window[:split_point]
                target_tokens_list = token_window[split_point:split_point + int(len(token_window) * 0.3)]
                
                # Decode prompt and target (ground truth)
                prompt = self.tokenizer.decode(prompt_tokens_list, skip_special_tokens=True)
                right_completion = self.tokenizer.decode(target_tokens_list, skip_special_tokens=True)
                
                # Generate continuation: allow up to 30% of max_length as new tokens
                max_new_tokens = int(self.max_length * 0.3)
                generated = self.generate_text(pl_module, prompt, max_new_tokens=max_new_tokens)
                
                # Extract only the generated part (without the prompt prefix)
                if generated.startswith(prompt):
                    generated_only = generated[len(prompt):].strip()
                else:
                    generated_only = generated.strip()
                
                # Limit generated completion to same length as target for fair comparison
                generated_only_tokens = self.tokenizer.encode(generated_only)[:len(target_tokens_list)]
                generated_only = self.tokenizer.decode(generated_only_tokens, skip_special_tokens=True)
                
                print(f"\n💬 Sample {sample_idx + 1}/{len(self.sample_indices)}:")
                print(f"{BOLD}{CYAN}Input:{RESET} {prompt}")
                print(f"{BOLD}{GREEN}Predicted Completion:{RESET} {generated_only}")
                print(f"{BOLD}{YELLOW}GT Completion:{RESET} {right_completion}")
                print("-" * 70)
                
            except Exception as e:
                print(f"❌ Error generating text for sample {sample_idx + 1}: {e}")
        
        print("🎭 === END TEXT GENERATION EVALUATION ===\n")
        
        # Return model to training mode
        pl_module.train()

