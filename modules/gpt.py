from typing import Any

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
import torch
from torch import nn

from modules.embedding import PositionalEmbedding
from modules.transformer import DecoderBlock, RMSNorm

class GPT(LightningModule):

    def __init__(
        self,
        num_layers,
        num_heads,
        d_model,
        context_length,
        pad_token,
        vocab_size,
        norm="rms",
        activation="relu",
        proba_dropout=0.01,
        rope_embeddings=False,
        num_query_heads_per_key=None,
        intermediate_size=None,
        learning_rate=1e-3,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.d_model = d_model
        self.pad_token = pad_token
        self.context_length = context_length
        self.learning_rate = learning_rate
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        if rope_embeddings:
            # identity - embeddings are computed in the multi head attention layer
            self.pos_embedding = nn.Identity()
        else:
            self.pos_embedding = PositionalEmbedding(d_model, context_length)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    num_heads,
                    norm=norm,
                    dropout=proba_dropout,
                    activation=activation,
                    rope=rope_embeddings,
                    num_query_heads_per_key=num_query_heads_per_key,
                    intermediate_size=intermediate_size,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(p=proba_dropout)
        self.norm = {"rms": RMSNorm(d_model), "layer": nn.LayerNorm(d_model)}[norm]
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.text_embedding(x)
        x = self.pos_embedding(x)
        x = self.dropout(x)
        for layer in self.layers:
            # Use gradient checkpointing to save memory
            if self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.norm(x)
        x = self.lm_head(x)
        return x

    def _predict_probas(self, x):
        assert len(x) <= self.context_length
        x = x + [self.pad_token] * (self.context_length - len(x))
        x = torch.tensor(x).unsqueeze(0).to(self.device)
        y = self.forward(x)[0]
        return torch.softmax(y, dim=1)

    def generate(self, input_ids, tokenizer, max_new_tokens=50, temperature=0.8, top_k=None):
        """Generate text tokens given input token IDs.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with input token IDs
            tokenizer: Tokenizer to get EOS token ID
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (> 0 for sampling, 0 for greedy)
            top_k: If set, only sample from top k tokens
            
        Returns:
            Generated token IDs as tensor of shape (batch_size, original_len + new_tokens)
        """
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass - only need logits for last position
                logits = self.forward(generated)  # Shape: (batch_size, seq_len, vocab_size)
                next_logits = logits[:, -1, :]   # Shape: (batch_size, vocab_size)
                
                # Apply temperature and sample
                if temperature > 0:
                    next_logits = next_logits / temperature
                    
                    # Apply top_k filtering if specified
                    if top_k is not None:
                        # Get top k values
                        topk_logits, topk_indices = torch.topk(next_logits, k=top_k, dim=-1)
                        # Create mask for top k
                        mask = torch.full_like(next_logits, float('-inf'))
                        mask.scatter_(-1, topk_indices, topk_logits)
                        next_logits = mask
                    
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # Shape: (batch_size, 1)
                else:
                    # Greedy: take argmax
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # Shape: (batch_size, 1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for EOS token in all batch items
                if (next_token.squeeze(-1) == tokenizer.eos_token_id).all():
                    break
        
        return generated

    def compute_metrics(self, batch) -> torch.Tensor:
        inputs, target = batch
        output = self.forward(inputs).transpose(1, 2)
        loss = torch.nn.functional.cross_entropy(output, target, ignore_index=self.pad_token)
        # accuracy
        preds = torch.argmax(output, dim=1)
        correct_mask = (preds == target)[target != self.pad_token]
        correct_sum = correct_mask.sum()
        total = correct_mask.size(0)
        acc = correct_sum / total
        return loss, acc

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, acc = self.compute_metrics(batch)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.compute_metrics(batch)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return super().train_dataloader()

    def configure_optimizers(self) -> Any:
        # Use SGD with momentum instead of Adam to save memory
        # Adam requires 2x state (momentum + variance), SGD only needs 1x
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
