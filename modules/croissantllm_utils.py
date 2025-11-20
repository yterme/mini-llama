#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM, AutoConfig

def load_croissantllm_weights(model, model_name="croissantllm/CroissantLLMBase"):
    """
    Load CroissantLLM weights into our GPT2 model.
    
    CroissantLLM architecture:
    - vocab_size: 32000
    - hidden_size: 2048  
    - num_layers: 24
    - num_heads: 16
    - num_key_value_heads: 16 (no grouped query attention)
    - intermediate_size: 5504
    """
    print(f"Loading CroissantLLM weights from {model_name}...")
    
    # Load the original model
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"  # Load on CPU first to avoid memory issues
    )
    
    # Get config
    config = AutoConfig.from_pretrained(model_name)
    print(f"CroissantLLM config:")
    print(f"  - vocab_size: {config.vocab_size}")
    print(f"  - hidden_size: {config.hidden_size}")
    print(f"  - num_hidden_layers: {config.num_hidden_layers}")
    print(f"  - num_attention_heads: {config.num_attention_heads}")
    print(f"  - num_key_value_heads: {config.num_key_value_heads}")
    print(f"  - intermediate_size: {config.intermediate_size}")
    
    print(f"Our model has {len(model.layers)} layers, CroissantLLM has {config.num_hidden_layers} layers")
    
    # Create weight mapping
    weight_mappings = create_croissantllm_weight_mapping(len(model.layers))
    
    print(f"\nAttempting to load {len(weight_mappings)} weight mappings...")
    
    # Load weights
    successful_loads = 0
    failed_loads = 0
    failed_weights = []
    
    with torch.no_grad():
        for original_name, our_name in weight_mappings.items():
            try:
                # Get original weight
                original_weight = original_model.state_dict()[original_name]
                
                # Get our model's parameter
                our_param_parts = our_name.split('.')
                our_param = model
                for part in our_param_parts:
                    our_param = getattr(our_param, part)
                
                # Check shape compatibility
                if original_weight.shape == our_param.shape:
                    our_param.copy_(original_weight)
                    print(f"✓ Loaded {original_name} -> {our_name} {original_weight.shape}")
                    successful_loads += 1
                else:
                    print(f"✗ Shape mismatch {original_name} -> {our_name}: {original_weight.shape} vs {our_param.shape}")
                    failed_loads += 1
                    failed_weights.append((original_name, "shape mismatch"))
                    
            except KeyError:
                print(f"✗ Key not found: {original_name}")
                failed_loads += 1
                failed_weights.append((original_name, "key not found"))
            except Exception as e:
                print(f"✗ Error loading {original_name}: {e}")
                failed_loads += 1
                failed_weights.append((original_name, str(e)))
    
    print(f"\nCroissantLLM weights loaded successfully! ({successful_loads}/{successful_loads + failed_loads} weights loaded)")
    
    if failed_loads > 0:
        print(f"\n❌ Failed to load {failed_loads} weights:")
        for weight_name, error in failed_weights:
            print(f"  - {weight_name}: {error}")
    
    # Clean up memory by deleting the original model
    print("🧹 Cleaning up memory: deleting original HuggingFace model...")
    del original_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✅ Memory cleanup completed")
    
    return successful_loads == len(weight_mappings)


def create_croissantllm_weight_mapping(num_layers):
    """Create mapping from CroissantLLM parameter names to our model parameter names."""
    
    mappings = {}
    
    # Embedding layers
    mappings["model.embed_tokens.weight"] = "text_embedding.weight"
    mappings["lm_head.weight"] = "lm_head.weight"
    
    # Final norm
    mappings["model.norm.weight"] = "norm.scale"
    
    # Transformer layers
    for i in range(num_layers):
        # Attention layers
        mappings[f"model.layers.{i}.self_attn.q_proj.weight"] = f"layers.{i}.attention.linear_q.weight"
        mappings[f"model.layers.{i}.self_attn.k_proj.weight"] = f"layers.{i}.attention.linear_k.weight"
        mappings[f"model.layers.{i}.self_attn.v_proj.weight"] = f"layers.{i}.attention.linear_v.weight"
        mappings[f"model.layers.{i}.self_attn.o_proj.weight"] = f"layers.{i}.attention.linear_proj.weight"
        
        # Layer norms
        mappings[f"model.layers.{i}.input_layernorm.weight"] = f"layers.{i}.norm1.scale"
        mappings[f"model.layers.{i}.post_attention_layernorm.weight"] = f"layers.{i}.norm2.scale"
        
        # MLP layers (SwiGLU)
        mappings[f"model.layers.{i}.mlp.gate_proj.weight"] = f"layers.{i}.activation_unit.linear1.weight"
        mappings[f"model.layers.{i}.mlp.up_proj.weight"] = f"layers.{i}.activation_unit.linear2.weight"
        mappings[f"model.layers.{i}.mlp.down_proj.weight"] = f"layers.{i}.proj.weight"
    
    return mappings