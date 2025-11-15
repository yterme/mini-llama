"""
TinyLlama model weight loading utilities
"""
from typing import Optional

def load_tinyllama_weights(gpt_model, tinyllama_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Load TinyLlama weights into our GPT model"""
    from transformers import AutoModelForCausalLM, AutoConfig
    
    print(f"Loading TinyLlama weights from {tinyllama_model_name}...")
    
    # Get TinyLlama config first to verify compatibility
    tinyllama_config = AutoConfig.from_pretrained(tinyllama_model_name)
    print(f"TinyLlama config:")
    print(f"  - vocab_size: {tinyllama_config.vocab_size}")
    print(f"  - hidden_size: {tinyllama_config.hidden_size}")
    print(f"  - num_hidden_layers: {tinyllama_config.num_hidden_layers}")
    print(f"  - num_attention_heads: {tinyllama_config.num_attention_heads}")
    print(f"  - num_key_value_heads: {tinyllama_config.num_key_value_heads}")
    print(f"  - intermediate_size: {tinyllama_config.intermediate_size}")
    
    tinyllama = AutoModelForCausalLM.from_pretrained(tinyllama_model_name)
    tinyllama_state_dict = tinyllama.state_dict()
    
    # Create mapping from TinyLlama to our model
    weight_mapping = {}
    
    # Embedding layer
    weight_mapping["model.embed_tokens.weight"] = "text_embedding.weight"
    
    # Output layer  
    weight_mapping["lm_head.weight"] = "lm_head.weight"
    
    # Final norm
    weight_mapping["model.norm.weight"] = "norm.scale"
    
    # Transformer layers
    num_layers = len(gpt_model.layers)
    print(f"Our model has {num_layers} layers, TinyLlama has {tinyllama_config.num_hidden_layers} layers")
    
    for i in range(min(num_layers, tinyllama_config.num_hidden_layers)):
        # Attention weights - need to handle the naming difference
        weight_mapping[f"model.layers.{i}.self_attn.q_proj.weight"] = f"layers.{i}.attention.linear_q.weight"
        weight_mapping[f"model.layers.{i}.self_attn.k_proj.weight"] = f"layers.{i}.attention.linear_k.weight"  
        weight_mapping[f"model.layers.{i}.self_attn.v_proj.weight"] = f"layers.{i}.attention.linear_v.weight"
        weight_mapping[f"model.layers.{i}.self_attn.o_proj.weight"] = f"layers.{i}.attention.linear_proj.weight"
        
        # Layer norms
        weight_mapping[f"model.layers.{i}.input_layernorm.weight"] = f"layers.{i}.norm1.scale"
        weight_mapping[f"model.layers.{i}.post_attention_layernorm.weight"] = f"layers.{i}.norm2.scale"
        
        # MLP weights (SwiGLU)
        weight_mapping[f"model.layers.{i}.mlp.gate_proj.weight"] = f"layers.{i}.activation_unit.linear1.weight"
        weight_mapping[f"model.layers.{i}.mlp.up_proj.weight"] = f"layers.{i}.activation_unit.linear2.weight"
        weight_mapping[f"model.layers.{i}.mlp.down_proj.weight"] = f"layers.{i}.proj.weight"
    
    # Load the weights
    our_state_dict = gpt_model.state_dict()
    loaded_count = 0
    total_count = len(weight_mapping)
    
    print(f"\nAttempting to load {total_count} weight mappings...")
    
    for tinyllama_key, our_key in weight_mapping.items():
        if tinyllama_key in tinyllama_state_dict and our_key in our_state_dict:
            tinyllama_weight = tinyllama_state_dict[tinyllama_key]
            our_weight = our_state_dict[our_key]
            
            # Check if shapes match
            if tinyllama_weight.shape == our_weight.shape:
                our_state_dict[our_key] = tinyllama_weight.clone()
                print(f"✓ Loaded {tinyllama_key} -> {our_key} {tinyllama_weight.shape}")
                loaded_count += 1
            else:
                print(f"✗ Shape mismatch for {tinyllama_key} -> {our_key}: {tinyllama_weight.shape} vs {our_weight.shape}")
        else:
            if tinyllama_key not in tinyllama_state_dict:
                print(f"✗ Missing in TinyLlama: {tinyllama_key}")
            if our_key not in our_state_dict:
                print(f"✗ Missing in our model: {our_key}")
    
    # Load the updated state dict
    gpt_model.load_state_dict(our_state_dict, strict=False)
    print(f"\nTinyLlama weights loaded successfully! ({loaded_count}/{total_count} weights loaded)")
    
    return gpt_model


def get_tinyllama_config(model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0") -> dict:
    """Get TinyLlama configuration for creating compatible models"""
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained(model_name)
    
    return {
        "vocab_size": config.vocab_size,
        "d_model": config.hidden_size,
        "num_layers": config.num_hidden_layers,
        "num_heads": config.num_attention_heads,
        "num_query_heads_per_key": config.num_attention_heads // config.num_key_value_heads,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
        "activation": "swiglu",
        "norm": "rms",
        "rope_embeddings": True,
    }