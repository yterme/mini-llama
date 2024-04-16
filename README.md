LLama implementation:
- SwiGLU activation
- ROPE embeddings
- Grouped Query Attention
- RMSNorm

## Train the model
```
python train.py --dataset tinystories --tokenizer-name gpt2 --epochs 10
```

## TODO:
- Key-value caching
