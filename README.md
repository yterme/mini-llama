Mini LLama implementation:
- SwiGLU activation
- ROPE embeddings
- Grouped Query Attention
- RMSNorm

## Train the model
```
python train.py --dataset tinystories --tokenizer-name gpt2 --epochs 10
```

## Load LLama weights
- Request access to LLama weights https://llama.meta.com/llama-downloads/
- Run the download file `download.sh` found at https://github.com/meta-llama/llama/blob/main/download.sh.
```
chmod +x download.sh
./download.sh
```
- Load the model with the constructor `LLaMA.build`. For instance:
```
python llama_inference.py --checkpoints-dir llama-2-7b --tokenizer-path tokenizer.model
```

## TODO:
- Key-value caching
