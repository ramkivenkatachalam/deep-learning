# 9 — NanoGPT: build a GPT from scratch

Builds a small decoder-only Transformer (GPT) in PyTorch and trains it as a character-level language model on Tiny Shakespeare. Based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY).

## Notebook

`course_notebook_torch.ipynb` builds the model up step by step:

1. **Data + tokenization** — 1.1M characters of Shakespeare, 65-character vocabulary (downloaded to `data/input.txt` on first run)
2. **Next-token prediction** — context windows, batching
3. **Bigram baseline** — a lookup table that only sees the previous character
4. **Self-attention from scratch** — averaging with loops → matrix multiply → masked softmax → query/key/value
5. **GPT blocks** — `Head`, `MultiHeadAttention`, `FeedForward`, `Block` (pre-LayerNorm + residuals)
6. **Training + generation** — temperature and top-k sampling, attention heatmaps
7. **From nanoGPT to ChatGPT** — scale, BPE tokens, fine-tuning/RLHF; exercises

## Results

Default config: 6 layers, 6 heads, 192-dim embeddings, 128-character context, dropout 0.2, AdamW lr 1e-3, batch 64, 3000 steps.

| Model | Params | Val loss | Train time (Apple MPS) |
|---|---|---|---|
| Bigram | 4.2K | 2.50 | ~4s |
| GPT | 2.72M | **1.51** | ~4.5 min |

Sample (prompt `ROMEO:\nBut soft, `, temperature 0.5, top-k 20):

```
ROMEO:
But soft, I am to prove the good common to speak.

LADY CAPULET:
What is it is not become to me?
```

## Running

```bash
cd 9_nano_gpt
uv run jupyter notebook course_notebook_torch.ipynb
```

On CPU only, reduce `n_embd`, `n_layer` and `max_iters` (see the hyperparameter cell).
