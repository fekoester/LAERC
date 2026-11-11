# LAERC – Layered Attention-Enhanced Reservoir Language Model

LAERC is a small PyTorch language model built from **frozen random reservoirs** and a learned, **content-dependent readout**. There are no self-attention layers. Each layer instead has:
- a fixed random recurrent network (the reservoir) that carries the context
- a learned MLP and gate that decide, token by token, *what* to take from the reservoir state

That readout is the "attention": the model attends to features of a high-dimensional, fading memory of the past, not to individual past tokens.

The repo has a `laerc` package (model, training with validation and resuming, memmapped data) and scripts to:
- tokenize text with the GPT-2 tokenizer
- train
- sample with carried reservoir state
- benchmark against simple baselines

## The model

Each of the *L* blocks maps the residual stream $x_t \in \mathbb{R}^D$ to a new one:

$$
\begin{aligned}
h_t &= \tanh\!\big(U\,\mathrm{LN}(x_t) + W h_{t-1}\big) && \text{frozen reservoir, } h_t\in\mathbb{R}^{R},\ R = 4D \\
r_t &= e^{s}\,\mathrm{MLP}\big(\mathrm{LN}(h_t)\big) && \text{learned readout of the reservoir state} \\
g_t &= \sigma\big(\mathrm{MLP}(\mathrm{LN}([x_t, r_t]))\big) && \text{content-dependent gate} \\
m_t &= g_t \odot r_t + (1-g_t)\odot x_t \\
y_t &= m_t + \alpha\,\mathrm{FFN}(\mathrm{LN}(m_t)) && \alpha \text{ initialised at 0 (ReZero)}
\end{aligned}
$$

Only $U$ and $W$ are random and frozen. $W$ is scaled to **spectral radius** ρ, which sets how long the reservoir remembers: a perturbation decays like $\rho^t$, so the memory timescale is about $1/\ln(1/\rho)$ tokens. That is roughly 10 tokens at ρ = 0.9 and 100 at ρ = 0.99. Embedding and output weights are tied.

Since everything except the reservoir acts on one token at a time, the model can generate **incrementally**. `sample.py` carries each block's reservoir state forward, so every new token costs one step instead of a pass over the whole sequence (15× faster for 500 tokens on CPU). Feeding a sequence in pieces gives exactly the same logits as feeding it at once, and a test checks this.

## Results: Shakespeare

Setup:
- complete works (2.7M GPT-2 tokens), compact vocabulary of 20,204 tokens
- D = 64, L = 4, sequence length 128, batch 16
- 1,200 steps at lr 3e-3, one seed, 2 CPU cores, about 9 minutes per model
- validation loss on fixed windows of the held-out last 1%

Reproduce with `python benchmarks/run_benchmark.py --data_dir data/shakespeare_compact`.

| model | trainable non-embedding params | best val loss | val perplexity |
|---|---|---|---|
| FFN only, `--no_reservoir` (no information flows between tokens) | 184k | 5.384 | 217.9 |
| LAERC, v0.1 reservoir scaling (actual ρ ≈ 0.47) | 269k | 5.191 | 179.7 |
| **LAERC, ρ = 0.9** | 269k | **5.142** | **171.0** |
| LAERC, ρ = 0.99 | 269k | 5.145 | 171.5 |
| GPT, same width and depth (4 heads, learned positions) | 208k | 5.295 | 199.2 |

![Validation curves](results/shakespeare/val_curves.png)

- **The reservoirs carry real context.** Going from the context-free FFN to LAERC lowers perplexity from 218 to 171, a 22% reduction.
- **The spectral-radius fix helps:** perplexity drops from 180 to 171. Between ρ = 0.9 and ρ = 0.99 there is no difference at this budget.
- **On this short CPU budget, LAERC beats a small GPT of the same width and depth.** Two caveats: the GPT has about 22% fewer trainable parameters, and its curve is still falling faster at the end. So this is a statement about sample efficiency early in training, not about which architecture wins at scale.

Sample from the ρ = 0.9 model (`--prompt "ROMEO:" --top_p 0.9 --temperature 0.9`):

```text
ROMEO:
[He exits.]

Scene 4
=====

SceneUD  What was all, the people of a world
To see it well to hear thee not.

DUKE, [aside]
Nose man's body, we do in the Duke of the
somes a pertbroius, and the Lord of France.
```

## Installation

```bash
git clone https://github.com/fekoester/LAERC.git && cd LAERC
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .                      # add ".[openwebtext]" for OpenWebText, ".[dev]" for tests
```

For a CPU-only machine, run `pip install torch --index-url https://download.pytorch.org/whl/cpu` first, which is a much smaller download.

## Quick start: Shakespeare

```bash
mkdir -p data/raw
curl -L https://raw.githubusercontent.com/fekoester/Shakespeare_Res/main/data/shakespeare.txt -o data/raw/shakespeare.txt

# tokenize (GPT-2 BPE); --compact_vocab keeps only the ~20k tokens that occur -> 2.5x smaller output layer
python scripts/prepare_gpt2_tokens.py --dataset custom --raw_dir data/raw \
    --out_dir data/shakespeare_compact --val_fraction 0.01 --compact_vocab

# train (validation loss every 500 steps, best.pt kept)
python scripts/train_openwebtext.py --data_dir data/shakespeare_compact \
    --seq 128 --emb 64 --layers 4 --batch 32 --epochs 10 --steps_per_epoch 1000 --lr 3e-3

# sample (best.pt in the run directory, carried reservoir state)
python scripts/sample.py --ckpt_dir checkpoints/laerc_v20204_seq128_d64_L4_R4_RMLP1_FF4 \
    --prompt "ROMEO:" --max_tokens 200 --top_p 0.9 --temperature 0.9
```

The vocabulary size is read from the dataset's `meta.json`, so `--vocab` doesn't need to be set. The id map travels inside each checkpoint, and `sample.py` uses it to decode compact ids.

A run directory contains:

| file | content |
|---|---|
| `model_step<N>.pt` | checkpoint at the end of each epoch (model, optimizer, scheduler, RNG) |
| `best.pt` | checkpoint with the lowest validation loss |
| `loss_log.csv` | training loss every `--log_interval_steps` |
| `val_log.csv` | validation loss and perplexity every `--eval_interval_steps` |

Continue a run with `--resume path/to/model_stepN.pt`.

## OpenWebText

```bash
python scripts/prepare_gpt2_tokens.py --dataset openwebtext --out_dir data/openwebtext_tokens \
    --val_fraction 0.01 --max_docs 100000
python scripts/train_openwebtext.py --data_dir data/openwebtext_tokens --seq 128 --emb 64 --layers 8 \
    --batch 64 --epochs 5 --steps_per_epoch 2000 --amp      # --amp: bf16 autocast on CUDA
```

Tokens are streamed to disk, so 100k documents (about 113M tokens) need little RAM. Documents are separated by GPT-2's end-of-text token.

## Options

`scripts/train_openwebtext.py --help` lists everything. The main options:

| group | flags |
|---|---|
| data | `--data_dir`, `--seq`, `--vocab` (taken from `meta.json` when present) |
| model | `--emb`, `--layers`, `--resv_mult` (R = mult·D), `--res_mlp_mult`, `--ffn_mult`, `--res_radius`, `--res_scale {radius,norm}`, `--no_reservoir` |
| optimization | `--batch`, `--accum`, `--lr`, `--weight_decay`, `--grad_clip`, `--beta2`, `--epochs`, `--steps_per_epoch` |
| schedule | `--no_sched`, `--warmup_frac`, `--hold_frac`, `--min_lr_ratio` |
| evaluation | `--eval_interval_steps`, `--eval_batches`, `--resume` |
| hardware | `--amp` (bf16 on CUDA), `--compile` |

Sampling options are `--temperature`, `--top_k`, `--top_p`, `--seed` and `--num_samples`.

## Changes in 0.2

- **Fixed: the reservoir's spectral radius.** v0.1 scaled the *spectral norm* to 0.9, which for a random Gaussian matrix leaves a spectral radius of only about 0.47 (memory of about 1.3 tokens). The radius is now scaled exactly. `--res_scale norm` reproduces v0.1, and a test checks this against the original code.
- **Fixed: validation.** `val.bin` was never evaluated. There is now validation loss and perplexity during training, `best.pt` and `--resume`.
- **Fixed: `--beta2` was ignored.**
- **Fixed: checkpoints from `--compile` couldn't be loaded** because of the `_orig_mod.` prefix. Both old and new checkpoints now load.
- **Fixed: data preparation** now separates documents with end-of-text tokens and streams to disk instead of building one Python list.
- **New:**
  - incremental generation with carried reservoir state, plus top-p sampling
  - `--compact_vocab`
  - bf16 autocast on CUDA
  - `tiktoken` added to the requirements
- **New:** pyproject, a benchmark against an FFN-only model and a small GPT, tests and CI.
