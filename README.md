# LAERC – Layered Attention-Enhanced Reservoir Language Model

LAERC is a small, modular PyTorch implementation of a **layered reservoir + feed-forward** language model, trained on GPT-2 tokenized text.

The code is structured as a minimal research playground:

- A `laerc` Python package with:
  - **Config** (`TrainConfig`)
  - **Reservoir + FFN model** (`ReservoirFFNLanguageModel`)
  - **Training loop** with cosine schedule
  - **Data helpers** for memmapped token datasets
- Simple **scripts** to:
  - Prepare datasets with **GPT-2 tokenizer** (`scripts/prepare_gpt2_tokens.py`)
  - Train LAERC on any text dataset (`scripts/train_openwebtext.py`)
  - Sample text from trained checkpoints (`scripts/sample.py`)

You can start with a tiny **Shakespeare** dataset and then move to an **OpenWebText** subset for a more realistic experiment.

---

## 1. Repository structure

```text
LAERC/
  laerc/
    __init__.py
    config.py
    data.py
    model.py
    train.py
    utils.py

  scripts/
    prepare_gpt2_tokens.py   # generic GPT-2 tokenizer data prep
    train_openwebtext.py     # generic LM training script
    sample.py                # text generation from checkpoints

  requirements.txt
  .gitignore
  README.md

  data/          # created locally, ignored by git (raw + tokenized data)
  checkpoints/   # created locally, ignored by git (model checkpoints)
  .venv/         # local virtual environment (ignored)
```

## 2. Installation
## 2.1. Clone and create a virtual environment

```text
git clone git@github.com:fekoester/LAERC.git
cd LAERC

# create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# upgrade pip
pip install --upgrade pip

2.2. Install dependencies
pip install -r requirements.txt
```

This installs:

torch

transformers (for GPT-2 tokenizer)

datasets (for OpenWebText)

numpy, tqdm, etc.

Make sure PyTorch is installed with CUDA support if you want GPU training (see PyTorch website for the correct pip index URL for your system).

3. Data preparation with GPT-2 tokenizer

All datasets are converted to a simple memmapped format:

train.bin – uint16 GPT-2 token ids for training

val.bin – uint16 GPT-2 token ids for validation

using the GPT-2 tokenizer (GPT2TokenizerFast from transformers).

The main script is:

python -m scripts.prepare_gpt2_tokens ...


It supports two modes:

--dataset custom – use local .txt files.

--dataset openwebtext – download and tokenize HuggingFace OpenWebText.

4. Quick start: Shakespeare mini-example

This is the recommended minimal test. It’s small, quick, and shows everything works.

4.1. Download the Shakespeare text
mkdir -p data/raw

curl -L https://raw.githubusercontent.com/fekoester/Shakespeare_Res/main/data/shakespeare.txt \
  -o data/raw/shakespeare.txt

4.2. Tokenize with GPT-2
python -m scripts.prepare_gpt2_tokens \
  --dataset custom \
  --raw_dir data/raw \
  --out_dir data/shakespeare_tokens \
  --val_fraction 0.01


This will:

Read all .txt files under data/raw/

Tokenize them using GPT-2 tokenizer

Write:

data/shakespeare_tokens/train.bin

data/shakespeare_tokens/val.bin

Typical output:

Preparing dataset from local text files under: data/raw
Encoding text: ...
Collected XXXXX tokens in total.
Wrote NNNNN train tokens to data/shakespeare_tokens/train.bin
Wrote MMMMM val tokens to data/shakespeare_tokens/val.bin

4.3. Train a small LAERC model on Shakespeare

Run a small model for a modest number of epochs:

python -m scripts.train_openwebtext \
  --data_dir data/shakespeare_tokens \
  --seq 128 \
  --emb 64 \
  --layers 4 \
  --batch 32 \
  --accum 1 \
  --epochs 50 \
  --steps_per_epoch 1000 \
  --log_interval_steps 50 \
  --lr 3e-3


Key arguments:

--data_dir – points to the tokenized dataset with train.bin / val.bin

--seq – sequence length (context)

--emb – embedding dimension

--layers – number of LAERC blocks

--batch – batch size (sequences per step)

--accum – gradient accumulation steps (1 = no accumulation)

--epochs, --steps_per_epoch – training duration

--lr – learning rate

Typical console output:

{
  "data_dir": "data/shakespeare_tokens",
  "vocab": 50257,
  "seq": 128,
  "emb": 64,
  "layers": 4,
  ...
}
Model parameters: total=3,814,856 | trainable=3,485,128
Epoch 1: 100%|...| loss=6.18, lr=3.00e-03
Epoch 2: 100%|...| loss=4.22, lr=3.00e-03
...


Checkpoints are saved to:

checkpoints/laerc_v50257_seq128_d64_L4_R4_RMLP1_FF4/model_stepXXXX.pt


and a loss_log.csv with step-wise loss is written inside the checkpoint directory.

4.4. Generate Shakespeare-like text

After training, sample from the latest checkpoint:

python -m scripts.sample \
  --ckpt checkpoints/laerc_v50257_seq128_d64_L4_R4_RMLP1_FF4/model_step12000.pt \
  --prompt "ROMEO: " \
  --max_tokens 200


Example output (abbreviated):

ROMEO:  And yet thy worth must
be done to 't, and I must be a very fantastical.
Here in the platform and Soldiers.]

...


The exact text will vary, but you should see recognizable Shakespeare-like structure (speaker names, line breaks, archaic phrasing).

5. OpenWebText example

For a more realistic experiment, you can use a subset of OpenWebText

via HuggingFace datasets.

5.1. Download + tokenize OpenWebText subset

This command uses the openwebtext dataset, tokenizes it with GPT-2, and writes a large memmapped dataset:

python -m scripts.prepare_gpt2_tokens \
  --dataset openwebtext \
  --out_dir data/openwebtext_tokens \
  --val_fraction 0.01 \
  --max_docs 100000


Notes:

--max_docs limits how many documents to use (here: first 100,000).
Remove it to use the full dataset (this may be large).

Tokens are stored as uint16 IDs consistent with GPT-2’s tokenizer.

Example output:

Preparing dataset: openwebtext (HuggingFace)
Encoding text: 100000it [...]
Collected 112863538 tokens in total.
Wrote 111734903 train tokens to data/openwebtext_tokens/train.bin
Wrote 1128635  val tokens to data/openwebtext_tokens/val.bin

5.2. Train LAERC on OpenWebText

Now train a slightly bigger model:

python -m scripts.train_openwebtext \
  --data_dir data/openwebtext_tokens \
  --seq 128 \
  --emb 64 \
  --layers 8 \
  --batch 64 \
  --accum 1 \
  --epochs 5 \
  --steps_per_epoch 2000 \
  --log_interval_steps 100


Example console output:

Model parameters: total=4,413,136 | trainable=3,753,680
Epoch 1: 100%|...| loss=7.47, lr=3.91e-04
Epoch 2: 100%|...| loss=6.18, lr=3.14e-04
Epoch 3: 100%|...| loss=5.96, lr=1.91e-04
Epoch 4: 100%|...| loss=5.84, lr=8.30e-05
Epoch 5: 100%|...| loss=5.80, lr=4.00e-05


Loss starts around ~10–11 for random initialization on GPT-2 vocabulary and then decreases as training proceeds (the example above uses accum=1, so logged loss is the true CE).

Checkpoints are stored under:

checkpoints/laerc_v50257_seq128_d64_L8_R4_RMLP1_FF4/model_stepXXXX.pt

5.3. Generate OpenWebText-style text

Pick the latest checkpoint:

python -m scripts.sample \
  --ckpt checkpoints/laerc_v50257_seq128_d64_L8_R4_RMLP1_FF4/model_step10000.pt \
  --prompt "In a recent study, researchers discovered that" \
  --max_tokens 200


You can experiment with different prompts and model sizes.

6. Script overview and options
6.1. scripts/prepare_gpt2_tokens.py

Core arguments:

--dataset {custom, openwebtext}

--raw_dir data/raw (for custom)

--out_dir data/my_tokens

--val_fraction 0.01

--max_docs N (optional; for openwebtext)

6.2. scripts/train_openwebtext.py

Core arguments:

Data / model

--data_dir (path containing train.bin / val.bin)

--vocab (default: 50257; GPT-2 vocab size)

--seq (context length, e.g. 128 / 512)

--emb (embedding size)

--layers (number of LAERC blocks)

--resv_mult, --res_mlp_mult, --ffn_mult, --res_radius

--no_reservoir (disable reservoir path → pure FFN baseline)

Optimization

--batch

--accum (gradient accumulation factor)

--lr

--epochs

--steps_per_epoch (<=0 → auto-compute from data size)

--weight_decay

--grad_clip

--beta2

Scheduler

--no_sched (disable scheduler)

--warmup_frac, --hold_frac, --min_lr_ratio

Logging / checkpoint

--tag (run name; otherwise auto-generated from hyperparams)

--ckpt_dir (override default checkpoint directory)

--log_interval_steps

--flush_every

--seed

Call python -m scripts.train_openwebtext --help for full details.

6.3. scripts/sample.py

Core arguments:

--ckpt – path to a specific checkpoint .pt file
or

--ckpt_dir – directory with model_step*.pt files (latest step is auto-selected)

--prompt – initial text prompt

--max_tokens – number of tokens to generate

You can extend this script to add --temperature, --top_k, --top_p for more interesting sampling strategies.

7. Notes

All training currently runs in float32 (no AMP) for maximum stability, especially with cuDNN RNNs.

The reservoir part uses a frozen RNN with controlled spectral radius, followed by a small MLP and gating mechanism with the direct path.

The implementation is intentionally compact and readable to facilitate research and experimentation with layered reservoir architectures.
