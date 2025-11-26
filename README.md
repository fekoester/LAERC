# LAERC-LM: Layered Attention-Enhanced Reservoir Language Model

This repo contains a modular implementation of a **layered reservoir + FFN language model (LAERC)** in PyTorch.

- `laerc/`: core library (config, model, data, training utilities)
- `scripts/train_openwebtext.py`: train LAERC on a memmapped token dataset
- `scripts/prepare_openwebtext.py`: turn raw `.txt` into `train.bin` / `val.bin` using GPT-2 BPE (`tiktoken`)
- `scripts/sample.py`: generate text from a trained checkpoint

## Quickstart

```bash
pip install -r requirements.txt

# prepare data (put .txt in data/raw/)
python scripts/prepare_openwebtext.py

# train
python scripts/train_openwebtext.py --compile

# sample
python scripts/sample.py --ckpt_dir checkpoints/<auto-tag> --prompt "Once upon a time"
```
