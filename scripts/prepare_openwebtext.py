#!/usr/bin/env python3
"""
Prepare OpenWebText-style data into train.bin / val.bin using GPT-2 BPE (tiktoken).
You provide a directory with .txt files; we concatenate them and split into train/val.
"""

import argparse
import os
from glob import glob

import numpy as np
import tiktoken


def build_argparser():
    ap = argparse.ArgumentParser(description="Prepare memmapped train.bin / val.bin from raw text files.")
    ap.add_argument("--raw_dir", type=str, default="data/raw", help="Directory with .txt files.")
    ap.add_argument("--out_dir", type=str, default="data/openwebtext", help="Output directory.")
    ap.add_argument("--val_fraction", type=float, default=0.01, help="Fraction of tokens for validation.")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")

    txt_files = sorted(glob(os.path.join(args.raw_dir, "**", "*.txt"), recursive=True))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found under {args.raw_dir}")

    all_ids = []
    for fp in txt_files:
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()
        ids = enc.encode(text)
        all_ids.extend(ids)

    all_ids = np.array(all_ids, dtype=np.uint32)
    n = len(all_ids)
    n_val = int(n * args.val_fraction)
    n_train = n - n_val

    train_ids = all_ids[:n_train].astype(np.uint16)
    val_ids = all_ids[n_train:].astype(np.uint16)

    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    print(f"Wrote {n_train} train tokens to {train_path}")
    print(f"Wrote {n_val} val tokens to {val_path}")


if __name__ == "__main__":
    main()

