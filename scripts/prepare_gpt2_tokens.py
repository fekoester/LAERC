#!/usr/bin/env python3
import argparse
import os
from typing import Iterable, Optional

import numpy as np
from tqdm import tqdm
from transformers import GPT2TokenizerFast


def iter_text_from_dir(raw_dir: str) -> Iterable[str]:
    """Yield text from all .txt files under raw_dir (recursively)."""
    raw_dir = os.path.abspath(raw_dir)
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"raw_dir does not exist or is not a directory: {raw_dir}")

    for root, _, files in os.walk(raw_dir):
        for fname in files:
            if not fname.lower().endswith(".txt"):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    txt = f.read()
                if txt.strip():
                    yield txt
            except UnicodeDecodeError:
                # best-effort fallback
                with open(path, "r", encoding="latin-1") as f:
                    txt = f.read()
                if txt.strip():
                    yield txt


def iter_text_openwebtext(max_docs: Optional[int] = None) -> Iterable[str]:
    """Yield text from the HuggingFace 'openwebtext' dataset."""
    from datasets import load_dataset

    ds = load_dataset("openwebtext", split="train")
    n = len(ds)
    limit = n if max_docs is None else min(max_docs, n)

    for i in range(limit):
        txt = ds[i].get("text", "")
        if isinstance(txt, str) and txt.strip():
            yield txt


def encode_stream_to_ids(
    texts: Iterable[str],
    tokenizer: GPT2TokenizerFast,
) -> np.ndarray:
    """Encode a stream of texts into a single 1D array of token ids."""
    all_ids = []
    for txt in tqdm(texts, desc="Encoding text", dynamic_ncols=True):
        ids = tokenizer.encode(txt)
        if ids:
            all_ids.extend(ids)
    return np.array(all_ids, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(
        description="Prepare memmapped GPT-2-tokenized dataset (train.bin / val.bin)."
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="custom",
        choices=["custom", "openwebtext"],
        help="Which dataset to preprocess: 'custom' = local text files; 'openwebtext' = HF openwebtext.",
    )
    ap.add_argument(
        "--raw_dir",
        type=str,
        default="data/raw",
        help="Directory containing .txt files (used when --dataset custom).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="data/openwebtext",
        help="Output directory for train.bin / val.bin.",
    )
    ap.add_argument(
        "--val_fraction",
        type=float,
        default=0.01,
        help="Fraction of tokens to reserve for validation.",
    )
    ap.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Optional: maximum number of documents to use (for large datasets like openwebtext).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # GPT-2 tokenizer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token  # just in case

    # Choose text source
    if args.dataset == "openwebtext":
        print("Preparing dataset: openwebtext (HuggingFace)")
        texts = iter_text_openwebtext(max_docs=args.max_docs)
    else:
        print(f"Preparing dataset from local text files under: {args.raw_dir}")
        texts = iter_text_from_dir(args.raw_dir)

    # Encode to ids
    all_ids = encode_stream_to_ids(texts, tok)
    n = len(all_ids)
    print(f"Collected {n} tokens in total.")

    if n < 1000:
        raise ValueError(
            f"Too few tokens ({n}) collected. Add more text or adjust dataset options."
        )

    # Train / val split
    n_val = max(1, int(n * args.val_fraction))
    n_val = min(n_val, n - 1)  # ensure at least one train token
    n_train = n - n_val

    train_ids = all_ids[:n_train].astype(np.uint16)
    val_ids = all_ids[n_train:].astype(np.uint16)

    # Write memmaps
    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")

    train_mm = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=train_ids.shape)
    train_mm[:] = train_ids[:]
    del train_mm

    val_mm = np.memmap(val_path, dtype=np.uint16, mode="w+", shape=val_ids.shape)
    val_mm[:] = val_ids[:]
    del val_mm

    print(f"Wrote {n_train} train tokens to {train_path}")
    print(f"Wrote {n_val}  val tokens to {val_path}")


if __name__ == "__main__":
    main()

