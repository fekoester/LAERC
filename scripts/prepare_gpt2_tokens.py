#!/usr/bin/env python3
"""
Prepare a memmapped GPT-2-tokenized dataset: train.bin / val.bin (uint16) + meta.json.

  --dataset custom        all .txt files under --raw_dir
  --dataset openwebtext   HuggingFace OpenWebText, streamed (use --max_docs to limit)

Documents are separated by the GPT-2 end-of-text token, and tokens are streamed to disk in chunks,
so memory stays flat however large the dataset is. The last --val_fraction of tokens is validation.

--compact_vocab remaps the ids to only the tokens that occur in the data (e.g. 50,257 -> ~20k for
Shakespeare). The output layer shrinks accordingly; meta.json stores the map so sampling can decode.
"""
import argparse
import json
import os
from collections.abc import Iterable

import numpy as np
from tqdm import tqdm

GPT2_VOCAB = 50257
EOT = 50256
CHUNK = 1 << 22  # tokens per copy chunk


def get_encoder():
    """GPT-2 BPE: tiktoken if installed (fast), else HuggingFace transformers."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        return enc.encode_ordinary
    except ImportError:
        from transformers import GPT2TokenizerFast

        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        return lambda t: tok.encode(t)


def iter_text_from_dir(raw_dir: str) -> Iterable[str]:
    """Yield text from all .txt files under raw_dir (recursively, sorted for reproducibility)."""
    raw_dir = os.path.abspath(raw_dir)
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"raw_dir does not exist or is not a directory: {raw_dir}")
    for root, dirs, files in os.walk(raw_dir):
        dirs.sort()
        for fname in sorted(files):
            if not fname.lower().endswith(".txt"):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, encoding="utf-8") as f:
                    txt = f.read()
            except UnicodeDecodeError:
                with open(path, encoding="latin-1") as f:
                    txt = f.read()
            if txt.strip():
                yield txt


def iter_text_openwebtext(max_docs: int | None = None) -> Iterable[str]:
    """Stream OpenWebText from the HuggingFace Hub (Parquet-backed 'dylanebert/openwebtext')."""
    from datasets import load_dataset

    try:
        ds = load_dataset("dylanebert/openwebtext", split="train", streaming=True)
    except Exception as e:
        raise RuntimeError(
            "Could not stream 'dylanebert/openwebtext' from the Hugging Face Hub. Check internet access and "
            f"the `datasets` version, or use --dataset custom.\n\nOriginal error:\n{e}"
        ) from e
    count = 0
    for row in ds:
        txt = row.get("text", "")
        if isinstance(txt, str) and txt.strip():
            yield txt
            count += 1
        if max_docs is not None and count >= max_docs:
            break


def stream_tokens_to_file(texts: Iterable[str], encode, path: str, flush_tokens: int = 1 << 20) -> int:
    """Tokenize each document, append EOT, and append uint16 ids to `path`. Returns the token count."""
    n, buf = 0, []
    with open(path, "wb") as f:
        for txt in tqdm(texts, desc="Encoding", dynamic_ncols=True, unit="doc"):
            ids = encode(txt)
            if not ids:
                continue
            buf.extend(ids)
            buf.append(EOT)
            if len(buf) >= flush_tokens:
                np.asarray(buf, dtype=np.uint16).tofile(f)
                n += len(buf)
                buf = []
        if buf:
            np.asarray(buf, dtype=np.uint16).tofile(f)
            n += len(buf)
    return n


def write_split(src: np.memmap, start: int, stop: int, path: str, lut: np.ndarray | None):
    out = np.memmap(path, dtype=np.uint16, mode="w+", shape=(stop - start,))
    for i in range(start, stop, CHUNK):
        j = min(i + CHUNK, stop)
        block = np.asarray(src[i:j])
        out[i - start: j - start] = lut[block] if lut is not None else block
    out.flush()
    del out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="custom", choices=["custom", "openwebtext"])
    ap.add_argument("--raw_dir", default="data/raw", help="directory with .txt files (--dataset custom)")
    ap.add_argument("--out_dir", default="data/openwebtext", help="output directory for train.bin / val.bin")
    ap.add_argument("--val_fraction", type=float, default=0.01)
    ap.add_argument("--max_docs", type=int, default=None, help="limit the number of documents (openwebtext)")
    ap.add_argument("--compact_vocab", action="store_true", help="remap to the token ids present in the data")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    encode = get_encoder()
    if args.dataset == "openwebtext":
        print("Preparing dataset: openwebtext (HuggingFace)")
        texts = iter_text_openwebtext(max_docs=args.max_docs)
    else:
        print(f"Preparing dataset from local text files under: {args.raw_dir}")
        texts = iter_text_from_dir(args.raw_dir)

    all_path = os.path.join(args.out_dir, "all.tmp.bin")
    n = stream_tokens_to_file(texts, encode, all_path)
    print(f"Collected {n} tokens in total.")
    if n < 1000:
        os.remove(all_path)
        raise ValueError(f"Too few tokens ({n}) collected. Add more text or adjust dataset options.")

    n_val = min(max(1, int(n * args.val_fraction)), n - 1)
    n_train = n - n_val
    src = np.memmap(all_path, dtype=np.uint16, mode="r")

    lut, id_map, vocab_size = None, None, GPT2_VOCAB
    if args.compact_vocab:
        counts = np.zeros(GPT2_VOCAB, dtype=np.int64)
        for i in range(0, n, CHUNK):
            counts += np.bincount(np.asarray(src[i: i + CHUNK]), minlength=GPT2_VOCAB)
        counts[EOT] = max(counts[EOT], 1)
        id_map = np.flatnonzero(counts)  # compact id -> GPT-2 id
        lut = np.zeros(GPT2_VOCAB, dtype=np.uint16)
        lut[id_map] = np.arange(len(id_map), dtype=np.uint16)
        vocab_size = int(len(id_map))
        print(f"Compact vocabulary: {vocab_size} of {GPT2_VOCAB} GPT-2 tokens occur in the data.")

    write_split(src, 0, n_train, os.path.join(args.out_dir, "train.bin"), lut)
    write_split(src, n_train, n, os.path.join(args.out_dir, "val.bin"), lut)
    del src
    os.remove(all_path)

    meta = {"tokenizer": "gpt2", "vocab_size": vocab_size, "n_train": int(n_train), "n_val": int(n_val),
            "eot_id": int(lut[EOT]) if lut is not None else EOT,
            "id_map": id_map.tolist() if id_map is not None else None}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    print(f"Wrote {n_train} train tokens to {os.path.join(args.out_dir, 'train.bin')}")
    print(f"Wrote {n_val} val tokens to {os.path.join(args.out_dir, 'val.bin')}")
    if args.dataset == "openwebtext":
        # Hard exit to avoid a PyGILState_Release crash in HF/pyarrow streaming teardown.
        os._exit(0)


if __name__ == "__main__":
    main()
