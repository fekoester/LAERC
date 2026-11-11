#!/usr/bin/env python3
"""
Prepare a dataset with a small byte-level BPE trained on the data itself: train.bin / val.bin (uint16)
+ meta.json (which embeds the tokenizer, so checkpoints can decode on their own).

The text of all .txt files under --raw_dir is joined (documents separated by <|endoftext|>), the last
--val_fraction of *characters* becomes validation, and the BPE is trained on the training part only.

    python scripts/prepare_bpe_tokens.py --raw_dir data/raw --out_dir data/shakespeare_bpe1024 --vocab_size 1024

With V = 1024 the tied embedding costs 1024*D parameters instead of ~20k*D for compact GPT-2 tokens.
Losses per token are then not comparable across tokenizers; compare bits per character instead
(meta.json stores val_chars and n_val, the benchmark reports bpc).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.prepare_gpt2_tokens import iter_text_from_dir  # noqa: E402

EOT = "<|endoftext|>"


def train_bpe(text: str, vocab_size: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOT], show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    # train on chunks (by line blocks) so the trainer doesn't see one giant string
    lines = text.splitlines(keepends=True)
    chunks = ["".join(lines[i: i + 1000]) for i in range(0, len(lines), 1000)]
    tok.train_from_iterator(chunks, trainer=trainer)
    return tok


def encode(tok, text: str) -> np.ndarray:
    eot = tok.token_to_id(EOT)
    ids = []
    for i, doc in enumerate(text.split(EOT)):
        if i:
            ids.append(eot)
        lines = doc.splitlines(keepends=True)
        # encode in line blocks (ByteLevel pre-tokenization never merges across them in a harmful way:
        # block boundaries fall on newlines, which already split pre-tokens)
        for enc in tok.encode_batch(["".join(lines[j: j + 1000]) for j in range(0, len(lines), 1000)]):
            ids.extend(enc.ids)
    return np.asarray(ids, dtype=np.uint16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/shakespeare_bpe1024")
    ap.add_argument("--vocab_size", type=int, default=1024)
    ap.add_argument("--val_fraction", type=float, default=0.01)
    args = ap.parse_args()

    text = EOT.join(iter_text_from_dir(args.raw_dir))
    n_val_chars = max(1, int(len(text) * args.val_fraction))
    cut = text.rfind("\n", 0, len(text) - n_val_chars) + 1 or len(text) - n_val_chars  # split at a line start
    train_text, val_text = text[:cut], text[cut:]

    tok = train_bpe(train_text, args.vocab_size)
    V = tok.get_vocab_size()
    train_ids, val_ids = encode(tok, train_text), encode(tok, val_text)
    assert tok.decode(val_ids.tolist(), skip_special_tokens=False) == val_text, "BPE round trip failed"

    os.makedirs(args.out_dir, exist_ok=True)
    train_ids.tofile(os.path.join(args.out_dir, "train.bin"))
    val_ids.tofile(os.path.join(args.out_dir, "val.bin"))
    meta = {"tokenizer": "bpe", "vocab_size": V, "n_train": int(len(train_ids)), "n_val": int(len(val_ids)),
            "eot_id": tok.token_to_id(EOT), "id_map": None,
            "train_chars": len(train_text), "val_chars": len(val_text),
            "tokenizer_json": tok.to_str()}
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    print(f"BPE vocabulary: {V} tokens")
    print(f"Wrote {len(train_ids)} train tokens ({len(train_text) / len(train_ids):.2f} chars/token)")
    print(f"Wrote {len(val_ids)} val tokens ({len(val_text) / len(val_ids):.2f} chars/token)")


if __name__ == "__main__":
    main()
