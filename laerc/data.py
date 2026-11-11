import json
import os

import numpy as np
import torch


def load_memmap_tokens(data_dir: str) -> tuple[np.memmap, np.memmap]:
    train_tokens = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_tokens = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return train_tokens, val_tokens


def load_meta(data_dir: str) -> dict:
    """meta.json written by prepare_gpt2_tokens.py (vocab size and, for compact vocab, the id map)."""
    path = os.path.join(data_dir, "meta.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def get_batch(
    tokens: np.memmap,
    seq: int,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random windows of `seq` tokens: inputs are the first seq-1, targets the last seq-1."""
    num_tokens = len(tokens)
    if num_tokens <= seq + 1:
        raise ValueError(f"Not enough tokens ({num_tokens}) for seq={seq}")
    max_start = num_tokens - seq - 1
    starts = (rng.integers(0, max_start, size=batch_size) if rng is not None
              else np.random.randint(0, max_start, size=batch_size))
    chunk = np.stack([np.asarray(tokens[s: s + seq]) for s in starts]).astype(np.int64)
    x = torch.from_numpy(chunk[:, :-1])
    y = torch.from_numpy(chunk[:, 1:])
    if device.type == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def eval_windows(tokens: np.memmap, seq: int, batch_size: int, n_batches: int):
    """Deterministic, evenly spaced validation windows (the same ones at every evaluation)."""
    n_win = n_batches * batch_size
    max_start = len(tokens) - seq - 1
    if max_start <= 0:
        raise ValueError(f"validation set too small for seq={seq}")
    starts = np.linspace(0, max_start, num=min(n_win, max_start + 1)).astype(np.int64)
    for i in range(0, len(starts), batch_size):
        chunk = np.stack([np.asarray(tokens[s: s + seq]) for s in starts[i: i + batch_size]]).astype(np.int64)
        yield torch.from_numpy(chunk[:, :-1]), torch.from_numpy(chunk[:, 1:])


def chars_per_token(data_dir: str) -> float | None:
    """Average characters per validation token, to turn nats/token into bits/char.

    BPE datasets store val_chars in meta.json; for GPT-2 datasets the validation tokens are decoded.
    """
    meta = load_meta(data_dir)
    if meta.get("val_chars") and meta.get("n_val"):
        return meta["val_chars"] / meta["n_val"]
    try:
        import tiktoken
    except ImportError:
        return None
    val = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16).astype(np.int64)
    id_map = meta.get("id_map")
    if id_map is not None:
        val = np.asarray(id_map)[val]
    enc = tiktoken.get_encoding("gpt2")
    text = enc.decode([int(i) for i in val if i != enc.eot_token])
    return len(text) / len(val)
