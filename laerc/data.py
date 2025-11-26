import os
from typing import Tuple

import numpy as np
import torch


def load_memmap_tokens(data_dir: str) -> Tuple[np.memmap, np.memmap]:
    train_tokens = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_tokens = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return train_tokens, val_tokens


def get_batch(
    train_tokens: np.memmap,
    seq: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_tokens = len(train_tokens)
    if num_tokens <= seq + 1:
        raise ValueError(f"Not enough tokens in train.bin ({num_tokens}) for seq={seq}")

    x_np = np.empty((batch_size, seq - 1), dtype=np.int64)
    y_np = np.empty((batch_size, seq - 1), dtype=np.int64)

    max_start = num_tokens - seq - 1
    starts = np.random.randint(0, max_start, size=batch_size)

    for i, s in enumerate(starts):
        chunk = train_tokens[s : s + seq]
        x_np[i] = chunk[:-1]
        y_np[i] = chunk[1:]

    x = torch.from_numpy(x_np).long().to(device)
    y = torch.from_numpy(y_np).long().to(device)
    return x, y


