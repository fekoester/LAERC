import csv
import os
import random
from contextlib import nullcontext
from typing import Iterable, List

import numpy as np
import torch

from .config import TrainConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_amp_dtype_and_ctx(device: torch.device):
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        use_scaler = (amp_dtype == torch.float16)
    else:
        amp_dtype = torch.float32
        amp_ctx = nullcontext()
        use_scaler = False
    return amp_dtype, amp_ctx, use_scaler


class CSVLogger:
    def __init__(self, path: str, flush_every: int = 100):
        self.path = path
        self.flush_every = flush_every
        self.buffer: List[List[str]] = []
        self._initialized = os.path.exists(path)

    def log_row(self, row: Iterable):
        self.buffer.append(list(map(str, row)))
        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        file_exists = os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            if (not file_exists) and (not self._initialized):
                writer.writerow(["global_batch", "epoch", "step_in_epoch", "loss", "lr", "timestamp"])
                self._initialized = True
            writer.writerows(self.buffer)
        self.buffer.clear()


def save_checkpoint(
    ckpt_dir: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: TrainConfig,
):
    ckpt_file = os.path.join(ckpt_dir, f"model_{step}.pt")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "config": cfg.as_dict(),
        },
        ckpt_file,
    )
    print(f"[checkpoint] Saved: {ckpt_file}")


def load_checkpoint(path: str, map_location=None):
    return torch.load(path, map_location=map_location)

