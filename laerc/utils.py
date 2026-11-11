import csv
import os
import random
from collections.abc import Iterable
from contextlib import nullcontext

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


def get_amp_dtype_and_ctx(device: torch.device, enabled: bool = False):
    """bf16 autocast on CUDA when enabled (no GradScaler needed for bf16), float32 otherwise."""
    if enabled and device.type == "cuda":
        return torch.bfloat16, torch.autocast("cuda", dtype=torch.bfloat16), False
    return torch.float32, nullcontext(), False


class CSVLogger:
    def __init__(self, path: str, flush_every: int = 100, header: list[str] | None = None):
        self.path = path
        self.flush_every = flush_every
        self.header = header or ["global_step", "epoch", "step_in_epoch", "loss", "lr", "timestamp"]
        self.buffer: list[list[str]] = []
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
                writer.writerow(self.header)
                self._initialized = True
            writer.writerows(self.buffer)
        self.buffer.clear()


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """The original module behind torch.compile (so state_dict keys have no `_orig_mod.` prefix)."""
    return getattr(model, "_orig_mod", model)


def save_checkpoint(
    ckpt_dir: str,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: TrainConfig,
    name: str | None = None,
    best_val: float | None = None,
    meta: dict | None = None,
    quiet: bool = False,
):
    ckpt_file = os.path.join(ckpt_dir, name or f"model_step{step}.pt")
    torch.save(
        {
            "step": step,
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "config": cfg.as_dict(),
            "best_val": best_val,
            "meta": meta or {},
            "np_rng": np.random.get_state(),
        },
        ckpt_file,
    )
    if not quiet:
        print(f"[checkpoint] Saved: {ckpt_file}")


def load_checkpoint(path: str, map_location=None):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    # checkpoints written by v0.1 with --compile carry a `_orig_mod.` prefix
    ck["model"] = {k.replace("_orig_mod.", "", 1): v for k, v in ck["model"].items()}
    return ck
