from dataclasses import dataclass, asdict
from typing import Optional
import os


@dataclass
class TrainConfig:
    # data
    data_dir: str = "data/openwebtext"
    vocab: int = 50257
    seq: int = 1024

    # model
    emb: int = 512
    layers: int = 16
    resv_mult: float = 4.0
    res_mlp_mult: float = 1.0
    ffn_mult: float = 4.0
    res_radius: float = 0.9
    use_reservoir: bool = True

    # optimization
    batch: int = 8
    accum: int = 8
    lr: float = 4e-4
    epochs: int = 1
    total_updates: int = 200_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta2: float = 0.999

    # scheduler
    sched: bool = True
    warmup_frac: float = 0.01
    hold_frac: float = 0.1
    min_lr_ratio: float = 0.1

    # hardware
    compile: bool = False
    grad_accum_fp32: bool = True

    # logging / ckpt
    tag: Optional[str] = None
    ckpt_dir: Optional[str] = None
    log_interval_tokens: int = 1_000_000
    flush_every: int = 100
    seed: int = 42

    def as_dict(self):
        return asdict(self)


def make_run_name(cfg: "TrainConfig") -> str:
    # dynamic, human-readable tag from hyperparams
    parts = [
        "laerc",
        f"v{cfg.vocab}",
        f"seq{cfg.seq}",
        f"d{cfg.emb}",
        f"L{cfg.layers}",
        f"R{cfg.resv_mult:g}",
        f"RMLP{cfg.res_mlp_mult:g}",
        f"FF{cfg.ffn_mult:g}",
    ]
    return "_".join(parts)


def finalize_config(cfg: TrainConfig) -> TrainConfig:
    if cfg.tag is None:
        cfg.tag = make_run_name(cfg)
    if cfg.ckpt_dir is None:
        cfg.ckpt_dir = os.path.join("checkpoints", cfg.tag)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    return cfg

