import os
from dataclasses import asdict, dataclass


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
    res_scale: str = "radius"  # "radius" (true spectral radius) | "norm" (v0.1 behaviour)
    use_reservoir: bool = True
    res_leak_min: float = 1.0  # < 1: leaky multi-timescale reservoir, leak rates log-spaced in [res_leak_min, 1]
    res_mode: str = "tanh"  # "tanh": nonlinear ESN (original) | "diag": frozen linear modal reservoir, parallel in T
    res_depth_growth: float = 1.0  # > 1: block k's timescales are res_depth_growth**k times longer (geometric depth)

    # optimization
    batch: int = 8
    accum: int = 8
    lr: float = 4e-4
    epochs: int = 1
    steps_per_epoch: int = 0  # if <= 0 → auto-compute from data size
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta2: float = 0.999

    # scheduler (interpreted over total_steps = epochs * steps_per_epoch)
    sched: bool = True
    warmup_frac: float = 0.01
    hold_frac: float = 0.1
    min_lr_ratio: float = 0.1

    # hardware
    compile: bool = False
    amp: bool = False  # autocast on CUDA: bf16 on Ampere+, fp16 with loss scaling on older GPUs (T4, P100)
    grad_accum_fp32: bool = True  # kept for compatibility (unused)

    # evaluation
    eval_interval_steps: int = 500  # validation loss every N optimizer steps (0: only at epoch end)
    eval_batches: int = 50          # number of validation windows of `batch` sequences
    resume: str | None = None    # checkpoint to resume from (model, optimizer, scheduler, step)

    # logging / ckpt
    tag: str | None = None
    ckpt_dir: str | None = None
    log_interval_steps: int = 100  # log every N optimizer steps
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
    if cfg.res_leak_min < 1.0:
        parts.append(f"leak{cfg.res_leak_min:g}")
    if cfg.res_mode != "tanh":
        parts.append(cfg.res_mode)
    if cfg.res_depth_growth != 1.0:
        parts.append(f"G{cfg.res_depth_growth:g}")
    return "_".join(parts)


def finalize_config(cfg: TrainConfig) -> TrainConfig:
    if cfg.tag is None:
        cfg.tag = make_run_name(cfg)
    if cfg.ckpt_dir is None:
        cfg.ckpt_dir = os.path.join("checkpoints", cfg.tag)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    return cfg

