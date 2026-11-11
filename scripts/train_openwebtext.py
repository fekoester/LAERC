#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laerc import TrainConfig, finalize_config  # noqa: E402
from laerc.data import load_meta  # noqa: E402
from laerc.train import train  # noqa: E402
from laerc.utils import set_seed  # noqa: E402


def build_argparser():
    ap = argparse.ArgumentParser(description="Train LAERC language model on memmapped token dataset.")
    # data
    ap.add_argument("--data_dir", type=str, default="data/openwebtext")
    ap.add_argument("--vocab", type=int, default=50257)
    ap.add_argument("--seq", type=int, default=1024)

    # model
    ap.add_argument("--emb", type=int, default=512)
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--resv_mult", type=float, default=4.0)
    ap.add_argument("--res_mlp_mult", type=float, default=1.0)
    ap.add_argument("--ffn_mult", type=float, default=4.0)
    ap.add_argument("--res_radius", type=float, default=0.9)
    ap.add_argument("--res_scale", type=str, default="radius", choices=["radius", "norm"],
                    help="scale the reservoir's spectral radius (default) or, as in v0.1, its spectral norm")
    ap.add_argument("--no_reservoir", action="store_true", help="Disable reservoir path (pure FFN model)")

    # optimization
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps_per_epoch", type=int, default=1000,
                    help="Number of optimizer steps per epoch (<=0 → auto-compute from data size)")
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--beta2", type=float, default=0.999)

    # scheduler
    ap.add_argument("--no_sched", action="store_true", help="Disable LR scheduler")
    ap.add_argument("--warmup_frac", type=float, default=0.01)
    ap.add_argument("--hold_frac", type=float, default=0.1)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)

    # hardware
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast on CUDA")

    # evaluation / resume
    ap.add_argument("--eval_interval_steps", type=int, default=500, help="validation every N steps (0: per epoch)")
    ap.add_argument("--eval_batches", type=int, default=50)
    ap.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    ap.add_argument("--no_grad_accum_fp32", action="store_true")  # kept for config compatibility

    # logging / ckpt
    ap.add_argument("--tag", type=str, default=None, help="Optional manual run tag (otherwise auto-generated)")
    ap.add_argument("--ckpt_dir", type=str, default=None)
    ap.add_argument("--log_interval_steps", type=int, default=100)
    ap.add_argument("--flush_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)

    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        vocab=args.vocab,
        seq=args.seq,
        emb=args.emb,
        layers=args.layers,
        resv_mult=args.resv_mult,
        res_mlp_mult=args.res_mlp_mult,
        ffn_mult=args.ffn_mult,
        res_radius=args.res_radius,
        res_scale=args.res_scale,
        use_reservoir=not args.no_reservoir,
        batch=args.batch,
        accum=args.accum,
        lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        beta2=args.beta2,
        sched=not args.no_sched,
        warmup_frac=args.warmup_frac,
        hold_frac=args.hold_frac,
        min_lr_ratio=args.min_lr_ratio,
        compile=args.compile,
        amp=args.amp,
        eval_interval_steps=args.eval_interval_steps,
        eval_batches=args.eval_batches,
        resume=args.resume,
        grad_accum_fp32=not args.no_grad_accum_fp32,
        tag=args.tag,
        ckpt_dir=args.ckpt_dir,
        log_interval_steps=args.log_interval_steps,
        flush_every=args.flush_every,
        seed=args.seed,
    )

    meta = load_meta(cfg.data_dir)
    if meta.get("vocab_size"):
        cfg.vocab = int(meta["vocab_size"])  # compact vocabularies set their own size
    cfg = finalize_config(cfg)
    set_seed(cfg.seed)

    print(json.dumps(cfg.as_dict(), indent=2))
    out = train(cfg)
    print(f"best validation loss {out['best_val_loss']:.4f} (perplexity {out['best_val_ppl']:.2f})")


if __name__ == "__main__":
    main()

