#!/usr/bin/env python3
"""
Shakespeare benchmark: validation perplexity at an equal step budget.

  ffn_only        --no_reservoir: no information flows between tokens (a context-free baseline)
  laerc_v01       reservoir scaled as in v0.1 (spectral norm 0.9 -> spectral radius ~0.47)
  laerc_r0.9      true spectral radius 0.9
  laerc_r0.99     true spectral radius 0.99
  gpt             small causal Transformer of the same width and depth (reference)

    python benchmarks/run_benchmark.py --data_dir data/shakespeare_compact --steps 1200
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from benchmarks.gpt_baseline import GPT  # noqa: E402
from laerc import TrainConfig, finalize_config  # noqa: E402
from laerc.data import load_meta  # noqa: E402
from laerc.train import train  # noqa: E402
from laerc.utils import set_seed  # noqa: E402

RUNS = {
    "ffn_only": dict(use_reservoir=False),
    "laerc_v01": dict(res_scale="norm", res_radius=0.9),
    "laerc_r0.9": dict(res_scale="radius", res_radius=0.9),
    "laerc_r0.99": dict(res_scale="radius", res_radius=0.99),
    "gpt": dict(),
}
LABELS = {
    "ffn_only": "FFN only (no context)",
    "laerc_v01": "LAERC v0.1 scaling (ρ≈0.47)",
    "laerc_r0.9": "LAERC, ρ = 0.9",
    "laerc_r0.99": "LAERC, ρ = 0.99",
    "gpt": "GPT (small Transformer)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/shakespeare_compact")
    ap.add_argument("--runs", nargs="+", default=list(RUNS))
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval_interval", type=int, default=200)
    ap.add_argument("--eval_batches", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/shakespeare")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, "results.json")
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    vocab = load_meta(a.data_dir).get("vocab_size", 50257)

    for name in a.runs:
        if name in results:
            continue
        cfg = TrainConfig(data_dir=a.data_dir, vocab=vocab, seq=a.seq, emb=a.emb, layers=a.layers, batch=a.batch,
                          accum=1, lr=a.lr, epochs=1, steps_per_epoch=a.steps, eval_interval_steps=a.eval_interval,
                          eval_batches=a.eval_batches, log_interval_steps=50, seed=a.seed,
                          ckpt_dir=os.path.join(a.out, "ckpt", name), tag=name)
        for k, v in RUNS[name].items():
            setattr(cfg, k, v)
        cfg = finalize_config(cfg)
        set_seed(a.seed)
        model = GPT(vocab, a.emb, a.layers, heads=4, max_len=a.seq) if name == "gpt" else None
        print(f"\n=== {name} ===", flush=True)
        out = train(cfg, model=model)
        m = out.pop("model")
        out["params_non_embedding"] = sum(p.numel() for n, p in m.named_parameters()
                                          if p.requires_grad and "embed" not in n)
        with open(os.path.join(cfg.ckpt_dir, "val_log.csv")) as f:
            out["val_curve"] = [(int(r["global_step"]), float(r["val_loss"])) for r in csv.DictReader(f)]
        results[name] = out
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2)

    lines = ["| model | non-embedding params | best val loss | val perplexity |", "|---|---|---|---|"]
    for name in RUNS:
        if name in results:
            r = results[name]
            lines.append(f"| {LABELS[name]} | {r['params_non_embedding']:,} | {r['best_val_loss']:.3f} | "
                         f"{r['best_val_ppl']:.1f} |")
    md = "\n".join(lines)
    open(os.path.join(a.out, "results.md"), "w").write(md + "\n")
    print(md)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for name in RUNS:
        if name in results:
            s, v = zip(*results[name]["val_curve"])
            ax.plot(s, v, marker="o", ms=3, label=LABELS[name])
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("validation loss (nats/token)")
    ax.set_title("Shakespeare, GPT-2 tokens (compact vocab)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "val_curves.png"), dpi=130)


if __name__ == "__main__":
    main()
