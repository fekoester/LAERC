#!/usr/bin/env python3
import argparse
import os

import torch
import tiktoken

from laerc.config import TrainConfig
from laerc.model import ReservoirFFNLanguageModel
from laerc.utils import get_device, load_checkpoint


def build_argparser():
    ap = argparse.ArgumentParser(description="Sample text from a trained LAERC checkpoint.")
    ap.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint .pt file.")
    ap.add_argument("--ckpt_dir", type=str, default=None, help="Directory with checkpoints (pick latest).")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--prompt", type=str, default="")
    return ap

def load_latest_checkpoint_in_dir(ckpt_dir: str) -> str:
    files = [f for f in os.listdir(ckpt_dir)
             if f.startswith("model_step") and f.endswith(".pt")]
    if not files:
        raise FileNotFoundError(f"No model_step*.pt files found in {ckpt_dir}")
    steps = []
    for f in files:
        try:
            # f is like "model_step400.pt"
            step_str = f[len("model_step") :].split(".")[0]
            step = int(step_str)
        except Exception:
            step = -1
        steps.append((step, f))
    steps.sort()
    return os.path.join(ckpt_dir, steps[-1][1])


@torch.no_grad()
def sample_tokens(
    model: ReservoirFFNLanguageModel,
    enc,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
):
    model.eval()
    if prompt:
        idx = enc.encode(prompt)
    else:
        idx = [enc.eot_token] if hasattr(enc, "eot_token") else [0]
    idx = torch.tensor(idx, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_tokens):
        logits, _ = model(idx)
        logits = logits[:, -1, :] / max(1e-4, temperature)
        if top_k > 0:
            v, ix = torch.topk(logits, top_k, dim=-1)
            probs = torch.zeros_like(logits).scatter_(-1, ix, torch.softmax(v, dim=-1))
        else:
            probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return enc.decode(idx[0].tolist())


def main():
    ap = build_argparser()
    args = ap.parse_args()

    if args.ckpt is None:
        if args.ckpt_dir is None:
            raise ValueError("Either --ckpt or --ckpt_dir must be provided.")
        args.ckpt = load_latest_checkpoint_in_dir(args.ckpt_dir)

    device = get_device()
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    cfg_dict = ckpt["config"]
    cfg = TrainConfig(**cfg_dict)

    enc = tiktoken.get_encoding("gpt2")

    model = ReservoirFFNLanguageModel(
        vocab_size=cfg.vocab,
        D=cfg.emb,
        num_layers=cfg.layers,
        reservoir_mult=cfg.resv_mult,
        reservoir_mlp_mult=cfg.res_mlp_mult,
        ff_mult=cfg.ffn_mult,
        reservoir_radius=cfg.res_radius,
        use_reservoir=cfg.use_reservoir,
        device=device,
        dtype=torch.float32,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    text = sample_tokens(
        model=model,
        enc=enc,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()

