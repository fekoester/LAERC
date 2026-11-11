#!/usr/bin/env python3
"""
Sample text from a trained LAERC checkpoint.

Generation carries the reservoir states forward, so each new token costs one step of the network
instead of re-running the whole sequence (the model is recurrent; everything else is per token).

    python -m scripts.sample --ckpt_dir checkpoints/<run> --prompt "ROMEO:" --max_tokens 200 --top_p 0.9
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from laerc.model import build_model  # noqa: E402
from laerc.utils import get_device, load_checkpoint  # noqa: E402


def build_argparser():
    ap = argparse.ArgumentParser(description="Sample text from a trained LAERC checkpoint.")
    ap.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint .pt file.")
    ap.add_argument("--ckpt_dir", type=str, default=None,
                    help="Directory with checkpoints (best.pt if present, else the latest step).")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50, help="0 disables")
    ap.add_argument("--top_p", type=float, default=1.0, help="nucleus sampling; 1.0 disables")
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num_samples", type=int, default=1)
    return ap


def load_latest_checkpoint_in_dir(ckpt_dir: str) -> str:
    if os.path.exists(os.path.join(ckpt_dir, "best.pt")):
        return os.path.join(ckpt_dir, "best.pt")
    files = [f for f in os.listdir(ckpt_dir) if f.startswith("model_step") and f.endswith(".pt")]
    if not files:
        raise FileNotFoundError(f"No best.pt or model_step*.pt files found in {ckpt_dir}")
    steps = []
    for f in files:
        try:
            step = int(f[len("model_step"):].split(".")[0])
        except ValueError:
            step = -1
        steps.append((step, f))
    steps.sort()
    return os.path.join(ckpt_dir, steps[-1][1])


def filter_logits(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    """Top-k and nucleus (top-p) filtering on [B, V] logits."""
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(-1)
        remove = cum - torch.softmax(sorted_logits, dim=-1) > top_p  # keep the token that crosses top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)
    return logits


@torch.no_grad()
def generate(model, prompt_ids, max_tokens, temperature=1.0, top_k=0, top_p=1.0, generator=None):
    """Incremental generation: feed the prompt once, then one token at a time with carried state."""
    model.eval()
    idx = prompt_ids
    logits, _, state = model(idx, return_state=True)
    out = [idx]
    for _ in range(max_tokens):
        nxt_logits = filter_logits(logits[:, -1, :].float() / max(1e-4, temperature), top_k, top_p)
        nxt = torch.multinomial(torch.softmax(nxt_logits, dim=-1), 1, generator=generator)
        out.append(nxt)
        logits, _, state = model(nxt, state=state, return_state=True)
    return torch.cat(out, dim=1)


def main():
    args = build_argparser().parse_args()
    if args.ckpt is None:
        if args.ckpt_dir is None:
            raise ValueError("Either --ckpt or --ckpt_dir must be provided.")
        args.ckpt = load_latest_checkpoint_in_dir(args.ckpt_dir)

    import tiktoken

    device = get_device()
    ckpt = load_checkpoint(args.ckpt, map_location=device)
    model = build_model(ckpt["config"], device=device, dtype=torch.float32).to(device)
    model.load_state_dict(ckpt["model"])

    enc = tiktoken.get_encoding("gpt2")
    id_map = (ckpt.get("meta") or {}).get("id_map")
    g2c = {g: c for c, g in enumerate(id_map)} if id_map is not None else None

    def to_model(ids):  # GPT-2 ids -> model ids (tokens unseen in training are dropped)
        return ids if g2c is None else [g2c[i] for i in ids if i in g2c]

    def to_gpt2(ids):
        return ids if id_map is None else [id_map[i] for i in ids]

    prompt = to_model(enc.encode_ordinary(args.prompt)) if args.prompt else []
    if not prompt:
        prompt = to_model([enc.eot_token])
    gen = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    for i in range(args.num_samples):
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        out = generate(model, ids, args.max_tokens, args.temperature, args.top_k, args.top_p, gen)
        if args.num_samples > 1:
            print(f"--- sample {i + 1} ---")
        print(enc.decode(to_gpt2(out[0].tolist())))


if __name__ == "__main__":
    main()
