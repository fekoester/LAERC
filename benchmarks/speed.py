"""Forward + backward throughput of the reservoir variants (no training, random tokens).

    python benchmarks/speed.py --emb 256 --layers 8 --seq 1024 --batch 8      # on a GPU
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from laerc.model import ReservoirFFNLanguageModel  # noqa: E402

VARIANTS = {
    "ffn_only": dict(use_reservoir=False),
    "tanh": dict(),
    "tanh_leaky": dict(res_leak_min=0.01),
    "diag": dict(res_mode="diag", res_leak_min=0.01),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {dev}{' (' + torch.cuda.get_device_name() + ')' if dev.type == 'cuda' else ''}, "
          f"D={a.emb} L={a.layers} T={a.seq} B={a.batch}")
    amp_dtype = torch.bfloat16 if dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    for name in a.variants:
        torch.manual_seed(0)
        m = ReservoirFFNLanguageModel(a.vocab, a.emb, a.layers, device=dev, **VARIANTS[name]).to(dev)
        idx = torch.randint(0, a.vocab, (a.batch, a.seq), device=dev)
        times = []
        try:
            for i in range(a.iters + 2):
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t = time.perf_counter()
                with torch.autocast(dev.type, dtype=amp_dtype, enabled=a.amp and dev.type == "cuda"):
                    _, loss = m(idx, idx)
                loss.backward()
                m.zero_grad(set_to_none=True)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                if i >= 2:
                    times.append(time.perf_counter() - t)
        except torch.OutOfMemoryError:
            print(f"{name:11s} out of memory")
            del m
            torch.cuda.empty_cache()
            continue
        dt = sorted(times)[len(times) // 2]
        print(f"{name:11s} {dt * 1000:8.1f} ms/step  {a.batch * a.seq / dt:10,.0f} tokens/s")


if __name__ == "__main__":
    main()
