import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm

from .config import TrainConfig
from .data import eval_windows, get_batch, load_memmap_tokens, load_meta
from .model import WarmHoldCosineLR, build_model, build_optimizer
from .utils import CSVLogger, get_device, load_checkpoint, save_checkpoint, unwrap


@torch.no_grad()
def evaluate(model, val_tokens, cfg: TrainConfig, device, amp_ctx=None) -> float:
    """Mean cross-entropy over fixed, evenly spaced validation windows."""
    was_training = model.training
    model.eval()
    tot, n = 0.0, 0
    for x, y in eval_windows(val_tokens, cfg.seq, cfg.batch, cfg.eval_batches):
        x, y = x.to(device), y.to(device)
        with amp_ctx or nullcontext():
            _, loss = model(x, y)
        tot += loss.item() * y.numel()
        n += y.numel()
    if was_training:
        model.train()
    return tot / max(n, 1)


def train(cfg: TrainConfig, model=None):
    """Train; `model` overrides the LAERC model (any module with forward(idx, targets) -> (logits, loss))."""
    device = get_device()
    use_amp = bool(cfg.amp) and device.type == "cuda"
    amp_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    # data
    train_tokens, val_tokens = load_memmap_tokens(cfg.data_dir)
    meta = load_meta(cfg.data_dir)
    if meta.get("vocab_size") and meta["vocab_size"] != cfg.vocab:
        print(f"[data] meta.json says vocab_size={meta['vocab_size']}; using it instead of --vocab {cfg.vocab}")
        cfg.vocab = int(meta["vocab_size"])
    num_tokens = len(train_tokens)
    if num_tokens <= cfg.seq + 1:
        raise ValueError(f"Not enough tokens in train.bin ({num_tokens}) for seq={cfg.seq}")

    # model (always float32 parameters; bf16 only inside autocast)
    model = (model if model is not None else build_model(cfg, device=device, dtype=torch.float32)).to(device)
    fwd = torch.compile(model) if (cfg.compile and hasattr(torch, "compile")) else model

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_embed = model.embed.weight.numel()
    print(f"Model parameters: total={n_params:,} | trainable={n_trainable:,} | embedding (tied)={n_embed:,}")

    opt = build_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay, beta2=cfg.beta2)

    if cfg.steps_per_epoch <= 0:
        steps_per_epoch = max(1, num_tokens // (cfg.batch * cfg.accum * (cfg.seq - 1)))
    else:
        steps_per_epoch = cfg.steps_per_epoch
    total_steps = cfg.epochs * steps_per_epoch

    sched = None
    if cfg.sched:
        sched = WarmHoldCosineLR(opt, total_updates=total_steps, warmup_frac=cfg.warmup_frac,
                                 hold_frac=cfg.hold_frac, min_lr_ratio=cfg.min_lr_ratio)

    global_step, best_val = 0, math.inf
    if cfg.resume:
        ck = load_checkpoint(cfg.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if sched is not None and ck.get("scheduler") is not None:
            sched.load_state_dict(ck["scheduler"])
        global_step = int(ck["step"])
        best_val = float(ck.get("best_val", math.inf))
        if ck.get("np_rng") is not None:
            np.random.set_state(ck["np_rng"])
        print(f"[resume] from {cfg.resume} at step {global_step}")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    logger = CSVLogger(os.path.join(cfg.ckpt_dir, "loss_log.csv"), flush_every=cfg.flush_every)
    val_logger = CSVLogger(os.path.join(cfg.ckpt_dir, "val_log.csv"), flush_every=1,
                           header=["global_step", "val_loss", "val_ppl", "train_tokens_seen", "timestamp"])
    tokens_per_step = cfg.batch * cfg.accum * (cfg.seq - 1)

    def run_eval():
        nonlocal best_val
        v = evaluate(model, val_tokens, cfg, device, amp_ctx)
        val_logger.log_row([global_step, f"{v:.6f}", f"{math.exp(min(v, 50)):.3f}", global_step * tokens_per_step,
                            time.strftime("%Y-%m-%d %H:%M:%S")])
        improved = v < best_val
        if improved:
            best_val = v
            save_checkpoint(cfg.ckpt_dir, global_step, model, opt, sched, cfg, name="best.pt", best_val=best_val,
                            meta=meta, quiet=True)
        tqdm.write(f"[val] step {global_step}: loss={v:.4f} ppl={math.exp(min(v, 50)):.2f}{' *' if improved else ''}")
        return v

    start_epoch = global_step // steps_per_epoch + 1
    model.train()
    for ep in range(start_epoch, cfg.epochs + 1):
        first = global_step - (ep - 1) * steps_per_epoch + 1
        pbar = tqdm(total=steps_per_epoch, initial=first - 1, desc=f"Epoch {ep}", dynamic_ncols=True)
        cur_lr = opt.param_groups[0]["lr"]
        running_loss, running_steps = 0.0, 0
        window_loss, window_steps = 0.0, 0

        for step_in_epoch in range(first, steps_per_epoch + 1):
            opt.zero_grad(set_to_none=True)
            for _ in range(cfg.accum):
                xb, yb = get_batch(train_tokens, cfg.seq, cfg.batch, device)
                with amp_ctx:
                    _, loss = fwd(xb, yb)
                raw = loss.item()
                running_loss += raw
                running_steps += 1
                window_loss += raw
                window_steps += 1
                (loss / cfg.accum).backward()

            if cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if sched is not None:
                sched.step()
                cur_lr = opt.param_groups[0]["lr"]
            global_step += 1

            pbar.set_postfix(loss=f"{running_loss / max(1, running_steps):.4f}", lr=f"{cur_lr:.2e}")
            if (global_step % cfg.log_interval_steps) == 0 and window_steps > 0:
                logger.log_row([global_step, ep, step_in_epoch, f"{window_loss / window_steps:.6f}",
                                f"{cur_lr:.6e}", time.strftime("%Y-%m-%d %H:%M:%S")])
                window_loss, window_steps = 0.0, 0
            if cfg.eval_interval_steps > 0 and global_step % cfg.eval_interval_steps == 0:
                run_eval()
            pbar.update(1)
        pbar.close()

        if cfg.eval_interval_steps <= 0 or global_step % cfg.eval_interval_steps != 0:
            run_eval()
        save_checkpoint(cfg.ckpt_dir, global_step, model, opt, sched, cfg, best_val=best_val, meta=meta)

    logger.flush()
    val_logger.flush()
    return {"best_val_loss": best_val, "best_val_ppl": math.exp(min(best_val, 50)), "steps": global_step,
            "params_total": n_params, "params_trainable": n_trainable, "params_embedding": n_embed,
            "model": unwrap(model)}
