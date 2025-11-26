import math
import os
import time

import torch
from tqdm import tqdm

from .config import TrainConfig
from .data import load_memmap_tokens, get_batch
from .model import ReservoirFFNLanguageModel, WarmHoldCosineLR, build_optimizer
from .utils import CSVLogger, get_amp_dtype_and_ctx, get_device, save_checkpoint


def train(cfg: TrainConfig):
    device = get_device()
    amp_dtype, amp_ctx, use_scaler = get_amp_dtype_and_ctx(device)  # use_scaler is False in current utils

    # load data
    train_tokens, val_tokens = load_memmap_tokens(cfg.data_dir)
    num_tokens = len(train_tokens)
    if num_tokens <= cfg.seq + 1:
        raise ValueError(f"Not enough tokens in train.bin ({num_tokens}) for seq={cfg.seq}")

    # model
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
        dtype=amp_dtype,
    ).to(device)

    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    # --- print parameter count ---
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: total={n_params:,} | trainable={n_trainable:,}")

    # optimizer
    opt = build_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # steps per epoch
    if cfg.steps_per_epoch <= 0:
        # simple heuristic: one "epoch" ≈ one pass over data at this batch/seq
        steps_per_epoch = max(1, num_tokens // (cfg.batch * (cfg.seq - 1)))
    else:
        steps_per_epoch = cfg.steps_per_epoch

    total_steps = cfg.epochs * steps_per_epoch

    # scheduler
    if cfg.sched:
        sched = WarmHoldCosineLR(
            opt,
            total_updates=total_steps,
            warmup_frac=cfg.warmup_frac,
            hold_frac=cfg.hold_frac,
            min_lr_ratio=cfg.min_lr_ratio,
        )
    else:
        sched = None

    # logging
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    csv_path = os.path.join(cfg.ckpt_dir, "loss_log.csv")
    logger = CSVLogger(csv_path, flush_every=cfg.flush_every)

    global_step = 0
    model.train()

    for ep in range(1, cfg.epochs + 1):
        pbar = tqdm(total=steps_per_epoch, desc=f"Epoch {ep}", dynamic_ncols=True)
        cur_lr = cfg.lr

        # running average for display
        running_loss = 0.0
        running_loss_steps = 0

        # accumulation for CSV logging window
        window_loss = 0.0
        window_steps = 0

        for step_in_epoch in range(1, steps_per_epoch + 1):
            opt.zero_grad(set_to_none=True)

            with amp_ctx:
                for _ in range(cfg.accum):
                    xb, yb = get_batch(train_tokens, cfg.seq, cfg.batch, device)
                    logits, loss = model(xb, yb)        # full CE loss

                    # --- logging uses the unscaled loss ---
                    raw_loss = loss.item()
                    running_loss += raw_loss
                    running_loss_steps += 1
                    window_loss += raw_loss
                    window_steps += 1

                    # --- scale only for gradient accumulation ---
                    loss = loss / cfg.accum
                    loss.backward()

            if cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            opt.step()
            if sched is not None:
                sched.step()
                cur_lr = opt.param_groups[0]["lr"]

            global_step += 1

            # update progress bar every step with running avg
            avg_disp_loss = running_loss / max(1, running_loss_steps)
            pbar.set_postfix(loss=f"{avg_disp_loss:.4f}", lr=f"{cur_lr:.2e}")

            # log to CSV every N optimizer steps (average over window)
            if (global_step % cfg.log_interval_steps) == 0 and window_steps > 0:
                avg_window_loss = window_loss / window_steps
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                logger.log_row(
                    [global_step, ep, step_in_epoch,
                     f"{avg_window_loss:.6f}", f"{cur_lr:.6e}", timestamp]
                )
                window_loss = 0.0
                window_steps = 0

            pbar.update(1)

        pbar.close()

        # checkpoint at end of epoch
        save_checkpoint(cfg.ckpt_dir, global_step, model, opt, sched, cfg)

    logger.flush()

