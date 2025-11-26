import math
import os
import time

import torch
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from .config import TrainConfig
from .data import load_memmap_tokens, get_batch
from .model import ReservoirFFNLanguageModel, WarmHoldCosineLR, build_optimizer
from .utils import CSVLogger, get_amp_dtype_and_ctx, get_device, save_checkpoint


def train(cfg: TrainConfig):
    device = get_device()
    amp_dtype, amp_ctx, use_scaler = get_amp_dtype_and_ctx(device)
    scaler = GradScaler(enabled=use_scaler)

    train_tokens, val_tokens = load_memmap_tokens(cfg.data_dir)
    num_tokens = len(train_tokens)
    if num_tokens <= cfg.seq + 1:
        raise ValueError(f"Not enough tokens in train.bin ({num_tokens}) for seq={cfg.seq}")

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
        dtype=amp_dtype if device.type == "cuda" else torch.float32,
    ).to(device)

    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt = build_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.sched:
        sched = WarmHoldCosineLR(
            opt,
            total_updates=cfg.total_updates,
            warmup_frac=cfg.warmup_frac,
            hold_frac=cfg.hold_frac,
            min_lr_ratio=cfg.min_lr_ratio,
        )
    else:
        sched = None

    csv_path = os.path.join(cfg.ckpt_dir, "loss_log.csv")
    logger = CSVLogger(csv_path, flush_every=cfg.flush_every)

    raw_batches = 0
    global_update = 0
    total_batches_est = max(1, int(math.ceil(cfg.total_updates * cfg.accum * 1.0)))

    model.train()
    for ep in range(1, cfg.epochs + 1):
        pbar = tqdm(total=total_batches_est, desc=f"Epoch {ep}", dynamic_ncols=True)
        micro_loss = 0.0
        step_in_epoch = 0
        cur_lr = cfg.lr

        while global_update < cfg.total_updates:
            opt.zero_grad(set_to_none=True)

            with amp_ctx:
                for _ in range(cfg.accum):
                    xb, yb = get_batch(train_tokens, cfg.seq, cfg.batch, device)
                    logits, loss = model(xb, yb)
                    loss = loss / cfg.accum
                    micro_loss += loss.item()
                    if use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    raw_batches += 1

            if cfg.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            if use_scaler:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

            if sched is not None:
                sched.step()
                cur_lr = opt.param_groups[0]["lr"]

            global_update += 1
            step_in_epoch += 1

            # log every ~log_interval_tokens
            if raw_batches * cfg.batch * (cfg.seq - 1) >= cfg.log_interval_tokens:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                logger.log_row(
                    [raw_batches, ep, step_in_epoch,
                     f"{micro_loss:.6f}", f"{cur_lr:.6e}", timestamp]
                )
                pbar.set_postfix(loss=f"{micro_loss:.4f}", lr=f"{cur_lr:.2e}", upd=global_update)
                micro_loss = 0.0

            if global_update % 25000 == 0:
                save_checkpoint(cfg.ckpt_dir, global_update, model, opt, sched, cfg)

            pbar.update(1)
            if global_update >= cfg.total_updates:
                break

        pbar.close()

    logger.flush()
    save_checkpoint(cfg.ckpt_dir, global_update, model, opt, sched, cfg)

