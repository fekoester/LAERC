import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class WarmHoldCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """Warmup → hold → cosine decay scheduler."""
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_updates: int,
        warmup_frac: float = 0.01,
        hold_frac: float = 0.0,
        min_lr_ratio: float = 0.1,
        last_epoch: int = -1,
    ):
        self.total = max(1, int(total_updates))
        self.warm = max(1, int(warmup_frac * self.total))
        self.hold = max(0, int(hold_frac * self.total))
        self.min_r = float(min_lr_ratio)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch + 1
        base_lrs = self.base_lrs

        # warmup
        if step <= self.warm:
            s = step / max(1, self.warm)
            return [base * s for base in base_lrs]

        # hold
        if step <= self.warm + self.hold:
            return list(base_lrs)

        # cosine decay
        progress = min(1.0, (step - self.warm - self.hold) /
                       max(1, self.total - self.warm - self.hold))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base * (self.min_r + (1.0 - self.min_r) * cos) for base in base_lrs]


@torch.no_grad()
def spectral_norm_power_dense(W: torch.Tensor, iters: int = 30) -> float:
    """Power iteration spectral norm for a dense weight matrix."""
    device = W.device
    dtype = W.dtype
    v = torch.randn(W.shape[1], device=device, dtype=dtype)
    for _ in range(iters):
        v = F.normalize(W.t() @ (W @ v), dim=0, eps=1e-12)
    u = F.normalize(W @ v, dim=0, eps=1e-12)
    sigma = torch.dot(u, W @ v).abs().item()
    return sigma


class ReservoirBlock(nn.Module):
    """Single LAERC block: reservoir + gate + FFN."""
    def __init__(
        self,
        D: int,
        reservoir_mult: float = 4.0,
        reservoir_mlp_mult: float = 1.0,
        ff_mult: float = 4.0,
        reservoir_radius: float = 0.9,
        use_reservoir: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.D = D
        self.R = int(D * reservoir_mult)
        self.H_res = int(D * reservoir_mlp_mult)
        self.H_ff = int(D * ff_mult)
        self.use_reservoir = use_reservoir

        self.ln_in = nn.LayerNorm(D)
        param_dtype = dtype if dtype is not None else torch.float32

        # reservoir
        if use_reservoir and self.R > 0:
            rnn = nn.RNN(
                input_size=D,
                hidden_size=self.R,
                num_layers=1,
                nonlinearity="tanh",
                batch_first=True,
                bias=True,
            )
            rnn.to(device=device, dtype=param_dtype)
            rnn.flatten_parameters()

            with torch.no_grad():
                W = torch.randn(self.R, self.R, device=device, dtype=param_dtype) / math.sqrt(self.R)
                U = torch.randn(self.D, self.R, device=device, dtype=param_dtype) / math.sqrt(self.D)
                sigma = spectral_norm_power_dense(W, iters=30)
                if sigma > 0:
                    W.mul_(reservoir_radius / sigma)
                rnn.weight_ih_l0.copy_(U.t().contiguous())
                rnn.weight_hh_l0.copy_(W.t().contiguous())
                rnn.bias_ih_l0.zero_()
                rnn.bias_hh_l0.zero_()

            for p in rnn.parameters():
                p.requires_grad = False

            self.rnn = rnn
            self.res_mlp = nn.Sequential(
                nn.LayerNorm(self.R),
                nn.Linear(self.R, self.H_res),
                nn.GELU(),
                nn.Linear(self.H_res, D),
            )
            self.res_log_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.rnn = None
            self.res_mlp = None
            self.res_log_scale = None

        # input-dependent mixing gate
        self.mix_ln = nn.LayerNorm(2 * D)
        self.mix_gate = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.GELU(),
            nn.Linear(D, D),
            nn.Sigmoid(),
        )

        # FFN with ReZero-like scaling
        self.ln_ffn = nn.LayerNorm(D)
        self.ffn = nn.Sequential(
            nn.Linear(D, self.H_ff),
            nn.GELU(),
            nn.Linear(self.H_ff, D),
        )
        self.ffn_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        h = self.ln_in(x)

        if self.rnn is not None:
            r, _ = self.rnn(h)
            r = self.res_mlp(r)
            scale = torch.exp(self.res_log_scale)
            r = scale * r
        else:
            r = torch.zeros_like(h)

        mix_in = torch.cat([x, r], dim=-1)
        g = self.mix_gate(self.mix_ln(mix_in))
        m = g * r + (1.0 - g) * x

        ffn_out = self.ffn(self.ln_ffn(m))
        y = m + self.ffn_alpha * ffn_out
        return y


class ReservoirFFNLanguageModel(nn.Module):
    """Top-level LAERC LM: embedding + stacked ReservoirBlocks + output."""
    def __init__(
        self,
        vocab_size: int,
        D: int,
        num_layers: int,
        reservoir_mult: float = 4.0,
        reservoir_mlp_mult: float = 1.0,
        ff_mult: float = 4.0,
        reservoir_radius: float = 0.9,
        use_reservoir: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.V = vocab_size
        self.D = D

        self.embed = nn.Embedding(vocab_size, D)
        blocks = []
        for _ in range(num_layers):
            blk = ReservoirBlock(
                D=D,
                reservoir_mult=reservoir_mult,
                reservoir_mlp_mult=reservoir_mlp_mult,
                ff_mult=ff_mult,
                reservoir_radius=reservoir_radius,
                use_reservoir=use_reservoir,
                device=device,
                dtype=dtype,
            )
            blocks.append(blk)
        self.blocks = nn.ModuleList(blocks)
        self.norm_f = nn.LayerNorm(D)
        self.out = nn.Linear(D, vocab_size, bias=False)

        self._init_weights(tie_weights=True)

    def _init_weights(self, tie_weights: bool = True) -> None:
        with torch.no_grad():
            nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
            if tie_weights:
                self.out.weight = self.embed.weight
            else:
                nn.init.normal_(self.out.weight, mean=0.0, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    if m is self.out and tie_weights:
                        continue
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        """
        idx: (B, T) token ids
        targets: (B, T) or None
        """
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        logits = self.out(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.V),
                targets.view(-1),
            )
        return logits, loss


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    emb_lr_mult: float = 0.2,
):
    """AdamW with different decay for embeddings / norms / biases."""
    decay, nodecay, seen = [], [], set()
    for _, m in model.named_modules():
        for p_name, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            if p_name.endswith("bias") or isinstance(m, (nn.LayerNorm, nn.Embedding)):
                nodecay.append(p)
            elif isinstance(m, nn.Linear):
                decay.append(p)
            else:
                nodecay.append(p)

    emb_param = getattr(getattr(model, "embed", None), "weight", None)
    other_decay = [p for p in decay if p is not emb_param]
    other_nodecay = [p for p in nodecay if p is not emb_param]

    groups = []
    if other_decay:
        groups.append({"params": other_decay, "lr": lr, "weight_decay": weight_decay})
    if other_nodecay:
        groups.append({"params": other_nodecay, "lr": lr, "weight_decay": 0.0})
    if (emb_param is not None) and emb_param.requires_grad:
        groups.append({"params": [emb_param], "lr": lr * emb_lr_mult, "weight_decay": 0.0})

    try:
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.0, fused=True)
    except TypeError:
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.0)
    return opt

