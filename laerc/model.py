import math

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

    def get_lr(self) -> list[float]:
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


@torch.no_grad()
def spectral_radius(W: torch.Tensor) -> float:
    """Largest |eigenvalue| (computed once at init, in float64 on CPU)."""
    return torch.linalg.eigvals(W.detach().double().cpu()).abs().max().item()


def diag_scan(u: torch.Tensor, log_lam: torch.Tensor, h0: torch.Tensor | None = None, chunk: int = 32):
    """h_t = lam * h_{t-1} + u_t for a complex diagonal lam = exp(log_lam), all t at once.

    u: (B, T, N) complex, log_lam: (N,) complex with Re < 0, h0: (B, N) complex or None.
    Within chunks of C steps the recurrence is one (C x C) Toeplitz matmul per mode; the chunks are then
    chained by a short loop over T / C carries. Only non-negative powers of lam appear, so it is stable,
    and the work is O(B T N C) in parallel instead of T sequential steps. Returns (h (B,T,N), h_T (B,N)).
    """
    B, T, N = u.shape
    C = max(1, min(chunk, T))
    K = -(-T // C)
    if K * C != T:
        u = F.pad(u, (0, 0, 0, K * C - T))
    i = torch.arange(C, device=u.device)
    diff = (i[:, None] - i[None, :]).to(log_lam.real.dtype)  # (C, C)
    L = torch.exp(diff[..., None] * log_lam) * (diff >= 0)[..., None]  # (C, C, N): lam^(i-j), lower triangular
    uc = u.reshape(B, K, C, N)
    y = torch.einsum("ijn,bkjn->bkin", L, uc)
    decay = torch.exp((i + 1).to(log_lam.real.dtype)[:, None] * log_lam)  # (C, N): lam^(i+1)
    carry = u.new_zeros(B, N) if h0 is None else h0
    outs = []
    for k in range(K):
        yk = y[:, k] + decay * carry[:, None]
        outs.append(yk)
        carry = yk[:, -1]
    h = torch.stack(outs, dim=1).reshape(B, K * C, N)[:, :T]
    return h, h[:, -1]


class DiagReservoir(nn.Module):
    """Frozen linear reservoir in modal (complex diagonal) form.

        h_t = lam * h_{t-1} + gamma * B x_t,   features = [Re h_t, Im h_t]   (N = R/2 complex modes)

    A linear reservoir h_t = W h_{t-1} + U x_t is, after diagonalising W, exactly this with lam = eig(W),
    so it is sampled directly: |lam_i| = exp(-1/tau_i) with tau_i log-spaced (rates a_i as for the leaky
    reservoir: tau_i = tau_rho / a_i, tau_rho = 1 / ln(1/rho)), phases uniform in [0, pi] (the conjugate
    halves are implied by taking Re and Im), and gamma = sqrt(1 - |lam|^2) so every mode has unit variance.
    No R x R matmul per step and no sequential loop over tokens: see diag_scan.
    """

    def __init__(self, D: int, R: int, rho: float, rates: torch.Tensor, device=None):
        super().__init__()
        if not 0.0 < rho < 1.0:
            raise ValueError("res_mode='diag' needs 0 < res_radius < 1")
        self.N = R // 2
        rates = rates[torch.linspace(0, len(rates) - 1, self.N).round().long()].double()
        tau = (1.0 / math.log(1.0 / rho)) / rates
        log_mag = -1.0 / tau
        theta = torch.rand(self.N, dtype=torch.float64) * math.pi
        gamma = torch.sqrt(-torch.expm1(2.0 * log_mag))
        Bm = torch.randn(D, 2 * self.N, dtype=torch.float64) / math.sqrt(2.0 * D)
        Bm = Bm * torch.cat([gamma, gamma])[None, :]
        self.register_buffer("log_mag", log_mag.float().to(device))
        self.register_buffer("theta", theta.float().to(device))
        self.register_buffer("B_in", Bm.float().to(device))  # (D, 2N): [real part | imaginary part]

    @property
    def timescales(self) -> tuple[float, float]:
        tau = -1.0 / self.log_mag
        return tau.min().item(), tau.max().item()

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        """x: (B, T, D) real; h0: (1, B, N) complex or None. Returns features (B, T, 2N) and h_T (1, B, N)."""
        with torch.autocast(device_type=x.device.type, enabled=False):
            p = x.float() @ self.B_in
            u = torch.complex(p[..., : self.N], p[..., self.N:])
            log_lam = torch.complex(self.log_mag, self.theta)
            h, h_T = diag_scan(u, log_lam, None if h0 is None else h0[0])
            return torch.cat([h.real, h.imag], dim=-1), h_T.unsqueeze(0)


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
        res_scale: str = "radius",
        res_leak_min: float = 1.0,
        res_leak_scale: float = 1.0,
        res_mode: str = "tanh",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        if res_mode not in ("tanh", "diag"):
            raise ValueError("res_mode must be 'tanh' or 'diag'")
        if not 0.0 < res_leak_min <= 1.0 or not 0.0 < res_leak_scale <= 1.0:
            raise ValueError("res_leak_min and res_leak_scale must lie in (0, 1]")
        self.D = D
        self.leaky = use_reservoir and res_mode == "tanh" and (res_leak_min < 1.0 or res_leak_scale < 1.0)
        self.R = int(D * reservoir_mult)
        self.diag = None
        self.H_res = int(D * reservoir_mlp_mult)
        self.H_ff = int(D * ff_mult)
        self.use_reservoir = use_reservoir

        self.ln_in = nn.LayerNorm(D)
        param_dtype = dtype if dtype is not None else torch.float32

        # reservoir
        if use_reservoir and self.R > 1 and res_mode == "diag":
            rates = torch.logspace(math.log10(res_leak_min), 0.0, self.R) * res_leak_scale
            self.diag = DiagReservoir(D, self.R, reservoir_radius, rates, device=device)
            self.R = 2 * self.diag.N
            self.rnn = None
        elif use_reservoir and self.R > 0:
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
                # "radius": scale the spectral radius (largest |eigenvalue|), which sets the memory
                # timescale ~ 1/ln(1/rho). "norm" reproduces v0.1, which scaled the spectral norm instead
                # (for a random Gaussian matrix the radius is then only about half the requested value).
                if res_scale == "radius":
                    sigma = spectral_radius(W)
                elif res_scale == "norm":
                    sigma = spectral_norm_power_dense(W, iters=30)
                else:
                    raise ValueError("res_scale must be 'radius' or 'norm'")
                if sigma > 0:
                    W.mul_(reservoir_radius / sigma)
                rnn.weight_ih_l0.copy_(U.t().contiguous())
                rnn.weight_hh_l0.copy_(W.t().contiguous())
                rnn.bias_ih_l0.zero_()
                rnn.bias_hh_l0.zero_()

            for p in rnn.parameters():
                p.requires_grad = False

            self.rnn = rnn
        if use_reservoir and self.R > 0:
            if self.leaky:
                # multi-timescale (leaky) reservoir: unit i integrates with its own leak rate a_i,
                #   h_t = (1 - a) * h_{t-1} + a * tanh(U x_t + W h_{t-1}),
                # a log-spaced in [res_leak_min, 1] -> memory timescales from ~1 to ~1/res_leak_min tokens.
                # res_leak_scale multiplies all rates, i.e. shifts the whole band to timescales 1/res_leak_scale
                # times longer (set per layer by the model's res_depth_growth).
                leak = torch.logspace(math.log10(res_leak_min), 0.0, self.R, device=device, dtype=param_dtype)
                leak = leak * res_leak_scale
                self.register_buffer("leak", leak)
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

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        """
        x: (B, T, D); h0: optional reservoir state (1, B, R) to continue from.
        Returns (y, h_T) where h_T is the final reservoir state (None without a reservoir).
        """
        h = self.ln_in(x)
        h_T = None

        if self.res_mlp is not None:
            # cuDNN RNN sometimes chokes on non-contiguous inputs, especially under torch.compile
            h_contig = h.contiguous()
            if h0 is not None and not h0.is_complex():
                h0 = h0.to(h_contig.dtype)
            if self.diag is not None:
                r, h_T = self.diag(h_contig, h0)
            elif self.leaky:
                r, h_T = self._leaky_scan(h_contig, h0)
            else:
                r, h_T = self.rnn(h_contig, h0)
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
        return y, h_T


    def _leaky_scan(self, x: torch.Tensor, h0: torch.Tensor | None):
        """Leaky tanh reservoir over (B, T, D) with the frozen U, W of self.rnn. Returns (B,T,R), (1,B,R)."""
        rnn = self.rnn
        pre = F.linear(x, rnn.weight_ih_l0, rnn.bias_ih_l0 + rnn.bias_hh_l0)  # (B, T, R), one matmul
        B, T, _ = pre.shape
        h = pre.new_zeros(B, self.R) if h0 is None else h0[0]
        a, W = self.leak, rnn.weight_hh_l0
        outs = []
        for t in range(T):
            h = h + a * (torch.tanh(pre[:, t] + h @ W.t()) - h)
            outs.append(h)
        return torch.stack(outs, dim=1), h.unsqueeze(0)


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
        res_scale: str = "radius",
        res_leak_min: float = 1.0,
        res_depth_growth: float = 1.0,
        res_mode: str = "tanh",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        res_mode: "tanh" (nonlinear echo-state reservoir, the original) or "diag" (frozen linear reservoir in
        complex-diagonal form, computed in parallel over the sequence; see DiagReservoir).

        res_depth_growth = g > 1 gives the layers geometrically growing timescales: block k's leak rates are
        divided by g**k, so its memory band sits g**k times further back. Stacking L equal blocks only adds
        horizons (~L * tau); a geometric schedule reaches ~g**(L-1) * tau, i.e. depth ~ log(context).
        """
        super().__init__()
        if res_depth_growth < 1.0:
            raise ValueError("res_depth_growth must be >= 1")
        self.V = vocab_size
        self.D = D

        self.embed = nn.Embedding(vocab_size, D)
        blocks = []
        for k in range(num_layers):
            blk = ReservoirBlock(
                D=D,
                reservoir_mult=reservoir_mult,
                reservoir_mlp_mult=reservoir_mlp_mult,
                ff_mult=ff_mult,
                reservoir_radius=reservoir_radius,
                use_reservoir=use_reservoir,
                res_scale=res_scale,
                res_leak_min=res_leak_min,
                res_leak_scale=float(res_depth_growth) ** -k,
                res_mode=res_mode,
                device=device,
                dtype=dtype,
            )
            blocks.append(blk)
        self.blocks = nn.ModuleList(blocks)
        self.norm_f = nn.LayerNorm(D)
        self.out = nn.Linear(D, vocab_size, bias=False)

        self._init_weights(tie_weights=True)

    def reservoir_timescales(self) -> list[tuple[float, float] | None]:
        """Per block (tau_min, tau_max) in tokens: 1/a for the leaky tanh reservoir, the exact mode decay times
        for res_mode="diag", (1, 1) for a plain tanh reservoir, None without one.

        This is the leak timescale alone; the recurrence W (spectral radius rho) stretches each mode by up to
        about 1/(1 - rho) on top of it.
        """
        out = []
        for blk in self.blocks:
            if blk.res_mlp is None:
                out.append(None)
            elif blk.diag is not None:
                out.append(blk.diag.timescales)
            elif blk.leaky:
                out.append((1.0 / blk.leak.max().item(), 1.0 / blk.leak.min().item()))
            else:
                out.append((1.0, 1.0))
        return out

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
        targets: torch.Tensor | None = None,
        state: list[torch.Tensor | None] | None = None,
        return_state: bool = False,
    ):
        """
        idx: (B, T) token ids; targets: (B, T) or None.
        state: per-block reservoir states from a previous call (continue the sequence from there).
        Everything except the reservoir acts per token, so feeding a sequence in pieces while
        carrying `state` gives exactly the same logits as feeding it at once (used for generation).
        Returns (logits, loss) or, with return_state=True, (logits, loss, new_state).
        """
        x = self.embed(idx)
        new_state = []
        for i, blk in enumerate(self.blocks):
            x, h = blk(x, None if state is None else state[i])
            new_state.append(h)
        x = self.norm_f(x)
        logits = self.out(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.V).float(),
                targets.reshape(-1),
            )
        if return_state:
            return logits, loss, new_state
        return logits, loss


def build_model(cfg, device=None, dtype=None) -> "ReservoirFFNLanguageModel":
    """Construct the model described by a TrainConfig (or a dict with the same keys)."""
    get = cfg.get if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    return ReservoirFFNLanguageModel(
        vocab_size=get("vocab"), D=get("emb"), num_layers=get("layers"),
        reservoir_mult=get("resv_mult", 4.0), reservoir_mlp_mult=get("res_mlp_mult", 1.0),
        ff_mult=get("ffn_mult", 4.0), reservoir_radius=get("res_radius", 0.9),
        use_reservoir=get("use_reservoir", True), res_scale=get("res_scale", "norm"),
        res_leak_min=get("res_leak_min", 1.0) or 1.0, res_depth_growth=get("res_depth_growth", 1.0) or 1.0,
        res_mode=get("res_mode", "tanh") or "tanh",
        device=device, dtype=dtype,
    )


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    emb_lr_mult: float = 0.2,
    beta2: float = 0.999,
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
        opt = torch.optim.AdamW(groups, betas=(0.9, beta2), weight_decay=0.0, fused=True)
    except TypeError:
        opt = torch.optim.AdamW(groups, betas=(0.9, beta2), weight_decay=0.0)
    return opt

