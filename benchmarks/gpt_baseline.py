"""A small GPT (pre-LN causal Transformer) used only as a reference point in the benchmark."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, D, heads, ff_mult=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.qkv = nn.Linear(D, 3 * D)
        self.proj = nn.Linear(D, D)
        self.ff = nn.Sequential(nn.Linear(D, ff_mult * D), nn.GELU(), nn.Linear(ff_mult * D, D))
        self.heads = heads

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size, D=64, layers=4, heads=4, max_len=1024):
        super().__init__()
        self.V = vocab_size
        self.embed = nn.Embedding(vocab_size, D)
        self.pos = nn.Embedding(max_len, D)
        self.blocks = nn.ModuleList(Block(D, heads) for _ in range(layers))
        self.norm_f = nn.LayerNorm(D)
        self.out = nn.Linear(D, vocab_size, bias=False)
        self.out.weight = self.embed.weight
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
        for blk in self.blocks:
            nn.init.normal_(blk.proj.weight, std=0.02 / math.sqrt(2 * layers))

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.embed(idx) + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.out(self.norm_f(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, self.V).float(), targets.reshape(-1))
        return logits, loss
