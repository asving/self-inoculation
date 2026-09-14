"""Pre-LN causal GPT with per-sublayer residual capture (adapted from shortcut_scan/model.py).

Input: [BOS, tok_1, ..., tok_L] with tok = 3*x + y in 0..8, BOS = 9 (vocab 10). Logits at
position t (0-based, t = 0..L-1) predict tok_{t+1}, having seen t data tokens.
Capture convention (list index):
  resid[0]      = embeddings (+ positions)
  resid[2l-1]   = after attention sub-block of layer l   (l = 1..n_layers)
  resid[2l]     = after MLP sub-block of layer l
"""
import math
import torch
import torch.nn as nn

VOCAB, BOS = 10, 9


class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.n_heads = n_heads

    def forward(self, x, return_attn=False):
        B, T, d = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q, k, v = (z.view(B, T, self.n_heads, -1).transpose(1, 2) for z in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float('-inf')).softmax(-1)
        a = (att @ v).transpose(1, 2).reshape(B, T, d)
        x_mid = x + self.proj(a)
        x_out = x_mid + self.mlp(self.ln2(x_mid))
        return x_out, x_mid, (att if return_attn else None)


class GPT(nn.Module):
    def __init__(self, d=120, n_layers=4, n_heads=4, n_ctx=65):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, d)
        self.wpe = nn.Embedding(n_ctx, d)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.unembed = nn.Linear(d, VOCAB, bias=False)
        self.n_ctx = n_ctx

    def forward(self, idx, return_resid=False, return_attn=False):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        resid, attns = [x], []
        for blk in self.blocks:
            x, x_mid, att = blk(x, return_attn)
            resid += [x_mid, x]
            attns.append(att)
        logits = self.unembed(self.ln_f(x))
        if return_resid or return_attn:
            return logits, resid, attns
        return logits
