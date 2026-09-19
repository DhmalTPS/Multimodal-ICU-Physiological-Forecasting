"""
nets.py  --  dataset access, neural architectures, training loop, checkpoints (NOT an executable step).

Models
------
* MultimodalRiskNet : the research model
      physiology TCN | lab MLP | frozen-text projection
      -> causal cross-modal attention (phys<-text, phys<-lab, lab<-text; a learned null token keeps attention
         defined when a modality is absent)
      -> missingness-aware reliability gate  Z_t = sum_k m_k alpha_k Z_k / (renormalised over ACTIVE modalities)
      -> GRU longitudinal patient state (initialised from static features)
      -> 3 sigmoid heads (1h / 3h / 6h)
* ConcatGRU         : the naive baseline  [phys || lab || text || flags] -> GRU -> heads
Both share `off=` (disable modalities at inference) and training-time modality dropout (multimodal only).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (HORIZONS_H, LAB_MISSING, LOOKBACK_H, MODALITIES, PHYS_MISSING, SEED,
                    TEXT_MISSING_TAIL, safe_auprc)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_torch_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================================== data
class WindowData:
    """Concatenated per-stay hourly rows + anchor index.  A sample = 24 consecutive rows ending at the anchor row."""

    KEYS = ["stay_ids", "row_start", "n_rows", "phys", "lab", "text", "mod", "static",
            "a_stay", "a_hour", "a_min", "y", "next_onset"]

    def __init__(self, path=None, arrays=None):
        if arrays is None:
            z = np.load(path, allow_pickle=False)
            arrays = {k: z[k] for k in self.KEYS}
        self.a = arrays
        for k in self.KEYS:
            setattr(self, k, arrays[k])
        self.offsets = np.arange(-(LOOKBACK_H - 1), 1)[None, :]

    def __len__(self):
        return len(self.a_hour)

    def replace(self, **over) -> "WindowData":
        arr = dict(self.a)
        arr.update(over)
        return WindowData(arrays=arr)

    @property
    def dims(self) -> dict:
        return {"d_p": self.phys.shape[1], "d_l": self.lab.shape[1], "d_t": self.text.shape[1],
                "d_s": self.static.shape[1] + 1}

    def window_np(self, idx):
        s = self.a_stay[idx]
        a = self.a_hour[idx]
        rows = self.row_start[s][:, None] + a[:, None] + self.offsets           # (B, 24)
        stat = np.concatenate([self.static[s], (np.log1p(a) / 6.0)[:, None].astype(np.float32)], axis=1)
        return (self.phys[rows], self.lab[rows], self.text[rows],
                self.mod[rows].astype(np.float32), stat.astype(np.float32))

    def batch(self, idx, device=None, with_y=True):
        device = device or DEVICE
        p, l, t, m, s = self.window_np(idx)
        b = {"phys": torch.from_numpy(p), "lab": torch.from_numpy(l), "text": torch.from_numpy(t),
             "mod": torch.from_numpy(m), "static": torch.from_numpy(s)}
        if with_y:
            b["y"] = torch.from_numpy(self.y[idx].astype(np.float32))
        return {k: v.to(device) for k, v in b.items()}

    def frame(self):
        import pandas as pd
        d = {"stay_id": self.stay_ids[self.a_stay], "anchor_min": self.a_min}
        for j, h in enumerate(HORIZONS_H):
            d[f"y{h}"] = self.y[:, j]
        d["next_onset_min"] = self.next_onset
        return pd.DataFrame(d)


def flat_features(data: WindowData, bs: int = 4096):
    """Flat-table summary of a window (the 'throw away temporal shape' baseline the problem statement criticises)."""
    d = data.dims
    C, L = d["d_p"] // 5, d["d_l"] // 5
    out = []
    for i in range(0, len(data), bs):
        idx = np.arange(i, min(i + bs, len(data)))
        p, l, t, m, s = data.window_np(idx)
        B, T = p.shape[0], p.shape[1]
        ph = p.reshape(B, T, C, 5)
        mean, mn, mx, obs = ph[..., 0], ph[..., 1], ph[..., 2], ph[..., 3]
        n_obs = obs.sum(1)
        wmean = (mean * obs).sum(1) / np.maximum(n_obs, 1)
        wmin = np.where(obs > 0, mn, np.inf).min(1)
        wmax = np.where(obs > 0, mx, -np.inf).max(1)
        wmin, wmax = np.where(np.isinf(wmin), 0, wmin), np.where(np.isinf(wmax), 0, wmax)
        h1 = (mean[:, :6] * obs[:, :6]).sum(1) / np.maximum(obs[:, :6].sum(1), 1)
        h2 = (mean[:, -6:] * obs[:, -6:]).sum(1) / np.maximum(obs[:, -6:].sum(1), 1)
        tm = m[:, :, 2:3]
        tmean = (t[:, :, :-2] * tm).sum(1) / np.maximum(tm.sum(1), 1)
        out.append(np.concatenate([s, wmean, wmin, wmax, mean[:, -1], h2 - h1, obs.mean(1),
                                   l[:, -1, :], tmean, tm.mean(1)], axis=1).astype(np.float32))
    return np.concatenate(out, 0)


# =========================================================================== building blocks
class _TCNBlock(nn.Module):
    def __init__(self, d, k, dil, drop):
        super().__init__()
        self.pad = (k - 1) * dil
        self.conv = nn.Conv1d(d, d, k, dilation=dil)
        self.norm = nn.GroupNorm(1, d)
        self.drop = nn.Dropout(drop)

    def forward(self, h):
        y = self.conv(F.pad(h, (self.pad, 0)))               # causal: left padding only
        return h + self.drop(F.gelu(self.norm(y)))


class TCN(nn.Module):
    def __init__(self, d_in, d, k=3, dils=(1, 2, 4), drop=0.1):
        super().__init__()
        self.inp = nn.Conv1d(d_in, d, 1)
        self.blocks = nn.ModuleList([_TCNBlock(d, k, dl, drop) for dl in dils])

    def forward(self, x):                                    # (B,T,d_in) -> (B,T,d)
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        return h.transpose(1, 2)


class CausalCrossAttn(nn.Module):
    """Row t of the query stream attends to source rows s <= t whose modality is present.
    A learned null token is always attendable, so the softmax is defined even when the modality is absent."""

    def __init__(self, d, heads=4, drop=0.1):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh, self.drop = heads, d // heads, drop
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.null = nn.Parameter(torch.zeros(1, 1, d))

    def forward(self, xq, xs, src_mask):
        B, T, D = xq.shape
        kv = torch.cat([self.null.expand(B, 1, D), xs], 1)
        q = self.q(xq).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(kv).view(B, T + 1, self.h, self.dh).transpose(1, 2)
        v = self.v(kv).view(B, T + 1, self.h, self.dh).transpose(1, 2)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=xq.device))
        allowed = causal[None] & (src_mask > 0.5)[:, None, :]
        allowed = torch.cat([torch.ones(B, T, 1, dtype=torch.bool, device=xq.device), allowed], 2)[:, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
                                           dropout_p=self.drop if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, T, D))


class _RiskBase(nn.Module):
    """Shared handling of modality availability: templates for 'missing' rows, `off=` and modality dropout."""

    def __init__(self, dims: dict, use, mod_drop: float):
        super().__init__()
        self.dims, self.use, self.mod_drop = dims, tuple(use), mod_drop
        self.register_buffer("t_phys", torch.tensor(np.tile(PHYS_MISSING, dims["d_p"] // 5), dtype=torch.float32))
        self.register_buffer("t_lab", torch.tensor(np.tile(LAB_MISSING, dims["d_l"] // 5), dtype=torch.float32))
        self.register_buffer("t_text", torch.tensor(np.r_[np.zeros(dims["d_t"] - 2), TEXT_MISSING_TAIL],
                                                    dtype=torch.float32))

    def replace(self, batch, off_mask):
        """off_mask (B,3) bool: replace whole modality by its 'missing' representation and zero its flag."""
        out = dict(batch)
        for j, (key, tmpl) in enumerate((("phys", self.t_phys), ("lab", self.t_lab), ("text", self.t_text))):
            out[key] = torch.where(off_mask[:, j].view(-1, 1, 1), tmpl.view(1, 1, -1).expand_as(batch[key]), batch[key])
        out["mod"] = batch["mod"] * (~off_mask).float().unsqueeze(1)
        return out

    def _prep(self, batch, off=()):
        B = batch["phys"].shape[0]
        dev = batch["phys"].device
        avail = torch.tensor([(m in self.use) and (m not in off) for m in MODALITIES], device=dev)
        off_mask = (~avail).unsqueeze(0).expand(B, 3).clone()
        if self.training and self.mod_drop > 0:
            drop = torch.rand(B, 3, device=dev) < self.mod_drop
            drop &= avail.unsqueeze(0)
            all_gone = ~((~drop) & avail.unsqueeze(0)).any(1)
            drop[all_gone] = False                               # never drop every available modality
            off_mask = off_mask | drop
        return self.replace(batch, off_mask)


class MultimodalRiskNet(_RiskBase):
    arch = "multimodal"

    def __init__(self, dims, use=MODALITIES, D=64, H=96, heads=4, drop=0.2, mod_drop=0.15):
        super().__init__(dims, use, mod_drop)
        self.D, self.H = D, H
        self.enc_p = TCN(dims["d_p"], D)
        self.enc_l = nn.Sequential(nn.Linear(dims["d_l"], D), nn.GELU(), nn.Dropout(drop), nn.Linear(D, D), nn.LayerNorm(D))
        self.enc_t = nn.Sequential(nn.Linear(dims["d_t"], D), nn.GELU(), nn.LayerNorm(D))
        self.x_pt, self.x_pl, self.x_lt = (CausalCrossAttn(D, heads) for _ in range(3))
        self.norm_p, self.norm_l = nn.LayerNorm(D), nn.LayerNorm(D)
        self.gate = nn.ModuleList([nn.Linear(D, 1) for _ in range(3)])
        self.static = nn.Sequential(nn.Linear(dims["d_s"], H), nn.GELU())
        self.gru = nn.GRU(D + 3, H, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, len(HORIZONS_H)))

    def forward(self, batch, off=(), return_alpha=False):
        b = self._prep(batch, off)
        m = b["mod"]
        zp, zl, zt = self.enc_p(b["phys"]), self.enc_l(b["lab"]), self.enc_t(b["text"])
        zp2 = self.norm_p(zp + self.x_pt(zp, zt, m[..., 2]) + self.x_pl(zp, zl, m[..., 1]))   # phys <- text, lab
        zl2 = self.norm_l(zl + self.x_lt(zl, zt, m[..., 2]))                                  # lab  <- text
        Zs = torch.stack([zp2, zl2, zt], 2)                                                   # (B,T,3,D)
        sc = torch.cat([g(Zs[:, :, i]) for i, g in enumerate(self.gate)], -1)                 # reliability scores
        sc = sc.masked_fill(m < 0.5, -1e4)
        sc = sc - sc.max(-1, keepdim=True).values
        e = torch.exp(sc) * m
        alpha = e / (e.sum(-1, keepdim=True) + 1e-6)                                          # renormalised over active
        Z = (alpha.unsqueeze(-1) * Zs).sum(2)
        st = self.static(b["static"])
        out, _ = self.gru(torch.cat([Z, m], -1), st.unsqueeze(0).contiguous())
        logits = self.head(torch.cat([out[:, -1], st], -1))
        return (logits, alpha) if return_alpha else logits


class ConcatGRU(_RiskBase):
    arch = "concat"

    def __init__(self, dims, use=MODALITIES, H=96, drop=0.2, mod_drop=0.0):
        super().__init__(dims, use, mod_drop)
        self.H = H
        d_in = sum(dims[k] for m, k in zip(MODALITIES, ("d_p", "d_l", "d_t")) if m in self.use) + len(self.use)
        self.inp = nn.Sequential(nn.Linear(d_in, H), nn.GELU(), nn.LayerNorm(H))
        self.static = nn.Sequential(nn.Linear(dims["d_s"], H), nn.GELU())
        self.gru = nn.GRU(H, H, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.Dropout(drop), nn.Linear(H, len(HORIZONS_H)))

    def forward(self, batch, off=()):
        b = self._prep(batch, off)
        parts = [b[m] for m in MODALITIES if m in self.use]
        flags = b["mod"][..., [i for i, m in enumerate(MODALITIES) if m in self.use]]
        st = self.static(b["static"])
        out, _ = self.gru(self.inp(torch.cat(parts + [flags], -1)), st.unsqueeze(0).contiguous())
        return self.head(torch.cat([out[:, -1], st], -1))


def build_model(arch: str, dims: dict, use=MODALITIES, **kw):
    return {"multimodal": MultimodalRiskNet, "concat": ConcatGRU}[arch](dims, use=use, **kw)


# =========================================================================== training / inference
def pos_weights(y: np.ndarray) -> torch.Tensor:
    """Mild positive weighting only when a horizon is rare (<5 %): sqrt(neg/pos) capped at 10."""
    w = []
    for j in range(y.shape[1]):
        pos = max(float(y[:, j].sum()), 1.0)
        neg = float(len(y) - y[:, j].sum())
        prev = pos / len(y)
        w.append(1.0 if prev >= 0.05 else float(min(np.sqrt(neg / pos), 10.0)))
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict(model, data: WindowData, off=(), bs: int = 2048, transform=None):
    model.eval()
    logits = []
    for i in range(0, len(data), bs):
        b = data.batch(np.arange(i, min(i + bs, len(data))), with_y=False)
        if transform is not None:
            b = transform(b)
        out = model(b, off=off)
        logits.append(out.float().cpu().numpy())
    lg = np.concatenate(logits, 0) if logits else np.zeros((0, len(HORIZONS_H)))
    return 1.0 / (1.0 + np.exp(-lg)), lg


def mean_auprc(y, p) -> float:
    v = [safe_auprc(y[:, j], p[:, j]) for j in range(y.shape[1])]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def fit_model(model, train: WindowData, val: WindowData, name: str, epochs: int = 20, bs: int = 512,
              lr: float = 1e-3, wd: float = 1e-4, patience: int = 4, seed: int = SEED, verbose: bool = True):
    """Selection rule: best VALIDATION mean AUPRC (over horizons). The test set is never touched here."""
    set_torch_seed(seed)
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    pw = pos_weights(train.y).to(DEVICE)
    rng = np.random.default_rng(seed)
    best, best_state, best_ep, bad, log = -1.0, None, 0, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(train))
        tot, nb, t0 = 0.0, 0, time.time()
        for i in range(0, len(perm), bs):
            idx = np.sort(perm[i:i + bs])
            b = train.batch(idx)
            loss = F.binary_cross_entropy_with_logits(model(b), b["y"], pos_weight=pw)   # equal lambda_H = 1
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss); nb += 1
        p, _ = predict(model, val)
        score = mean_auprc(val.y, p)
        log.append({"model": name, "epoch": ep, "train_loss": tot / max(nb, 1), "val_mean_auprc": score,
                    "seconds": time.time() - t0})
        if verbose:
            print(f"  [{name}] epoch {ep:02d}  loss {tot / max(nb, 1):.4f}  val mean-AUPRC {score:.4f}")
        if score > best:
            best, best_ep, bad = score, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return {"best_val_mean_auprc": best, "best_epoch": best_ep, "log": log}


# =========================================================================== checkpoints
def save_checkpoint(path, model, meta: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "arch": model.arch, "dims": model.dims,
                "use": list(model.use), "meta": meta}, path)


def load_model(path, device=None):
    device = device or DEVICE
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["arch"], ck["dims"], use=tuple(ck["use"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck["meta"]


# =========================================================================== occlusion helpers (explain / stress)
def occlude_rows(model, batch, r0: int, r1: int):
    """Replace rows [r0, r1) of every modality by the 'missing' representation (time-window occlusion)."""
    out = dict(batch)
    for key, tmpl in (("phys", model.t_phys), ("lab", model.t_lab), ("text", model.t_text)):
        x = batch[key].clone()
        x[:, r0:r1, :] = tmpl.view(1, 1, -1)
        out[key] = x
    m = batch["mod"].clone()
    m[:, r0:r1, :] = 0.0
    out["mod"] = m
    return out


def occlude_columns(model, batch, key: str, block: int, width: int = 5):
    """Occlude one physiological channel / lab variable (all rows) by its 'missing' representation."""
    tmpl = {"phys": model.t_phys, "lab": model.t_lab}[key]
    out = dict(batch)
    x = batch[key].clone()
    sl = slice(block * width, (block + 1) * width)
    x[:, :, sl] = tmpl[sl].view(1, 1, -1)
    out[key] = x
    return out