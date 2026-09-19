"""
features.py  --  leakage-safe construction of the per-hour feature rows (NOT an executable step).

Core idea: for every ICU stay we build a dense table of hourly "state rows" g = 0..R-1, where row g is a
function ONLY of observations available at time 60*g minutes:

    physiology : hourly mean/min/max over the bin (60(g-1), 60g], observed flag, hours-since-last-observation
    labs       : latest value carried forward, freshness (hours since), 24 h slope, 24 h count, ever-measured flag
    text       : mean embedding of notes whose *documentation* time (noteenteredoffset) fell in (60(g-1), 60g],
                 note flag, hours since last note

A model sample for anchor hour a is simply rows a-23 .. a  (24 rows) plus static features.  06_build_features.py
verifies the causal contract with a truncation test: deleting all data after 60*g must leave row g unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (GRID_MIN, LAB_STATS, LOOKBACK_H, PHYS_CHANNELS, PHYS_STATS, SEED, TEXT_DIM)


# --------------------------------------------------------------------------- layout
class Layout:
    """Maps (stay, hour) -> global row index.  stay_ids must be sorted ascending."""

    def __init__(self, stay_ids, n_rows):
        self.stay_ids = np.asarray(stay_ids, dtype=np.int64)
        self.n_rows = np.asarray(n_rows, dtype=np.int64)
        assert np.all(np.diff(self.stay_ids) > 0), "stay_ids must be strictly increasing"
        self.row_start = np.r_[0, np.cumsum(self.n_rows)[:-1]].astype(np.int64)
        self.total = int(self.n_rows.sum())
        self.row_of_start = np.repeat(self.row_start, self.n_rows)       # start row of the stay owning each row
        self.row_stay = np.repeat(self.stay_ids, self.n_rows)
        self.row_hour = np.arange(self.total) - self.row_of_start

    def pos_of(self, stay_ids) -> np.ndarray:
        s = np.asarray(stay_ids, dtype=np.int64)
        pos = np.searchsorted(self.stay_ids, s)
        pos_c = np.clip(pos, 0, len(self.stay_ids) - 1)
        ok = self.stay_ids[pos_c] == s
        return np.where(ok, pos_c, -1)

    def rows(self, stay_ids, hours) -> np.ndarray:
        pos = self.pos_of(stay_ids)
        h = np.asarray(hours, dtype=np.int64)
        ok = (pos >= 0) & (h >= 0)
        ok &= h < np.where(pos >= 0, self.n_rows[np.clip(pos, 0, None)], 0)
        return np.where(ok, self.row_start[np.clip(pos, 0, None)] + h, -1)


def _time_since_last(flag: np.ndarray, layout: Layout, cap: float) -> np.ndarray:
    """flag (Rtot, C) bool -> hours since last True inside the same stay, /cap, clipped to 1 (1 = never)."""
    rows = np.arange(layout.total)[:, None]
    idx = np.where(flag, rows, -1)
    last = np.maximum.accumulate(idx, axis=0)
    has = last >= layout.row_of_start[:, None]
    return np.where(has, np.minimum(rows - last, cap) / cap, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- physiology
def hourly_phys_table(vit: pd.DataFrame) -> pd.DataFrame:
    """Long cleaned vitals -> hourly table [stay_id, hour, <ch>_mean/_min/_max/_count].
    Observation at offset t (minutes) belongs to bin hour=ceil(t/60) (availability time 60*hour >= t)."""
    v = vit[["stay_id", "t", *PHYS_CHANNELS]].copy()
    v["hour"] = np.maximum(np.ceil(v["t"].to_numpy() / GRID_MIN), 0).astype(np.int64)
    g = v.groupby(["stay_id", "hour"])[PHYS_CHANNELS].agg(["mean", "min", "max", "count"])
    g.columns = [f"{c}_{s}" for c, s in g.columns]
    return g.reset_index()


def phys_raw(layout: Layout, hourly: pd.DataFrame) -> np.ndarray:
    """Dense (Rtot, C, 3) array of [mean, min, max]; NaN = unobserved."""
    vals = np.full((layout.total, len(PHYS_CHANNELS), 3), np.nan, dtype=np.float32)
    rows = layout.rows(hourly["stay_id"].to_numpy(), hourly["hour"].to_numpy())
    for c, ch in enumerate(PHYS_CHANNELS):
        ok = (rows >= 0) & (hourly[f"{ch}_count"].to_numpy() > 0)
        for j, s in enumerate(("mean", "min", "max")):
            vals[rows[ok], c, j] = hourly[f"{ch}_{s}"].to_numpy()[ok]
    return vals


def fit_phys_scaler(vals: np.ndarray, layout: Layout, train_stays) -> dict:
    m = np.isin(layout.row_stay, np.asarray(list(train_stays)))
    mu, sd = [], []
    for c in range(len(PHYS_CHANNELS)):
        x = vals[m, c, 0]
        x = x[~np.isnan(x)]
        mu.append(float(x.mean()) if len(x) else 0.0)
        sd.append(float(max(x.std(), 1e-3)) if len(x) else 1.0)
    return {"mu": mu, "sd": sd}


def phys_features(layout: Layout, vals: np.ndarray, scaler: dict,
                  noise_sigma: float = 0.0, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """-> (features (Rtot, C*5), obs_any (Rtot,) bool).  Layout per channel: mean,min,max (z-scored, 0 if missing),
    obs flag, hours-since-last-observation/24."""
    mu = np.asarray(scaler["mu"], dtype=np.float32)[None, :, None]
    sd = np.asarray(scaler["sd"], dtype=np.float32)[None, :, None]
    obs = ~np.isnan(vals[:, :, 0])
    z = (vals - mu) / sd
    z = np.where(np.isnan(z), 0.0, z)
    if noise_sigma > 0:
        rng = rng or np.random.default_rng(SEED)
        z = z + (rng.normal(0.0, noise_sigma, size=z.shape).astype(np.float32) * obs[:, :, None])
    tsl = _time_since_last(obs, layout, cap=24.0)
    feat = np.concatenate([z, obs[:, :, None].astype(np.float32), tsl[:, :, None]], axis=2)
    return feat.reshape(layout.total, -1).astype(np.float32), obs.any(axis=1)


def phys_feature_names() -> list[str]:
    return [f"{ch}:{s}" for ch in PHYS_CHANNELS for s in PHYS_STATS]


# --------------------------------------------------------------------------- laboratory
def select_lab_names(lab: pd.DataFrame, train_stays, n: int) -> list[str]:
    """Top-n lab names by number of TRAIN stays with >=1 numeric result (label-free selection)."""
    d = lab[lab["stay_id"].isin(set(train_stays))]
    cnt = d.groupby("name")["stay_id"].nunique().sort_values(ascending=False)
    return list(cnt.index[:n])


def fit_lab_scaler(lab: pd.DataFrame, names: list[str], train_stays) -> dict:
    d = lab[lab["stay_id"].isin(set(train_stays)) & lab["name"].isin(names)]
    out = {"names": names, "lo": [], "hi": [], "mu": [], "sd": []}
    for n in names:
        v = d.loc[d["name"] == n, "value"].to_numpy(dtype=float)
        lo, hi = np.percentile(v, [1, 99])
        c = np.clip(v, lo, hi)
        out["lo"].append(float(lo)); out["hi"].append(float(hi))
        out["mu"].append(float(c.mean())); out["sd"].append(float(max(c.std(), 1e-6)))
    return out


def lab_features(layout: Layout, lab: pd.DataFrame, scaler: dict,
                 drop_frac: float = 0.0, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """-> (features (Rtot, L*5), ever_any (Rtot,) bool).  Row g only sees results with t <= 60*g."""
    names = scaler["names"]
    L = len(names)
    k_of = {n: i for i, n in enumerate(names)}
    df = lab[lab["name"].isin(k_of)]
    pos = layout.pos_of(df["stay_id"].to_numpy())
    df = df[pos >= 0]
    pos = pos[pos >= 0]
    if drop_frac > 0 and len(df):
        rng = rng or np.random.default_rng(SEED)
        keep = rng.random(len(df)) >= drop_frac
        df, pos = df[keep], pos[keep]
    feats = np.zeros((layout.total, L, 5), dtype=np.float32)
    feats[:, :, 1] = 1.0                                         # default: never measured -> stale
    if len(df) == 0:
        return feats.reshape(layout.total, -1), np.zeros(layout.total, dtype=bool)
    k = df["name"].map(k_of).to_numpy()
    t = df["t"].to_numpy(dtype=float)
    lo, hi = np.asarray(scaler["lo"])[k], np.asarray(scaler["hi"])[k]
    mu, sd = np.asarray(scaler["mu"])[k], np.asarray(scaler["sd"])[k]
    z = (np.clip(df["value"].to_numpy(dtype=float), lo, hi) - mu) / sd
    key = pos.astype(np.int64) * L + k
    order = np.lexsort((t, key))
    key, t, z = key[order], t[order], z[order]
    ukey, start = np.unique(key, return_index=True)
    end = np.r_[start[1:], len(key)]
    for uk, s, e in zip(ukey, start, end):
        sp, kk = int(uk // L), int(uk % L)
        R, rs = int(layout.n_rows[sp]), int(layout.row_start[sp])
        tt, zz = t[s:e], z[s:e]
        G = GRID_MIN * np.arange(R, dtype=float)
        hi_i = np.searchsorted(tt, G, side="right")              # results with t <= G
        lo_i = np.searchsorted(tt, G - 24 * GRID_MIN, side="right")
        n = hi_i - lo_i
        has = hi_i > 0
        last = np.maximum(hi_i - 1, 0)
        latest = np.where(has, zz[last], 0.0)
        tsl = np.where(has, np.minimum((G - tt[last]) / 60.0, 72.0) / 72.0, 1.0)
        th = (tt - tt[0]) / 60.0                                 # centred hours (numerical stability)
        c1 = np.r_[0.0, np.cumsum(th)]
        c2 = np.r_[0.0, np.cumsum(zz)]
        c3 = np.r_[0.0, np.cumsum(th * zz)]
        c4 = np.r_[0.0, np.cumsum(th * th)]
        st, sz, stz, stt = (c1[hi_i] - c1[lo_i], c2[hi_i] - c2[lo_i], c3[hi_i] - c3[lo_i], c4[hi_i] - c4[lo_i])
        den = n * stt - st ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where((n >= 2) & (den > 1e-6), (n * stz - st * sz) / den, 0.0) * 24.0
        block = np.stack([latest, tsl, np.clip(slope, -5, 5), np.log1p(n) / 2.0, has.astype(float)], axis=1)
        feats[rs:rs + R, kk, :] = block
    return feats.reshape(layout.total, -1).astype(np.float32), feats[:, :, 4].max(axis=1) > 0


def lab_feature_names(names: list[str]) -> list[str]:
    return [f"{n}:{s}" for n in names for s in LAB_STATS]


# --------------------------------------------------------------------------- text
class TfidfSvdEncoder:
    """Frozen, offline, typo/shorthand-robust note encoder: char n-gram TF-IDF -> truncated SVD -> per-dim z-score.
    Fitted on TRAIN notes only (no vocabulary or scaling information leaks from validation/test)."""
    kind = "tfidf_char_svd"

    def __init__(self, dim: int = TEXT_DIM, seed: int = SEED):
        self.dim, self.seed = dim, seed

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000,
                                   sublinear_tf=True, lowercase=True, dtype=np.float32)
        X = self.vec.fit_transform(texts)
        k = max(1, min(self.dim, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=self.seed)
        Z = self._pad(self.svd.fit_transform(X))
        self.mu, self.sd = Z.mean(0), np.maximum(Z.std(0), 1e-6)
        return self

    def _pad(self, Z):
        if Z.shape[1] < self.dim:
            Z = np.pad(Z, ((0, 0), (0, self.dim - Z.shape[1])))
        return Z

    def transform(self, texts):
        Z = self._pad(self.svd.transform(self.vec.transform(texts)))
        return ((Z - self.mu) / self.sd).astype(np.float32)


class HFEncoder:
    """Frozen HuggingFace encoder (e.g. emilyalsentzer/Bio_ClinicalBERT) + PCA to `dim`; mean pooling, no fine-tuning."""
    kind = "hf"

    def __init__(self, model_name: str, dim: int = TEXT_DIM, seed: int = SEED, max_len: int = 256, bs: int = 32):
        self.model_name, self.dim, self.seed, self.max_len, self.bs = model_name, dim, seed, max_len, bs

    def _embed(self, texts):
        import torch
        from transformers import AutoModel, AutoTokenizer
        if not hasattr(self, "tok"):
            self.tok = AutoTokenizer.from_pretrained(self.model_name)
            self.mdl = AutoModel.from_pretrained(self.model_name).eval()
            self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.mdl.to(self.dev)
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.bs):
                enc = self.tok(list(texts[i:i + self.bs]), padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt").to(self.dev)
                h = self.mdl(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
        return np.concatenate(out, 0)

    def fit(self, texts):
        from sklearn.decomposition import PCA
        E = self._embed(texts)
        self.pca = PCA(n_components=min(self.dim, E.shape[1], E.shape[0]), random_state=self.seed).fit(E)
        Z = self._pad(self.pca.transform(E))
        self.mu, self.sd = Z.mean(0), np.maximum(Z.std(0), 1e-6)
        return self

    _pad = TfidfSvdEncoder._pad

    def transform(self, texts):
        Z = self._pad(self.pca.transform(self._embed(texts)))
        return ((Z - self.mu) / self.sd).astype(np.float32)


def text_features(layout: Layout, notes: pd.DataFrame, emb: np.ndarray, delay_min: float = 0.0,
                  drop_frac: float = 0.0, rng=None) -> tuple[np.ndarray, np.ndarray]:
    """notes: [stay_id, avail_min] aligned row-by-row with emb (N, d).
    A note becomes visible at row ceil((avail_min + delay)/60): i.e. at its DOCUMENTATION time (+ optional
    artificial delay for the stress test), never at its clinical event time.
    -> (features (Rtot, d+2) = [mean embedding of new notes, hours-since-last-note/72, log1p(count)/2], has_new (Rtot,))"""
    d = emb.shape[1]
    sums = np.zeros((layout.total, d), dtype=np.float64)
    cnt = np.zeros(layout.total, dtype=np.float64)
    if len(notes):
        g = np.maximum(np.ceil((notes["avail_min"].to_numpy(dtype=float) + delay_min) / GRID_MIN), 0).astype(np.int64)
        rows = layout.rows(notes["stay_id"].to_numpy(), g)
        ok = rows >= 0
        if drop_frac > 0:
            rng = rng or np.random.default_rng(SEED)
            ok &= rng.random(len(rows)) >= drop_frac
        np.add.at(sums, rows[ok], emb[ok].astype(np.float64))
        np.add.at(cnt, rows[ok], 1.0)
    has = cnt > 0
    mean = sums / np.maximum(cnt, 1.0)[:, None]
    tsl = _time_since_last(has[:, None], layout, cap=72.0)[:, 0]
    feat = np.concatenate([mean, tsl[:, None], (np.log1p(cnt) / 2.0)[:, None]], axis=1)
    return feat.astype(np.float32), has


# --------------------------------------------------------------------------- static
def fit_static_scaler(cohort: pd.DataFrame, train_stays) -> dict:
    tr = cohort[cohort["stay_id"].isin(set(train_stays))]
    sc = {"num": {}, "unit_types": []}
    for col in ["age_num", "admissionweight", "admissionheight", "pre_icu_h"]:
        v = pd.to_numeric(tr[col], errors="coerce").dropna()
        sc["num"][col] = {"median": float(v.median()) if len(v) else 0.0,
                          "mu": float(v.mean()) if len(v) else 0.0,
                          "sd": float(max(v.std(), 1e-6)) if len(v) > 1 else 1.0}
    sc["unit_types"] = list(tr["unittype"].astype(str).value_counts().index[:6])
    return sc


def static_features(cohort: pd.DataFrame, scaler: dict) -> tuple[np.ndarray, list[str]]:
    """Static admission features (NO outcome / discharge information)."""
    cols, names = [], []
    for col in ["age_num", "admissionweight", "admissionheight", "pre_icu_h"]:
        v = pd.to_numeric(cohort[col], errors="coerce")
        p = scaler["num"][col]
        miss = v.isna()
        z = ((v.fillna(p["median"]) - p["mu"]) / p["sd"]).clip(-5, 5)
        cols.append(z.to_numpy()); names.append(col)
        if col in ("admissionweight", "admissionheight"):
            cols.append(miss.to_numpy().astype(float)); names.append(col + "_missing")
    g = cohort["gender"].astype(str).str.lower()
    cols.append(np.where(g == "male", 1.0, np.where(g == "female", 0.0, 0.5))); names.append("male")
    ut = cohort["unittype"].astype(str)
    for u in scaler["unit_types"]:
        cols.append((ut == u).to_numpy().astype(float)); names.append(f"unit={u}")
    cols.append((~ut.isin(scaler["unit_types"])).to_numpy().astype(float)); names.append("unit=other")
    return np.stack(cols, axis=1).astype(np.float32), names