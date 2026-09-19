"""
common.py  --  single source of truth for the pipeline (NOT an executable step).

Why this file exists: the numbered scripts (00_*.py ... 11_*.py) cannot be imported
(Python module names cannot start with a digit), and duplicating the leakage-critical logic
(time grid, event definition, labels, alarm policy) across scripts is how subtle
train/test inconsistencies are born.  Everything shared lives here, `features.py`
and `nets.py`.  There is deliberately NO config file: constants are below.

Time convention (eICU): every offset is in MINUTES relative to ICU-unit admission.
Prediction anchors are t = 60 * a minutes, a = 24, 25, ...
Feature "row" g summarises information available at time 60*g minutes (never later).
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Force UTF-8 stdout/stderr so Unicode characters used in log messages (τ, ─, etc.)
# never crash on Windows consoles stuck on legacy code pages (cp1252).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# --------------------------------------------------------------------------- paths
ROOT = Path(os.environ.get("ICU_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPORT = ROOT / "report"


def ensure_dirs() -> None:
    for d in [RAW, INTERIM, PROCESSED, MODELS / "baseline", MODELS / "multimodal",
              RESULTS / "audit", RESULTS / "cohort", RESULTS / "features",
              RESULTS / "predictions", RESULTS / "evaluation", RESULTS / "alarms",
              RESULTS / "explanations", FIGURES, REPORT]:
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- frozen design constants
SEED = 42
GRID_MIN = 60                     # prediction grid  (Delta t = 1 hour)
LOOKBACK_H = 24                   # history length L = 24 hours  -> 24 feature rows per sample
HORIZONS_H = (1, 3, 6)            # forecast horizons
MAX_STAY_H = 336                  # anchors are capped at 14 days so very long stays cannot dominate
MIN_VITAL_OBS_FIRST24 = 12        # cohort rule: >=12 vital observations in the first 24 h
N_LABS = 16                       # number of laboratory variables kept (chosen on TRAIN stays only)
TEXT_DIM = 64                     # dimensionality of frozen note embeddings
ECE_BINS = 10                     # number of equal-frequency bins for expected calibration error

PHYS_CHANNELS = ["hr", "rr", "spo2", "temp", "sbp", "dbp", "map"]
PLAUSIBLE = {                      # physiologically plausible ranges (values outside -> missing)
    "hr": (20, 300), "rr": (2, 80), "spo2": (50, 100), "temp": (25, 45),
    "sbp": (30, 300), "dbp": (10, 200), "map": (20, 250),
}
PHYS_STATS = ["mean", "min", "max", "obs", "tsl"]     # per-channel feature layout (5 values)
LAB_STATS = ["latest", "tsl", "slope", "count", "ever"]
PHYS_MISSING = [0.0, 0.0, 0.0, 0.0, 1.0]              # "unobserved" representation of one channel
LAB_MISSING = [0.0, 1.0, 0.0, 0.0, 0.0]               # "never measured" representation of one lab
TEXT_MISSING_TAIL = [1.0, 0.0]                        # (time-since-note, count) when no note exists
MODALITIES = ("phys", "lab", "text")

# event definitions (all thresholds frozen here; audited in 02_task_audit.py)
HYPOTENSION = dict(ch="map", thr=55.0, min_dur=30, max_gap=45)   # MAP<55 mmHg sustained >=30 min
HYPOXEMIA = dict(ch="spo2", thr=88.0, min_dur=30, max_gap=20)    # SpO2<88 % sustained >=30 min
MERGE_GAP_MIN = 60                # episodes closer than this are one clinical episode
CANDIDATES = ["composite_collapse", "icu_death", "hospital_death",
              "severe_hypotension", "severe_hypoxemia"]           # also the priority order

# task feasibility gates (02_task_audit.py)
GATE_MIN_POS_STAYS = 60
GATE_MIN_POS_ANCHORS_6H = 300
GATE_MIN_POS_ANCHORS_1H = 50
GATE_MIN_IN_ICU_FRACTION = 0.99
GATE_MIN_MEDIAN_LEAD_H = 6.0


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def banner(msg: str) -> None:
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78)


# --------------------------------------------------------------------------- raw table access
# def find_raw(name: str):
#     """Case-insensitive lookup of <name>.csv.gz or <name>.csv inside data/raw."""
#     have = {p.name.lower(): p for p in RAW.glob("*")}
#     for ext in (".csv.gz", ".csv"):
#         p = have.get((name + ext).lower())
#         if p is not None:
#             return p
#     return None

_RAW_INDEX = None

def _raw_index() -> dict:
    """Build (once) a case-insensitive filename -> Path index over everything under data/raw,
    searched recursively so a nested snapshot folder (e.g. data/raw/eicu_demo/) works fine."""
    global _RAW_INDEX
    if _RAW_INDEX is None:
        _RAW_INDEX = {p.name.lower(): p for p in RAW.rglob("*") if p.is_file()}
    return _RAW_INDEX


def find_raw(name: str):
    """Case-insensitive lookup of <name>.csv.gz or <name>.csv anywhere under data/raw."""
    idx = _raw_index()
    for ext in (".csv.gz", ".csv"):
        p = idx.get((name + ext).lower())
        if p is not None:
            return p
    return None


def read_table(name: str, usecols=None, nrows=None) -> pd.DataFrame:
    """Read a raw eICU table; column names are lower-cased (eICU CSV headers are lower-case
    in the official release but camelCase in some mirrors)."""
    p = find_raw(name)
    if p is None:
        raise FileNotFoundError(f"data/raw/{name}.csv[.gz] not found (looked in {RAW})")
    want = None if usecols is None else {c.lower() for c in usecols}
    df = pd.read_csv(p, usecols=None if want is None else (lambda c: c.lower() in want),
                     nrows=nrows, low_memory=False)
    df.columns = [c.lower() for c in df.columns]
    return df


def load_patient() -> pd.DataFrame:
    df = read_table("patient").rename(columns={"patientunitstayid": "stay_id"})
    df["stay_id"] = df["stay_id"].astype("int64")
    for c in ["uniquepid", "patienthealthsystemstayid", "gender", "age", "unittype",
              "unitdischargeoffset", "unitdischargestatus", "hospitaldischargeoffset",
              "hospitaldischargestatus", "hospitaladmitoffset", "admissionweight", "admissionheight"]:
        if c not in df.columns:
            df[c] = np.nan
    if df["uniquepid"].isna().all():
        df["uniquepid"] = df["patienthealthsystemstayid"].fillna(df["stay_id"])
    df["uniquepid"] = df["uniquepid"].astype(str)
    age = df["age"].astype(str)
    df["age_num"] = pd.to_numeric(age.str.replace(">", "", regex=False).str.strip(), errors="coerce")
    df.loc[age.str.contains(">", regex=False), "age_num"] = 90.0       # eICU top-codes ages > 89
    for c in ["unitdischargeoffset", "hospitaldischargeoffset", "hospitaladmitoffset",
              "admissionweight", "admissionheight"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# --------------------------------------------------------------------------- vitals
_VP = {"heartrate": "hr", "respiration": "rr", "sao2": "spo2", "temperature": "temp",
       "systemicsystolic": "sys_s", "systemicdiastolic": "sys_d", "systemicmean": "sys_m"}
_VA = {"noninvasivesystolic": "ni_s", "noninvasivediastolic": "ni_d", "noninvasivemean": "ni_m"}


def load_clean_vitals(stay_ids=None) -> pd.DataFrame:
    """vitalPeriodic (5-min, invasive) + vitalAperiodic (non-invasive BP) -> one long table
    [stay_id, t, hr, rr, spo2, temp, sbp, dbp, map] sorted by (stay_id, t).
    Implausible values become NaN; nothing is interpolated; timestamps are preserved."""
    vp = read_table("vitalperiodic", usecols=["patientunitstayid", "observationoffset", *_VP])
    va = read_table("vitalaperiodic", usecols=["patientunitstayid", "observationoffset", *_VA])
    frames = []
    for df, m in ((vp, _VP), (va, _VA)):
        df = df.rename(columns={"patientunitstayid": "stay_id", "observationoffset": "t", **m})
        if stay_ids is not None:
            df = df[df["stay_id"].isin(stay_ids)]
        frames.append(df)
    vit = pd.concat(frames, ignore_index=True)
    for c in vit.columns:
        if c not in ("stay_id", "t"):
            vit[c] = pd.to_numeric(vit[c], errors="coerce")
    vit["sbp"] = vit["sys_s"].fillna(vit["ni_s"])
    vit["dbp"] = vit["sys_d"].fillna(vit["ni_d"])
    vit["map"] = vit["sys_m"].fillna(vit["ni_m"])
    for ch, (lo, hi) in PLAUSIBLE.items():
        v = vit[ch]
        vit[ch] = v.where((v >= lo) & (v <= hi))
    vit = vit[["stay_id", "t", *PHYS_CHANNELS]].dropna(subset=["t"])
    vit = vit.dropna(subset=PHYS_CHANNELS, how="all")
    vit["stay_id"] = vit["stay_id"].astype("int64")
    vit["t"] = vit["t"].astype("float64")
    vit[PHYS_CHANNELS] = vit[PHYS_CHANNELS].astype("float32")
    return vit.sort_values(["stay_id", "t"], kind="mergesort").reset_index(drop=True)


class SeriesIndex:
    """O(1) per-stay access to the long vital table (must be sorted by stay_id)."""

    def __init__(self, vit: pd.DataFrame):
        s = vit["stay_id"].to_numpy()
        self.stays, start = np.unique(s, return_index=True)
        self.start = start
        self.end = np.r_[start[1:], len(s)]
        self.pos = {int(k): i for i, k in enumerate(self.stays)}
        self.t = vit["t"].to_numpy()
        self.cols = {c: vit[c].to_numpy() for c in PHYS_CHANNELS}

    def get(self, stay_id: int, ch: str):
        i = self.pos.get(int(stay_id))
        if i is None:
            return np.empty(0), np.empty(0)
        sl = slice(self.start[i], self.end[i])
        t, v = self.t[sl], self.cols[ch][sl]
        m = ~np.isnan(v)
        return t[m], v[m]


# --------------------------------------------------------------------------- events, anchors, labels
def sustained_runs(t: np.ndarray, low: np.ndarray, min_dur: float, max_gap: float):
    """Runs of consecutive below-threshold observations lasting >= min_dur minutes.
    Returns (start, onset, end): `start` = first low observation, `onset` = start+min_dur is the time
    the criterion is *confirmed* (the event time used for labels), `end` = last low observation."""
    idx = np.flatnonzero(low)
    if idx.size == 0:
        return []
    brk = np.flatnonzero((np.diff(idx) > 1) | (np.diff(t[idx]) > max_gap))
    starts = np.r_[0, brk + 1]
    ends = np.r_[brk, idx.size - 1]
    out = []
    for s, e in zip(starts, ends):
        t0, t1 = float(t[idx[s]]), float(t[idx[e]])
        if t1 - t0 >= min_dur:
            out.append((t0, t0 + min_dur, t1))
    return out


def merge_episodes(eps, gap: float = MERGE_GAP_MIN):
    """eps: list of (start, onset, end, etype). Overlapping / near-adjacent episodes -> one episode."""
    if not eps:
        return []
    eps = sorted(eps, key=lambda e: e[0])
    out = [list(eps[0])]
    for s, o, e, ty in eps[1:]:
        cur = out[-1]
        if s <= cur[2] + gap:
            cur[2] = max(cur[2], e)
            cur[1] = min(cur[1], o)
        else:
            out.append([s, o, e, ty])
    return [tuple(x) for x in out]


def build_events(endpoint: str, patient: pd.DataFrame, sidx: SeriesIndex) -> pd.DataFrame:
    """Episode table [stay_id, start, onset, end, etype] for one candidate endpoint.
    label time = `onset`; `start..end` is the window during which the patient is 'already in' the event
    (anchors inside it are excluded so the task is onset forecasting, not persistence)."""
    if endpoint not in CANDIDATES:
        raise ValueError(f"unknown endpoint {endpoint}")
    rows = []
    cols = ["stay_id", "unitdischargeoffset", "unitdischargestatus",
            "hospitaldischargeoffset", "hospitaldischargestatus"]
    for sid, uoff, ust, hoff, hst in patient[cols].itertuples(index=False):
        eps = []
        if endpoint in ("icu_death", "composite_collapse"):
            if str(ust).strip().lower() == "expired" and np.isfinite(uoff):
                eps.append((float(uoff), float(uoff), float(uoff), "icu_death"))
        if endpoint == "hospital_death":
            if str(hst).strip().lower() == "expired" and np.isfinite(hoff):
                eps.append((float(hoff), float(hoff), float(hoff), "hospital_death"))
        if endpoint in ("severe_hypotension", "composite_collapse"):
            d = HYPOTENSION
            t, v = sidx.get(sid, d["ch"])
            eps += [(s, o, e, "hypotension")
                    for s, o, e in sustained_runs(t, v < d["thr"], d["min_dur"], d["max_gap"])]
        if endpoint in ("severe_hypoxemia", "composite_collapse"):
            d = HYPOXEMIA
            t, v = sidx.get(sid, d["ch"])
            eps += [(s, o, e, "hypoxemia")
                    for s, o, e in sustained_runs(t, v < d["thr"], d["min_dur"], d["max_gap"])]
        for s, o, e, ty in merge_episodes(eps):
            rows.append((sid, s, o, e, ty))
    return pd.DataFrame(rows, columns=["stay_id", "start", "onset", "end", "etype"])


def anchor_hours(unit_discharge_min: float) -> np.ndarray:
    """Anchor hours a = 24..a_max with 60*a strictly before ICU discharge (the patient is still monitored)."""
    if not np.isfinite(unit_discharge_min):
        return np.empty(0, dtype=int)
    a_max = min(int(np.ceil(unit_discharge_min / GRID_MIN)) - 1, MAX_STAY_H)
    return np.arange(LOOKBACK_H, a_max + 1, dtype=int)


def make_labels(anchors: np.ndarray, start: np.ndarray, onset: np.ndarray, end: np.ndarray) -> dict:
    """Vectorised labels for one stay.  y_H(t) = 1 iff an event ONSET lies in (t, t+H].
    `valid` is False when the anchor lies inside an ongoing episode (start <= t <= end).
    Episodes must be disjoint and sorted (guaranteed by merge_episodes)."""
    t = anchors * float(GRID_MIN)
    n = len(t)
    out = {"anchor_hour": anchors, "anchor_min": t}
    if len(onset) == 0:
        for h in HORIZONS_H:
            out[f"y{h}"] = np.zeros(n, dtype=np.uint8)
        out["next_onset_min"] = np.full(n, np.nan)
        out["valid"] = np.ones(n, dtype=bool)
        return out
    lo = np.searchsorted(onset, t, side="right")                # first onset strictly after t
    for h in HORIZONS_H:
        hi = np.searchsorted(onset, t + h * GRID_MIN, side="right")
        out[f"y{h}"] = (hi > lo).astype(np.uint8)
    nxt = np.full(n, np.nan)
    ok = lo < len(onset)
    nxt[ok] = onset[lo[ok]]
    out["next_onset_min"] = nxt
    k = np.searchsorted(start, t, side="right") - 1
    inside = (k >= 0) & (t <= end[np.clip(k, 0, None)])
    out["valid"] = ~inside
    return out


# --------------------------------------------------------------------------- splits
def load_locked_task() -> dict:
    p = RESULTS / "audit" / "locked_task.json"
    if not p.exists():
        raise FileNotFoundError("results/audit/locked_task.json missing -- run 02_task_audit.py first")
    return load_json(p)


# --------------------------------------------------------------------------- metrics
def safe_auroc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float("nan") if y.min() == y.max() else float(roc_auc_score(y, p))


def safe_auprc(y, p) -> float:
    from sklearn.metrics import average_precision_score
    y = np.asarray(y)
    return float("nan") if y.sum() == 0 else float(average_precision_score(y, p))


def brier(y, p) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean((np.asarray(p) - y) ** 2))


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error with equal-frequency bins."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y) == 0:
        return float("nan")
    order = np.argsort(p)
    parts = np.array_split(order, min(bins, len(y)))
    return float(sum(len(ix) / len(y) * abs(y[ix].mean() - p[ix].mean()) for ix in parts if len(ix)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(p_val, y_val):
    """Platt scaling fitted on VALIDATION predictions only. Returns (coef, intercept)."""
    from sklearn.linear_model import LogisticRegression
    y_val = np.asarray(y_val)
    if y_val.min() == y_val.max():
        return 1.0, 0.0
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(logit(p_val).reshape(-1, 1), y_val)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(p, coef: float, intercept: float):
    z = coef * logit(p) + intercept
    return 1.0 / (1.0 + np.exp(-z))


def load_calibrators() -> dict:
    """{(model, horizon_h): (coef, intercept)} written by 09_evaluate.py."""
    p = RESULTS / "evaluation" / "calibrators.csv"
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return {(r.model, int(r.horizon)): (float(r.coef), float(r.intercept)) for r in df.itertuples()}


# --------------------------------------------------------------------------- alarm policy (used by 09, 10, 11)
def apply_alarm_policy(stay: np.ndarray, t: np.ndarray, above: np.ndarray,
                       k: int = 1, cooldown_min: float = 0.0) -> np.ndarray:
    """Persistence + cooldown controller.  Rows must be sorted by (stay, t).
    Alarm at anchor i iff risk exceeded tau at k consecutive hourly anchors ending at i AND at least
    `cooldown_min` minutes passed since the previous alarm of the same stay."""
    n = len(t)
    alarms = np.zeros(n, dtype=bool)
    run, last, prev_s, prev_t = 0, -1e18, None, None
    for i, (s, ti, a) in enumerate(zip(stay.tolist(), t.tolist(), above.tolist())):
        if s != prev_s:
            run, last = 0, -1e18
        elif ti - prev_t != GRID_MIN:
            run = 0                              # gap (anchors inside an event episode were excluded)
        run = run + 1 if a else 0
        if run >= k and ti - last >= cooldown_min:
            alarms[i] = True
            last = ti
        prev_s, prev_t = s, ti
    return alarms


def event_ids(stay: np.ndarray, y6: np.ndarray, next_onset: np.ndarray):
    """Integer id of the event that makes each anchor positive at the 6h horizon (-1 otherwise)."""
    key = pd.Series(list(zip(stay.tolist(), np.round(next_onset, 3).tolist())))
    ev = np.full(len(stay), -1, dtype=np.int64)
    pos = np.asarray(y6) == 1
    if pos.any():
        codes, _ = pd.factorize(key[pos])
        ev[pos] = codes
    return ev


def alarm_metrics(t: np.ndarray, alarms: np.ndarray, ev: np.ndarray, next_onset: np.ndarray) -> dict:
    """Alarm burden and early-warning metrics.
    * a alarm is TRUE if an event onset follows within 6 h (y6 == 1), else FALSE.
    * events = distinct onsets that had >=1 prediction opportunity in the preceding 6 h.
    * warning time = onset - time of the first true alarm for that event."""
    n = len(t)
    patient_days = n / 24.0
    n_alarm = int(alarms.sum())
    true_alarm = alarms & (ev >= 0)
    n_true = int(true_alarm.sum())
    n_events = int(len(np.unique(ev[ev >= 0])))
    if true_alarm.any():
        e_ids, first_idx = np.unique(ev[true_alarm], return_index=True)
        t_first = t[true_alarm][first_idx]
        onset = next_onset[true_alarm][first_idx]
        lead_h = (onset - t_first) / 60.0
    else:
        e_ids, lead_h = np.array([], dtype=int), np.array([])
    return {
        "patient_days": patient_days,
        "n_alarms": n_alarm,
        "alarms_per_patient_day": n_alarm / patient_days if patient_days else np.nan,
        "false_alarms_per_patient_day": (n_alarm - n_true) / patient_days if patient_days else np.nan,
        "alarm_precision": n_true / n_alarm if n_alarm else np.nan,
        "n_events": n_events,
        "events_detected": int(len(e_ids)),
        "event_sensitivity": len(e_ids) / n_events if n_events else np.nan,
        "warning_h_median": float(np.median(lead_h)) if len(lead_h) else np.nan,
        "warning_h_q25": float(np.percentile(lead_h, 25)) if len(lead_h) else np.nan,
        "warning_h_q75": float(np.percentile(lead_h, 75)) if len(lead_h) else np.nan,
    }