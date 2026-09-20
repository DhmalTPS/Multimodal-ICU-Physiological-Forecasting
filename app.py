"""
deployment/app.py  --  ICU Monitor Dilemma: research-demonstration dashboard (Streamlit)

A THIN deployment layer around the frozen research artifacts.  Nothing is retrained, re-fitted or re-scaled.

  model      : models/multimodal/multimodal.pt      (loaded with nets.load_model -> architecture from checkpoint)
  replay data: data/processed/test.npz              (held-out TEST split, already in the exact training representation:
                                                     scaled physiology / lab / text rows, modality masks, static vector)
  performance: results/**.csv                       (pre-computed, read at runtime, never recomputed here)
  alarm policy: results/alarms/alarm_metrics.csv    (tau, k, cooldown of the frozen operating point)
                + common.apply_alarm_policy         (the SAME persistence + cooldown automaton used in 10_alarm_analysis.py)

Causality contract (Proposition 4.1 of the report):
  the sample at replay hour `a` is rows a-23 .. a of the stay, produced by 06_build_features.py (each row g only sees
  data with timestamp <= 60*g min).  This app only SLICES those rows -- it never recomputes a rolling feature over the
  full record.  Whole-stay predictions are pre-computed with one independent window per anchor, so prediction i can
  never depend on anchor i+1; the UI still only DISPLAYS predictions up to the current replay position
  (unless the explicit "retrospective" toggle is on).  A runtime self-test (Methodology tab) poisons every row after
  the anchor and verifies the prediction is bit-identical.

Interactive perturbations (modality off / lab dropout / note delay) reuse the model's own missingness machinery:
  `off=` in the forward pass, and the model's learned "missing" templates (t_phys / t_lab / t_text) + mask columns.

Run from the repository root:   streamlit run deployment/app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ICU Monitor Dilemma - Research Demo", page_icon="🫀", layout="wide")


# ============================================================================ paths / imports
def _find_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here.parent, here, Path.cwd()):
        if (cand / "src" / "nets.py").exists():
            return cand
    return here.parent


ROOT = _find_root()
APP_DIR = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import common
    import nets
except Exception as _exc:  # noqa: BLE001
    st.error(f"Could not import the project modules from `{ROOT / 'src'}`.\n\n`{type(_exc).__name__}: {_exc}`\n\n"
             "The deployment needs `src/common.py`, `src/nets.py` and PyTorch (CPU build is enough).")
    st.stop()

MODEL_PATH = ROOT / "models" / "multimodal" / "multimodal.pt"
TEST_NPZ = ROOT / "data" / "processed" / "test.npz"
META_JSON = ROOT / "data" / "processed" / "metadata.json"
INTERIM = ROOT / "data" / "interim"
FIG = ROOT / "figures"

HORIZONS = tuple(common.HORIZONS_H)            # (1, 3, 6)
J6 = HORIZONS.index(6)
LOOKBACK = int(common.LOOKBACK_H)              # 24 rows: a-23 .. a
MODALITIES = tuple(nets.MODALITIES)            # ("phys", "lab", "text")
MOD_LABEL = {"phys": "Physiology", "lab": "Laboratory", "text": "Clinical notes"}
PHYS_CH = list(common.PHYS_CHANNELS)
PHYS_LABEL = {"hr": "Heart rate", "rr": "Resp. rate", "spo2": "SpO₂", "temp": "Temperature",
              "sbp": "SBP", "dbp": "DBP", "map": "MAP"}
PHYS_UNIT = {"hr": "bpm", "rr": "/min", "spo2": "%", "temp": "°C", "sbp": "mmHg", "dbp": "mmHg", "map": "mmHg"}
HORIZON_COLOR = {"1h": "#2ca02c", "3h": "#ff7f0e", "6h": "#1f77b4"}
MODEL_NAMES = {"logistic": "Logistic regression", "phys_only": "Physiology-only GRU", "lab_only": "Laboratory-only GRU",
               "text_only": "Text-only GRU", "concat_all": "Naive concatenation GRU",
               "multimodal_all": "Proposed multimodal"}


# ============================================================================ small helpers
def stretch(fn, *args, **kwargs):
    """Full-width rendering that survives the width= / use_container_width= API change across Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:  # noqa: BLE001
        return fn(*args, use_container_width=True, **kwargs)


@st.cache_data(show_spinner=False)
def read_csv(rel: str):
    p = ROOT / rel
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(show_spinner=False)
def read_json(rel: str):
    p = ROOT / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def pct(x, nd=1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100.0 * x:.{nd}f}%"


# ============================================================================ frozen resources
@st.cache_resource(show_spinner="Loading frozen checkpoint ...")
def get_model():
    model, _ = nets.load_model(MODEL_PATH)      # rebuilds the architecture from checkpoint metadata
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@st.cache_resource(show_spinner="Loading held-out replay data ...")
def get_data():
    return nets.WindowData(TEST_NPZ)


def spos_of(stay_id: int) -> int:
    return int(np.flatnonzero(get_data().stay_ids == stay_id)[0])


def stay_anchor_idx(d, spos: int) -> np.ndarray:
    idx = np.flatnonzero(d.a_stay == spos)
    return idx[np.argsort(d.a_hour[idx], kind="stable")]


@st.cache_data(show_spinner=False)
def anchor_hours_of(stay_id: int) -> np.ndarray:
    d = get_data()
    return d.a_hour[stay_anchor_idx(d, spos_of(stay_id))].astype(np.int64)


@st.cache_data(show_spinner=False)
def stay_table() -> pd.DataFrame:
    d = get_data()
    rows = []
    for spos, sid in enumerate(d.stay_ids):
        idx = stay_anchor_idx(d, spos)
        if len(idx) == 0:
            continue
        rs, n = int(d.row_start[spos]), int(d.n_rows[spos])
        r = rs + d.a_hour[idx]
        rows.append(dict(stay_id=int(sid), spos=spos, n_anchors=len(idx), first_h=int(d.a_hour[idx][0]),
                         last_h=int(d.a_hour[idx][-1]), n_pos6=int(d.y[idx, J6].sum()),
                         lab_cov=float((d.mod[r, 1] > 0).mean()), note_ever=int(d.mod[rs:rs + n, 2].max() > 0)))
    return pd.DataFrame(rows).sort_values("stay_id").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def demo_cases() -> dict:
    """Offline (pre-replay) selection of a few informative stays.  Letters are assigned in a hash-shuffled order so
    the case name carries NO outcome information; the category is only revealed in the retrospective view."""
    t = stay_table()
    ok = t[t.n_anchors >= 24]
    ev = ok[ok.n_pos6 > 0].sort_values(["n_pos6", "stay_id"], ascending=[False, True]).head(3)
    ne = ok[(ok.n_pos6 == 0) & (ok.n_anchors >= 48)]
    if len(ne):
        ne = ne.iloc[(ne.n_anchors - ne.n_anchors.median()).abs().argsort(kind="stable")].head(3)
    used = set(ev.stay_id) | set(ne.stay_id)
    sp = ok[~ok.stay_id.isin(used)].sort_values(["lab_cov", "stay_id"]).head(2)
    cat = {}
    for s in ev.stay_id:
        cat[int(s)] = "contains a labelled event"
    for s in ne.stay_id:
        cat[int(s)] = "no labelled event"
    for s in sp.stay_id:
        cat[int(s)] = "sparse laboratory coverage" + (" (event stay)" if int(t.loc[t.stay_id == s, "n_pos6"].iloc[0]) else "")
    ids = sorted(cat, key=lambda s: (s * 2654435761) % 9973)
    return {s: dict(label=f"Demo Case {chr(65 + i)}", category=cat[s]) for i, s in enumerate(ids)}


# ============================================================================ causal window construction
def local_arrays(d, model, spos: int, stay_id: int, lab_drop: float, delay_h: int):
    """Row arrays of ONE stay (rows are hours 0..n-1) with the optional interactive perturbations applied.
    Perturbations only ever replace rows with the model's own 'missing' templates / shift note availability LATER."""
    rs, n = int(d.row_start[spos]), int(d.n_rows[spos])
    phys = d.phys[rs:rs + n]
    lab = d.lab[rs:rs + n].copy()
    text = d.text[rs:rs + n].copy()
    mod = d.mod[rs:rs + n].astype(np.float32).copy()
    if lab_drop > 0:
        rng = np.random.default_rng([common.SEED, int(stay_id), int(round(lab_drop * 100))])
        drop = rng.random(n) < lab_drop
        lab[drop] = model.t_lab.detach().cpu().numpy()
        mod[drop, 1] = 0.0
    if delay_h > 0:
        # every note becomes visible `delay_h` hours later: the whole note timeline (embedding, hours-since-note,
        # count, flag) is shifted; rows before the first delayed note take the 'no note yet' template.
        tmpl = model.t_text.detach().cpu().numpy()
        new_text = np.tile(tmpl, (n, 1)).astype(np.float32)
        new_flag = np.zeros(n, dtype=np.float32)
        if n > delay_h:
            new_text[delay_h:] = text[:n - delay_h]
            new_flag[delay_h:] = mod[:n - delay_h, 2]
        text, mod[:, 2] = new_text, new_flag
    return phys, lab, text, mod


def windows(arrs, static_row: np.ndarray, hours: np.ndarray):
    """24 rows ending AT the anchor (a-23..a).  Same construction as nets.WindowData.window_np."""
    phys, lab, text, mod = arrs
    hours = np.asarray(hours, dtype=np.int64)
    rows = hours[:, None] + np.arange(-(LOOKBACK - 1), 1)[None, :]
    assert rows.min() >= 0 and np.all(rows[:, -1] == hours), "window must end exactly at the anchor row"
    stat = np.concatenate([np.repeat(static_row[None, :], len(hours), 0),
                           (np.log1p(hours) / 6.0)[:, None].astype(np.float32)], axis=1).astype(np.float32)
    return phys[rows], lab[rows], text[rows], mod[rows], stat


def to_batch(win) -> dict:
    p, l, t, m, s = win
    dev = nets.DEVICE
    return {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)).to(dev)
            for k, v in zip(("phys", "lab", "text", "mod", "static"), (p, l, t, m, s))}


def infer(model, batch: dict, off: tuple = ()) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(batch, off=tuple(off)).float()).cpu().numpy()


def run_windows(model, win, off: tuple = (), chunk: int = 1024) -> np.ndarray:
    p, l, t, m, s = win
    out = [infer(model, to_batch((p[i:i + chunk], l[i:i + chunk], t[i:i + chunk], m[i:i + chunk], s[i:i + chunk])), off)
           for i in range(0, len(s), chunk)]
    return np.concatenate(out, 0) if out else np.zeros((0, len(HORIZONS)), dtype=np.float32)


@st.cache_data(show_spinner=False, max_entries=256)
def stay_predictions(stay_id: int, off: tuple, lab_drop: float, delay_h: int) -> dict:
    """Independent 24-row window per anchor -> raw sigmoid outputs for the horizons (1h, 3h, 6h)."""
    model, d = get_model(), get_data()
    spos = spos_of(stay_id)
    idx = stay_anchor_idx(d, spos)
    hours = d.a_hour[idx].astype(np.int64)
    arrs = local_arrays(d, model, spos, stay_id, lab_drop, delay_h)
    probs = run_windows(model, windows(arrs, d.static[spos], hours), off)
    return dict(hours=hours, minutes=d.a_min[idx].astype(float), probs=probs,
                y=d.y[idx].astype(int), next_onset=d.next_onset[idx].astype(float))


SCENARIOS = [("Full model", (), 0.0, 0), ("Physiology off", ("phys",), 0.0, 0), ("Laboratory off", ("lab",), 0.0, 0),
             ("Notes off", ("text",), 0.0, 0), ("Laboratory + notes off", ("lab", "text"), 0.0, 0),
             ("Lab dropout 20%", (), 0.2, 0), ("Lab dropout 40%", (), 0.4, 0), ("Lab dropout 60%", (), 0.6, 0),
             ("Note delay 60 min", (), 0.0, 1), ("Note delay 180 min", (), 0.0, 3)]


@st.cache_data(show_spinner=False, max_entries=512)
def scenario_table(stay_id: int, hour: int) -> pd.DataFrame:
    model, d = get_model(), get_data()
    spos = spos_of(stay_id)
    rows = []
    for name, off, ld, dh in SCENARIOS:
        p = run_windows(model, windows(local_arrays(d, model, spos, stay_id, ld, dh), d.static[spos],
                                       np.array([hour])), off)[0]
        rows.append({"Scenario": name, **{f"{h}h": 100.0 * float(p[j]) for j, h in enumerate(HORIZONS)}})
    df = pd.DataFrame(rows)
    df["Δ 6h vs full (pp)"] = df["6h"] - df["6h"].iloc[0]
    return df


# ============================================================================ alarm controller (frozen policy)
@st.cache_data(show_spinner=False)
def operating_point() -> dict:
    df = read_csv("results/alarms/alarm_metrics.csv")
    if df is not None and len(df) and {"tau", "k", "cooldown_min"} <= set(df.columns):
        r = df.iloc[0]
        return dict(tau=float(r["tau"]), k=int(r["k"]), cooldown=float(r["cooldown_min"]), frozen=True,
                    source="results/alarms/alarm_metrics.csv", metrics=r.to_dict())
    return dict(tau=0.30, k=2, cooldown=60.0, frozen=False, source="FALLBACK (alarm_metrics.csv missing)", metrics=None)


def alarm_series(minutes, p6, tau, k, cooldown):
    above = np.asarray(p6) > tau
    alarms = common.apply_alarm_policy(np.zeros(len(minutes), dtype=np.int64), np.asarray(minutes, dtype=float),
                                       above, k=k, cooldown_min=cooldown)
    return alarms, above


def alarm_state(minutes, above, alarms, i, k, cooldown) -> dict:
    run = 0
    for j in range(i, -1, -1):
        if not above[j] or (j < i and minutes[j + 1] - minutes[j] != common.GRID_MIN):
            break
        run += 1
    fired = np.flatnonzero(alarms[:i + 1])
    last = float(minutes[fired[-1]]) if len(fired) else None
    since = None if last is None else float(minutes[i] - last)
    in_cd = since is not None and since < cooldown
    if alarms[i]:
        status = "fired"
    elif above[i] and run < k:
        status = "pending"
    elif above[i] and in_cd:
        status = "cooldown"
    else:
        status = "none"
    return dict(run=run, last_min=last, since=since, in_cooldown=in_cd, status=status)


# ============================================================================ attribution (occlusion, as in 11_explain.py)
@st.cache_data(show_spinner=False, max_entries=400)
def compute_attribution(stay_id: int, hour: int, off: tuple, lab_drop: float, delay_h: int) -> dict:
    model, d = get_model(), get_data()
    spos = spos_of(stay_id)
    b = to_batch(windows(local_arrays(d, model, spos, stay_id, lab_drop, delay_h), d.static[spos], np.array([hour])))

    def risk(bb, o):
        return float(infer(model, bb, o)[0, J6])

    base = risk(b, off)
    mod = {m: (base - risk(b, off + (m,)) if m not in off else float("nan")) for m in MODALITIES}
    blocks = [base - risk(nets.occlude_rows(model, b, k, k + 6), off) for k in range(0, LOOKBACK, 6)]
    n_ph, n_lb = d.dims["d_p"] // 5, d.dims["d_l"] // 5
    phys = [base - risk(nets.occlude_columns(model, b, "phys", c, 5), off) for c in range(n_ph)]
    labs = [base - risk(nets.occlude_columns(model, b, "lab", c, 5), off) for c in range(n_lb)]
    return dict(base=base, modality=mod, blocks=blocks, phys=phys, labs=labs)


def block_labels():
    out = []
    for k in range(0, LOOKBACK, 6):
        lo, hi = LOOKBACK - 1 - k, LOOKBACK - 1 - k - 5
        out.append(f"t−{lo}h … " + ("now" if hi == 0 else f"t−{hi}h"))
    return out


# ============================================================================ raw-unit context (optional, display only)
@st.cache_data(show_spinner=False)
def load_timeline():
    """Raw hourly physiology (display only; the model never sees it).  Slim copy for Cloud: deployment/demo_data/timeline_test.csv"""
    ids = {int(x) for x in get_data().stay_ids}
    for p in (APP_DIR / "demo_data" / "timeline_test.csv", INTERIM / "timeline.csv"):
        if p.exists():
            try:
                df = pd.read_csv(p, usecols=lambda c: c in ("stay_id", "hour") or c.endswith(("_mean", "_count")))
            except Exception:  # noqa: BLE001
                continue
            df = df[df["stay_id"].isin(ids)].set_index(["stay_id", "hour"]).sort_index()
            return df, p.name
    return None, None


def raw_value(tl, stay_id: int, hour: int, ch: str):
    try:
        row = tl.loc[(stay_id, hour)]
    except KeyError:
        return None
    if row.get(f"{ch}_count", 0) > 0 and pd.notna(row.get(f"{ch}_mean")):
        return float(row[f"{ch}_mean"])
    return None


def phys_state(stay_id: int, spos: int, hour: int) -> list:
    d = get_data()
    tl, _ = load_timeline()
    r = int(d.row_start[spos]) + hour
    out = []
    for c, ch in enumerate(PHYS_CH):
        obs = bool(d.phys[r, c * 5 + 3] > 0.5)
        tsl = float(d.phys[r, c * 5 + 4]) * 24.0
        val = raw_value(tl, stay_id, hour, ch) if (tl is not None and obs) else None
        prev = raw_value(tl, stay_id, hour - 1, ch) if tl is not None else None
        out.append(dict(ch=ch, obs=obs, val=val, prev=prev, z=float(d.phys[r, c * 5]),
                        last_h=None if tsl >= 24 else tsl))
    return out


def trend_frame(stay_id: int, spos: int, hour: int, chs=("map", "hr", "spo2")) -> pd.DataFrame:
    d = get_data()
    tl, _ = load_timeline()
    rs = int(d.row_start[spos])
    rows = []
    for g in range(max(hour - LOOKBACK + 1, 0), hour + 1):
        for ch in chs:
            c = PHYS_CH.index(ch)
            if d.phys[rs + g, c * 5 + 3] > 0.5:
                v = raw_value(tl, stay_id, g, ch) if tl is not None else float(d.phys[rs + g, c * 5])
                if v is not None:
                    rows.append(dict(hour=g, ch=ch, value=v))
    return pd.DataFrame(rows, columns=["hour", "ch", "value"])


# ============================================================================ session / sidebar
def _on_stay_change():
    st.session_state["pos"] = 0
    st.session_state["playing"] = False


def _step(delta: int, n: int):
    st.session_state["pos"] = int(np.clip(st.session_state.get("pos", 0) + delta, 0, n - 1))
    st.session_state["playing"] = False


def _toggle_play(n: int):
    if not st.session_state.get("playing", False) and st.session_state.get("pos", 0) >= n - 1:
        st.session_state["pos"] = 0
    st.session_state["playing"] = not st.session_state.get("playing", False)


def sidebar() -> dict:
    tbl, cases = stay_table(), demo_cases()
    with st.sidebar:
        st.markdown("### 🫀 ICU replay")
        ids = list(cases) + [int(s) for s in tbl.stay_id if int(s) not in cases]
        stay_id = st.selectbox("Demo ICU stay (held-out test cohort)", ids, key="stay_id", on_change=_on_stay_change,
                               format_func=lambda s: f"{cases[s]['label']} · stay {s}" if s in cases else f"Stay {s}")
        hours = anchor_hours_of(int(stay_id))
        n = len(hours)
        st.session_state.setdefault("pos", 0)
        st.session_state.setdefault("playing", False)
        if st.session_state.pop("_advance", False):
            st.session_state["pos"] += 1
        st.session_state["pos"] = int(np.clip(st.session_state["pos"], 0, n - 1))
        pos = st.select_slider("Simulated ICU hour", options=list(range(n)), key="pos",
                               format_func=lambda i: f"hour {int(hours[i])}")
        c1, c2, c3 = st.columns(3)
        c1.button("◀ Prev", on_click=_step, args=(-1, n))
        c2.button("⏸ Pause" if st.session_state["playing"] else "▶ Play", on_click=_toggle_play, args=(n,))
        c3.button("Next ▶", on_click=_step, args=(1, n))
        speed = st.select_slider("Replay speed (steps / s)", options=[0.5, 1, 2, 4], value=1)
        st.caption(f"Anchor {pos + 1} of {n} · hours {int(hours[0])}–{int(hours[-1])} of the ICU stay")

        st.markdown("### Modality availability")
        on = [st.toggle(f"{MOD_LABEL[m]}", True, key=f"use_{m}") for m in MODALITIES]
        off = tuple(m for m, u in zip(MODALITIES, on) if not u)

        st.markdown("### Robustness simulation")
        lab_pct = st.select_slider("Random laboratory dropout", options=[0, 20, 40, 60], value=0,
                                   format_func=lambda v: f"{v}%")
        delay_min = st.select_slider("Note documentation delay", options=[0, 60, 180], value=0,
                                     format_func=lambda v: f"{v} min")
        st.markdown("### View")
        retro = st.toggle("Retrospective view (reveals future predictions & outcomes)", False)
        if retro and int(stay_id) in cases:
            st.caption(f"Case category: {cases[int(stay_id)]['category']}")
    return dict(stay_id=int(stay_id), pos=int(pos), n=n, hours=hours, off=off, lab_drop=lab_pct / 100.0,
                delay_h=int(delay_min // 60), speed=float(speed), retro=retro)


# ============================================================================ chart helpers
def delta_bar(df: pd.DataFrame, ycol: str, height: int) -> alt.Chart:
    return (alt.Chart(df).mark_bar().encode(
        x=alt.X("delta:Q", title="Δ 6h risk (base − occluded)", axis=alt.Axis(format="+.1%")),
        y=alt.Y(f"{ycol}:N", sort=None, title=None),
        color=alt.condition(alt.datum.delta >= 0, alt.value("#d35400"), alt.value("#2c7fb8")),
        tooltip=[ycol, alt.Tooltip("delta:Q", format="+.2%")]).properties(height=height))


def trajectory_chart(hours, probs, i, alarms, tau, events_h, retro) -> alt.Chart:
    n = len(hours) if retro else i + 1
    long = pd.DataFrame({"hour": np.repeat(hours[:n], len(HORIZONS)),
                         "horizon": np.tile([f"{h}h" for h in HORIZONS], n), "risk": probs[:n].reshape(-1)})
    ymax = float(min(1.0, max(tau * 1.5, float(probs[:n].max()) * 1.15, 0.05)))
    lines = alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=18)).encode(
        x=alt.X("hour:Q", title="ICU hour", scale=alt.Scale(domain=[float(hours[0]), float(hours[-1])], nice=False)),
        y=alt.Y("risk:Q", title="Predicted probability (raw model output)", axis=alt.Axis(format=".0%"),
                scale=alt.Scale(domain=[0, ymax])),
        color=alt.Color("horizon:N", scale=alt.Scale(domain=list(HORIZON_COLOR), range=list(HORIZON_COLOR.values())),
                        title="Horizon"),
        tooltip=["hour", "horizon", alt.Tooltip("risk:Q", format=".2%")])
    layers = [lines,
              alt.Chart(pd.DataFrame({"y": [tau]})).mark_rule(strokeDash=[5, 4], color="#c0392b").encode(y="y:Q"),
              alt.Chart(pd.DataFrame({"x": [float(hours[i])]})).mark_rule(color="#555", size=2).encode(x="x:Q")]
    am = np.flatnonzero(alarms[:n])
    if len(am):
        adf = pd.DataFrame({"hour": hours[am], "risk": probs[am, J6]})
        layers.append(alt.Chart(adf).mark_point(shape="triangle-up", size=140, filled=True, color="#c0392b",
                                                opacity=0.9).encode(x="hour:Q", y="risk:Q",
                                                                    tooltip=[alt.Tooltip("hour:Q", title="alarm at hour")]))
    if retro and len(events_h):
        layers.append(alt.Chart(pd.DataFrame({"x": events_h})).mark_rule(color="black", strokeDash=[2, 2], size=2)
                      .encode(x="x:Q", tooltip=[alt.Tooltip("x:Q", title="observed event onset (h)")]))
    return alt.layer(*layers).properties(height=300)


# ============================================================================ LIVE FORECAST TAB
def render_live(ctx: dict):
    model, d = get_model(), get_data()
    stay_id, i, hours = ctx["stay_id"], ctx["pos"], ctx["hours"]
    hour = int(hours[i])
    spos = spos_of(stay_id)
    off, lab_drop, delay_h, retro = ctx["off"], ctx["lab_drop"], ctx["delay_h"], ctx["retro"]
    op = operating_point()
    cases = demo_cases()

    if off or lab_drop > 0 or delay_h > 0:
        parts = ([f"{MOD_LABEL[m]} disabled" for m in off] + ([f"{int(lab_drop * 100)}% lab dropout"] if lab_drop else [])
                 + ([f"notes delayed {delay_h * 60} min"] if delay_h else []))
        st.warning("**Interactive robustness simulation active** – " + "; ".join(parts) +
                   ". Outputs below are a demonstration of model behaviour, not clinical performance.")
    if len(off) == len(MODALITIES):
        st.error("All modalities are disabled: the model receives only static features and 'missing' templates.")

    P = stay_predictions(stay_id, off, lab_drop, delay_h)
    probs = P["probs"]
    if not np.all(np.isfinite(probs)) or probs.min() < 0 or probs.max() > 1:
        st.error("Model produced non-finite / out-of-range probabilities - aborting this view.")
        return
    alarms, above = alarm_series(P["minutes"], probs[:, J6], op["tau"], op["k"], op["cooldown"])
    ast = alarm_state(P["minutes"], above, alarms, i, op["k"], op["cooldown"])

    name = cases.get(stay_id, {}).get("label", f"Stay {stay_id}")
    st.markdown(f"#### {name} · stay `{stay_id}` · ICU hour **{hour}** &nbsp; "
                f"<span style='color:gray'>(prediction-time view: only data documented at or before hour {hour})</span>",
                unsafe_allow_html=True)

    st.caption("Model input = 24 rows of `data/processed/test.npz` ending at this hour → `models/multimodal/multimodal.pt`. "
               "Raw-unit vitals, lab and note panels are display-only context (`data/interim/*`). See the **Data & Files** tab.")

    # ---- current patient state ---------------------------------------------------------------
    st.markdown("##### Current physiological state")
    ps = phys_state(stay_id, spos, hour)
    have_raw = load_timeline()[0] is not None
    for row_chs in (ps[:4], ps[4:]):
        cols = st.columns(4)
        for col, s in zip(cols, row_chs):
            lab = f"{PHYS_LABEL[s['ch']]} ({PHYS_UNIT[s['ch']]})" if have_raw else f"{PHYS_LABEL[s['ch']]} (z-score)"
            if s["obs"]:
                if have_raw and s["val"] is not None:
                    v = f"{s['val']:.1f}" if s["ch"] == "temp" else f"{s['val']:.0f}"
                    dlt = None if s["prev"] is None else f"{s['val'] - s['prev']:+.1f} vs prev h"
                    col.metric(lab, v, dlt, delta_color="off")
                else:
                    col.metric(lab, f"{s['z']:+.2f}")
            else:
                col.metric(lab, "Unavailable")
                col.caption("never observed" if s["last_h"] is None else f"last seen {s['last_h']:.0f} h ago")
    if not have_raw:
        st.caption("Raw-unit values need `data/interim/timeline.csv` (or `deployment/demo_data/timeline_test.csv`); "
                   "showing the train-normalised features the model actually consumes.")
    if "phys" in off:
        st.caption("⚠ Physiology is disabled *for the model* in this simulation; values above are shown for context only.")
    mrow = d.mod[int(d.row_start[spos]) + hour]
    st.caption(f"Modality availability at this hour → physiology observed: {'yes' if mrow[0] else 'no'} · "
               f"any lab so far: {'yes' if mrow[1] else 'no'} · new note documented this hour: {'yes' if mrow[2] else 'no'}")

    # ---- forecast cards ---------------------------------------------------------------------
    st.markdown("##### Predicted probability of composite deterioration")
    cols = st.columns(3)
    for j, (h, col) in enumerate(zip(HORIZONS, cols)):
        dlt = None
        if i > 0 and P["minutes"][i] - P["minutes"][i - 1] == common.GRID_MIN:
            dlt = f"{100 * (probs[i, j] - probs[i - 1, j]):+.1f} pp vs prev hour"
        col.metric(f"{h} HOUR{'S' if h > 1 else ''}", pct(probs[i, j]), dlt, delta_color="inverse")
    st.caption("Composite endpoint = first onset of sustained hypotension (MAP<55 mmHg ≥30 min), sustained hypoxemia "
               "(SpO₂<88% ≥30 min) or ICU death within the horizon. Values are **raw, uncalibrated model outputs** "
               "(training up-weights rare positives, so they are not observed frequencies) – not confidence, not mortality.")

    # ---- trajectory -------------------------------------------------------------------------
    st.markdown("##### Risk trajectory" + (" (retrospective – full replay shown)" if retro else " (up to current hour)"))
    ev = P["next_onset"][np.isfinite(P["next_onset"])]
    events_h = np.unique(np.round(ev / 60.0, 2)) if retro else np.array([])
    stretch(st.altair_chart, trajectory_chart(hours, probs, i, alarms, op["tau"], events_h, retro))
    st.caption("Dashed red line = alarm threshold on the 6 h score · red triangles = alarms fired by the controller"
               + (" · black dashed = observed event onsets that followed a replay anchor." if retro else "."))
    if retro:
        st.info(f"Retrospective analysis: {len(events_h)} event onset(s) followed at least one anchor of this stay "
                f"({', '.join(f'h{e:.1f}' for e in events_h) if len(events_h) else 'none'}). "
                "This is NOT shown in the prediction-time view.")

    # ---- alarm + attribution ----------------------------------------------------------------
    left, right = st.columns([1, 1])
    with left:
        st.markdown("##### Alarm controller")
        msg = {"fired": ("error", f"🔔 ALARM FIRED at hour {hour}"),
               "pending": ("warning", f"Risk above threshold - persistence not yet satisfied ({ast['run']}/{op['k']})"),
               "cooldown": ("warning", "Risk above threshold - alarm suppressed by cooldown"),
               "none": ("success", "No alarm")}[ast["status"]]
        getattr(st, msg[0])(msg[1])
        a, b = st.columns(2)
        a.metric("Current 6h score", pct(probs[i, J6], 2))
        b.metric("Threshold τ", f"{op['tau']:.2f}")
        a.metric("Consecutive qualifying anchors", f"{ast['run']} / {op['k']}")
        b.metric("Cooldown", f"{op['cooldown']:.0f} min" + (" · active" if ast["in_cooldown"] else ""))
        st.caption(f"Policy: score > τ for {op['k']} consecutive hourly anchors, then ≥{op['cooldown']:.0f} min since the previous "
                   f"alarm (`common.apply_alarm_policy`). Operating point read from `{op['source']}`"
                   + ("" if op["frozen"] else " - **frozen artifact not found, fallback used**") +
                   "; threshold is applied to the raw 6h score, as in `10_alarm_analysis.py`.")
    with right:
        st.markdown("##### Model attribution")
        A = compute_attribution(stay_id, hour, off, lab_drop, delay_h)
        mdf = pd.DataFrame({"modality": [MOD_LABEL[m] for m in MODALITIES],
                            "delta": [A["modality"][m] for m in MODALITIES]}).dropna()
        if len(mdf):
            stretch(st.altair_chart, delta_bar(mdf, "modality", 110))
        st.caption("Occlusion attribution of the 6 h score: Δ = f(x) − f(x without modality). "
                   "**Model attribution – not a causal clinical explanation.**")
        with st.expander("Which hours and which variables?"):
            tdf = pd.DataFrame({"window": block_labels(), "delta": A["blocks"]})
            st.markdown("**Time windows of the 24 h look-back**")
            stretch(st.altair_chart, delta_bar(tdf, "window", 150))
            lab_names = (read_json("data/processed/metadata.json") or {}).get("lab_names") or []
            lab_names = [str(x) for x in lab_names][:len(A["labs"])] or [f"lab_{k}" for k in range(len(A["labs"]))]
            fdf = pd.DataFrame({"feature": [PHYS_LABEL.get(c, c) for c in PHYS_CH[:len(A["phys"])]] + lab_names,
                                "delta": list(A["phys"]) + list(A["labs"])})
            fdf = fdf.reindex(fdf.delta.abs().sort_values(ascending=False).index).head(6)
            st.markdown("**Top variables (physiology channels + lab variables)**")
            stretch(st.altair_chart, delta_bar(fdf, "feature", 190))

    # ---- physiology trend / labs / notes ----------------------------------------------------
    with st.expander("Recent physiology (last 24 h, observed hours only)", expanded=True):
        tf = trend_frame(stay_id, spos, hour)
        cols = st.columns(3)
        for col, ch in zip(cols, ("map", "hr", "spo2")):
            sub = tf[tf.ch == ch]
            if sub.empty:
                col.caption(f"{PHYS_LABEL[ch]}: no observations")
                continue
            chart = alt.Chart(sub).mark_line(point=True).encode(
                x=alt.X("hour:Q", title="ICU hour", scale=alt.Scale(nice=False)),
                y=alt.Y("value:Q", title=PHYS_UNIT[ch] if have_raw else "z", scale=alt.Scale(zero=False)),
                tooltip=["hour", alt.Tooltip("value:Q", format=".1f")]).properties(title=PHYS_LABEL[ch], height=150)
            with col:
                stretch(st.altair_chart, chart)

    with st.expander("Laboratory results available at this hour"):
        labs = read_csv("data/interim/lab_clean.csv")
        if "lab" in off:
            st.caption("Laboratory modality disabled in this simulation.")
        elif labs is None:
            st.caption("`data/interim/lab_clean.csv` not available - raw lab values cannot be displayed.")
        else:
            x = labs[(labs.stay_id == stay_id) & (labs.t <= 60.0 * hour)]
            if x.empty:
                st.caption("No laboratory result documented yet.")
            else:
                last = x.sort_values("t").groupby("name").tail(1).set_index("name")
                n24 = x[x.t > 60.0 * hour - 1440].groupby("name").size()
                tab = pd.DataFrame({"Latest value": last["value"], "Age (h)": (60.0 * hour - last["t"]) / 60.0,
                                    "Results in last 24 h": n24.reindex(last.index).fillna(0).astype(int)}).round(2)
                stretch(st.dataframe, tab.reset_index().rename(columns={"name": "Lab"}), hide_index=True)

    with st.expander("Clinical notes documented at or before this hour (de-identified demo data)"):
        notes = read_csv("data/interim/note_clean.csv")
        if "text" in off:
            st.caption("Notes modality disabled in this simulation.")
        elif notes is None:
            st.caption("`data/interim/note_clean.csv` not available.")
        else:
            x = notes[(notes.stay_id == stay_id) & (notes.avail_min + 60.0 * delay_h <= 60.0 * hour)]
            st.caption(f"{len(x)} note(s) visible" + (f" (documentation delayed by {delay_h * 60} min)" if delay_h else "")
                       + " - availability time = documentation time (noteEnteredOffset), not clinical time.")
            for _, r in x.sort_values("avail_min").tail(3).iterrows():
                st.markdown(f"- **h{(r.avail_min + 60.0 * delay_h) / 60:.1f}** · {r.note_type} · "
                            f"{str(r.text)[:160].replace(chr(10), ' ')}…")


# ============================================================================ ROBUSTNESS TAB
def render_robustness(ctx: dict):
    stay_id, i, hours = ctx["stay_id"], ctx["pos"], ctx["hours"]
    hour = int(hours[i])
    st.markdown("#### Interactive robustness simulation")
    st.caption("Same frozen weights, perturbed inputs. Modality removal uses the model's `off=` mechanism; lab dropout and "
               "note delay replace / shift hourly rows with the model's own 'missing' representation. This is a single-patient "
               "demonstration – **not** the held-out evaluation (see the Model Performance tab for that).")
    st.markdown(f"**Stay {stay_id}, hour {hour}** - all scenarios use the same causal 24 h window.")
    tab = scenario_table(stay_id, hour)
    stretch(st.dataframe, tab, hide_index=True, column_config={
        "1h": st.column_config.NumberColumn(format="%.2f%%"), "3h": st.column_config.NumberColumn(format="%.2f%%"),
        "6h": st.column_config.NumberColumn(format="%.2f%%"),
        "Δ 6h vs full (pp)": st.column_config.NumberColumn(format="%+.2f")})
    chart = alt.Chart(tab).mark_bar().encode(
        x=alt.X("6h:Q", title="Predicted 6h probability (%)"), y=alt.Y("Scenario:N", sort=None, title=None),
        color=alt.condition(alt.datum.Scenario == "Full model", alt.value("#1f77b4"), alt.value("#9aa5b1")),
        tooltip=["Scenario", alt.Tooltip("6h:Q", format=".2f")]).properties(height=280)
    stretch(st.altair_chart, chart)

    st.markdown("##### Trajectory under selected scenarios")
    names = [s[0] for s in SCENARIOS]
    pick = st.multiselect("Scenarios", names, default=["Full model", "Physiology off", "Laboratory off"])
    rows = []
    n = len(hours) if ctx["retro"] else i + 1
    for nm, off, ld, dh in SCENARIOS:
        if nm in pick:
            P = stay_predictions(stay_id, off, ld, dh)
            rows.append(pd.DataFrame({"hour": hours[:n], "scenario": nm, "risk": P["probs"][:n, J6]}))
    if rows:
        df = pd.concat(rows)
        ch = alt.Chart(df).mark_line().encode(
            x=alt.X("hour:Q", title="ICU hour", scale=alt.Scale(domain=[float(hours[0]), float(hours[-1])], nice=False)),
            y=alt.Y("risk:Q", title="6h probability", axis=alt.Axis(format=".0%")), color="scenario:N",
            tooltip=["hour", "scenario", alt.Tooltip("risk:Q", format=".2%")]).properties(height=280)
        rule = alt.Chart(pd.DataFrame({"x": [float(hours[i])]})).mark_rule(color="#555").encode(x="x:Q")
        stretch(st.altair_chart, ch + rule)

    st.markdown("##### Reported held-out results (pre-computed, 6h AUPRC)")
    mm = read_csv("results/evaluation/missing_modality.csv")
    if mm is None:
        st.info("`results/evaluation/missing_modality.csv` not found.")
    else:
        t = mm[mm.horizon_h == 6][["perturbation", "auroc", "auprc"]].drop_duplicates().sort_values("auprc", ascending=False)
        stretch(st.dataframe, t.round(4), hide_index=True)
        st.caption("Test-set differences among perturbations that keep physiology are well inside the bootstrap intervals; "
                   "removing physiology is the only perturbation with a large drop.")


# ============================================================================ MODEL PERFORMANCE TAB
def render_performance():
    st.markdown("#### Held-out test performance (pre-computed artifacts)")
    st.caption("Every number on this page is read from `results/**` at runtime – nothing is re-evaluated here, and the replay "
               "patient above is never used to compute performance.")
    m = read_csv("results/evaluation/metrics.csv")
    if m is None:
        st.info("`results/evaluation/metrics.csv` not found.")
    else:
        t = m[(m.split == "test") & (m.model.isin(MODEL_NAMES))]
        piv = t.pivot_table(index="model", columns="horizon_h", values=["auprc", "auroc"])
        piv.columns = [f"{a.upper()} {h}h" for a, h in piv.columns]
        piv = piv.reindex([k for k in MODEL_NAMES if k in piv.index]).rename(index=MODEL_NAMES).round(4)
        piv = piv[[c for c in piv.columns if c.startswith("AUPRC")] + [c for c in piv.columns if c.startswith("AUROC")]]
        stretch(st.dataframe, piv.reset_index().rename(columns={"model": "Model"}), hide_index=True)
        r6 = t[t.horizon_h == 6].iloc[0] if (t.horizon_h == 6).any() else None
        if r6 is not None:
            st.caption(f"Test cohort at 6 h: {int(r6.n):,} anchors, {int(r6.n_pos)} positives (prevalence {pct(r6.prevalence, 2)}). "
                       "AUPRC of a random ranker ≈ prevalence.")

    st.markdown("##### Uncertainty - stay-level cluster bootstrap (6 h AUPRC)")
    ci, df_ = read_csv("results/evaluation/bootstrap_ci_6h.csv"), read_csv("results/evaluation/bootstrap_differences_6h.csv")
    if ci is None:
        st.info("`results/evaluation/bootstrap_ci_6h.csv` not found.")
    else:
        ci = ci.assign(name=ci.model.map(MODEL_NAMES).fillna(ci.model))
        base = alt.Chart(ci)
        chart = (base.mark_rule(size=3).encode(x=alt.X("ci_lower:Q", title="6h AUPRC (95% CI)"), x2="ci_upper:Q",
                                               y=alt.Y("name:N", sort=None, title=None)) +
                 base.mark_point(filled=True, size=90, color="#1f77b4").encode(x="auprc:Q", y=alt.Y("name:N", sort=None),
                                                                                tooltip=["name", alt.Tooltip("auprc:Q", format=".4f")]))
        stretch(st.altair_chart, chart.properties(height=190))
        show = ci[["name", "auprc", "ci_lower", "ci_upper", "n_stays", "n_positive"]].rename(
            columns={"name": "Model", "auprc": "6h AUPRC", "ci_lower": "CI low", "ci_upper": "CI high",
                     "n_stays": "Stays", "n_positive": "Positive anchors"})
        stretch(st.dataframe, show.round(4), hide_index=True)
    if df_ is not None:
        df_ = df_.assign(**{"CI includes zero": (df_.ci_lower <= 0) & (df_.ci_upper >= 0)})
        stretch(st.dataframe, df_[["comparison", "delta_auprc", "ci_lower", "ci_upper", "CI includes zero"]].round(4),
                hide_index=True)
        if df_["CI includes zero"].all():
            st.info("Every paired interval includes zero: the data do **not** establish a difference between the multimodal "
                    "model and any comparator at this sample size. Point estimates are not a ranking.")
    st.caption("Hourly anchors within one ICU stay are correlated; uncertainty is therefore estimated by resampling whole stays "
               "(B = 2000), paired across models for differences.")

    st.markdown("##### Ablation (single checkpoint, inference-time modality masking)")
    ab = read_csv("results/evaluation/ablations.csv")
    if ab is not None:
        a6 = ab[(ab.split == "test") & (ab.horizon_h == 6)].sort_values("auprc", ascending=False)
        c1, c2 = st.columns([1, 1])
        with c1:
            stretch(st.dataframe, a6[["ablation_set", "auroc", "auprc", "brier"]].round(4), hide_index=True)
        with c2:
            stretch(st.altair_chart, alt.Chart(a6).mark_bar().encode(
                x=alt.X("auprc:Q", title="Test 6h AUPRC"), y=alt.Y("ablation_set:N", sort="-x", title=None),
                tooltip=["ablation_set", alt.Tooltip("auprc:Q", format=".4f")]).properties(height=230))
    else:
        st.info("`results/evaluation/ablations.csv` not found.")

    st.markdown("##### Stress tests")
    st_ = read_csv("results/evaluation/stress_tests.csv")
    if st_ is not None and len(st_) and "parameter" in st_.columns:
        s6 = st_[st_.horizon_h == 6].drop_duplicates(["stress_test", "parameter"])[["stress_test", "parameter", "auroc", "auprc"]]
        stretch(st.dataframe, s6.round(4), hide_index=True)

    st.markdown("##### Calibration")
    if m is not None:
        cal = m[(m.split == "test") & (m.model == "multimodal_all")][["horizon_h", "brier", "brier_calibrated", "ece", "ece_calibrated"]]
        stretch(st.dataframe, cal.round(5), hide_index=True)
        st.caption("Platt scaling was fitted on validation only. The live view above shows RAW outputs (no calibrator applied).")
    for f, cap in (("calibration.png", "Calibration curves (from 09_evaluate.py)"), ("lead_time.png", "Lead-time distribution")):
        if (FIG / f).exists():
            st.image(str(FIG / f), caption=cap)

    st.markdown("##### Alarm operating point and trade-off")
    op = operating_point()
    if op["metrics"] is None:
        st.info("`results/alarms/alarm_metrics.csv` not found.")
    else:
        r = op["metrics"]
        st.caption(f"Frozen operating point from `{op['source']}`: τ = {op['tau']:.2f}, persistence k = {op['k']}, "
                   f"cooldown = {op['cooldown']:.0f} min (threshold selected on validation, applied once to test).")
        c = st.columns(4)
        c[0].metric("Alarms / patient-day", f"{r['alarms_per_patient_day']:.2f}")
        c[1].metric("False alarms / patient-day", f"{r['false_alarms_per_patient_day']:.2f}")
        c[2].metric("Alarm precision", pct(r["alarm_precision"], 2))
        c[3].metric("Event sensitivity", pct(r["event_sensitivity"], 1))
        c = st.columns(4)
        c[0].metric("Events detected", f"{int(r['events_detected'])} / {int(r['n_events'])}")
        c[1].metric("Median warning", f"{r['warning_h_median']:.2f} h")
        c[2].metric("Warning Q25", f"{r['warning_h_q25']:.2f} h")
        c[3].metric("Warning Q75", f"{r['warning_h_q75']:.2f} h")
        st.caption("Alarm burden is reported openly: the alarm-fatigue trade-off is part of the scientific result.")
    sw = read_csv("results/alarms/alarm_threshold_sweep.csv")
    if sw is not None:
        sw = sw.assign(tau_s=sw.tau.map(lambda v: f"{v:.2f}"))
        line = alt.Chart(sw).mark_line(point=True, color="#444").encode(
            x=alt.X("alarms_per_patient_day:Q", title="Alarms / patient-day"),
            y=alt.Y("event_sensitivity:Q", title="Event sensitivity", axis=alt.Axis(format=".0%")),
            tooltip=["tau_s", alt.Tooltip("event_sensitivity:Q", format=".1%"),
                     alt.Tooltip("alarms_per_patient_day:Q", format=".2f"), alt.Tooltip("alarm_precision:Q", format=".2%")])
        layers = [line]
        sel = sw[np.isclose(sw.tau, op["tau"])] if op["frozen"] else sw.iloc[0:0]
        if len(sel):
            layers.append(alt.Chart(sel).mark_point(size=180, filled=True, color="#c0392b").encode(
                x="alarms_per_patient_day:Q", y="event_sensitivity:Q", tooltip=["tau_s"]))
        stretch(st.altair_chart, alt.layer(*layers).properties(height=300))
        st.caption("Threshold sweep on VALIDATION predictions (k, cooldown fixed); red = the frozen operating point when it lies on the grid.")
    lt = read_csv("results/evaluation/lead_time_summary.csv")
    if lt is not None:
        st.markdown("**Lead-time summary** (positive 6 h anchors, `lead_time_summary.csv`)")
        stretch(st.dataframe, lt.round(3), hide_index=True)


# ============================================================================ METHODOLOGY TAB
@st.cache_data(show_spinner=False)
def integrity_checks(stay_id: int) -> list:
    model, d = get_model(), get_data()
    spos = spos_of(stay_id)
    idx = stay_anchor_idx(d, spos)
    hours = d.a_hour[idx].astype(np.int64)
    out = []
    dims_ok = all(int(model.dims[k]) == int(d.dims[k]) for k in ("d_p", "d_l", "d_t", "d_s"))
    out.append(("Checkpoint feature dimensions equal processed-data dimensions", dims_ok, f"{dict(d.dims)}"))
    arrs = local_arrays(d, model, spos, stay_id, 0.0, 0)
    mine, ref = windows(arrs, d.static[spos], hours), d.window_np(idx)
    out.append(("Window extraction is identical to nets.WindowData.window_np",
                all(np.array_equal(a, b) for a, b in zip(mine, ref)), f"{len(hours)} anchors"))
    P = stay_predictions(stay_id, (), 0.0, 0)
    ok = bool(np.all(np.isfinite(P["probs"])) and P["probs"].min() >= 0 and P["probs"].max() <= 1)
    out.append(("Predictions finite and within [0, 1]", ok, f"range {P['probs'].min():.4f} – {P['probs'].max():.4f}"))
    n = int(d.n_rows[spos])
    mid = len(hours) // 2
    a = int(hours[mid])
    if a + 1 < n:
        base = run_windows(model, windows(arrs, d.static[spos], hours[mid:mid + 1]))
        poisoned = tuple(x.copy() for x in arrs)
        for x in poisoned:
            x[a + 1:] = 1e6
        pois = run_windows(model, windows(poisoned, d.static[spos], hours[mid:mid + 1]))
        out.append((f"Future-poisoning test: rows after hour {a} overwritten with junk → prediction unchanged",
                    bool(np.allclose(base, pois, atol=1e-7, rtol=0)), f"max |Δ| = {float(np.abs(base - pois).max()):.2e}"))
    else:
        out.append(("Future-poisoning test", True, "n/a - no rows after the chosen anchor"))
    out.append(("All replay anchors have a complete 24 h look-back (hour ≥ 24)", bool(hours.min() >= LOOKBACK), f"min hour {int(hours.min())}"))
    return out


def render_methodology(ctx: dict):
    meta = read_json("data/processed/metadata.json") or {}
    task = read_json("results/audit/locked_task.json") or {}
    st.markdown("#### Methodology")
    st.code("raw eICU tables  (offline research pipeline)\n"
            "   → hourly temporal alignment (row g uses only data with timestamp ≤ 60·g min)\n"
            "   → 24-hour causal window  [t−23h … t]\n"
            "   → physiology (35) | laboratory (80) | notes (66) | static (14 + log-hour) | modality mask (3)\n"
            "   → frozen multimodal network (TCN, lab MLP, note projection, causal cross-attention, reliability gate, GRU)\n"
            "   → p(1h), p(3h), p(6h)   [composite deterioration onset]\n"
            "   → alarm controller: score > τ for k consecutive anchors + cooldown\n"
            "   → this dashboard", language="text")
    st.markdown("**Causal constraint.** Only information available at or before the current anchor may be used. Notes become "
                "available at their *documentation* time (noteEnteredOffset), not their clinical time; scalers, lab selection and "
                "the text vocabulary were fitted on TRAIN stays only; the split is by patient.")
    st.markdown("**What this deployment does.** It *replays* held-out (test-split) eICU-demo stays. For a chosen hour it slices the "
                "24 pre-computed feature rows ending at that hour and runs the frozen checkpoint on CPU. No preprocessing is refit, "
                "no external API is called, and no raw clinical data leave the app.")
    if task.get("definition"):
        st.markdown(f"**Endpoint:** {task['definition']}.")
    if meta:
        te = meta.get("text_encoder", {})
        st.markdown(f"**Feature dimensions:** {meta.get('dims')} · text encoder: `{te.get('kind')}` · look-back rows: "
                    f"{meta.get('lookback_rows')} · horizons (h): {meta.get('horizons_h')}")
        if meta.get("splits", {}).get("test"):
            s = meta["splits"]["test"]
            st.markdown(f"**Replay cohort (test split):** {s.get('n_stays')} stays, {s.get('n_anchors'):,} anchors.")
    st.markdown("**Limitations.** Demo-scale data (eICU-CRD Demo), single split, retrospective; the note encoder is a lightweight frozen "
                "TF-IDF+SVD; attributions describe the network, not causal physiology; the null multimodal-vs-physiology result "
                "is reported, not hidden. **Research demonstration only – not for clinical use.**")
    with st.expander("Deployment integrity checks (run live for the selected stay)", expanded=False):
        for name, ok, detail in integrity_checks(ctx["stay_id"]):
            st.markdown(f"{'✅' if ok else '❌'} {name} - `{detail}`")
        st.caption("Feature-row causality (truncation test) is verified in `06_build_features.py`; the checks above verify that this "
                   "app's slicing/inference path adds no leakage of its own.")
    tl_name = load_timeline()[1]
    st.caption(f"Repository root: `{ROOT}` · device: `{nets.DEVICE}` · raw-unit display source: `{tl_name or 'not available (z-scores shown)'}`")


# ============================================================================ DATA & FILES TAB
def _fmt_size(p: Path) -> str:
    try:
        n = p.stat().st_size
    except OSError:
        return "-"
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{max(n / 1e3, 0.1):.0f} KB"


# (title, note, [(alternative paths relative to repo root, role in the app, required?, what happens if absent)])
FILE_GROUPS = [
    ("🔴 A · MODEL INPUT - the only files that feed the network",
     "Required. Replace these two (from the SAME pipeline run) to change what is scored.",
     [(["models/multimodal/multimodal.pt"], "Frozen weights + architecture + feature dims; loaded by `nets.load_model`", True, ""),
      (["data/processed/test.npz"], "Replay data: scaled phys / lab / text rows, modality mask, static vector, anchor index, labels "
                                    "(labels only used in the retrospective view)", True, "")]),
    ("🟠 B · RUNTIME CODE the app imports",
     "Required. Not data, but the app cannot start without them (plus PyTorch).",
     [(["src/nets.py"], "`WindowData` (reads test.npz), `MultimodalRiskNet`/`load_model`, occlusion helpers", True, ""),
      (["src/common.py"], "Constants (horizons, 24 h look-back, channel names, SEED, grid) and `apply_alarm_policy`", True, "")]),
    ("🟡 C · POLICY INPUT - frozen alarm operating point",
     "Read as data: τ, k and cooldown. Applied to the raw 6 h score.",
     [(["results/alarms/alarm_metrics.csv"], "Columns `tau`, `k`, `cooldown_min` + test alarm metrics", False,
       "fallback τ=0.30, k=2, 60 min (flagged in the UI)")]),
    ("🔵 D · DISPLAY-ONLY CONTEXT - the model never sees these",
     "Optional. They make the dashboard human-readable (raw units, lab table, notes, names); predictions are identical without them.",
     [(["deployment/demo_data/timeline_test.csv", "data/interim/timeline.csv"],
       "Raw hourly HR/RR/SpO₂/temp/SBP/DBP/MAP for the vitals cards and trend charts (slim test-only copy preferred on Cloud)",
       False, "vitals shown as z-scores instead of raw units"),
      (["data/interim/lab_clean.csv"], "Raw lab values (latest, age, count) in the laboratory panel", False, "lab table hidden"),
      (["data/interim/note_clean.csv"], "De-identified note snippets in the notes panel", False, "notes panel hidden"),
      (["data/processed/metadata.json"], "Lab-variable names for attribution labels; dims/splits on the Methodology tab", False,
       "generic lab_k names, Methodology facts omitted"),
      (["results/audit/locked_task.json"], "Endpoint definition text on the Methodology tab", False, "text omitted")]),
    ("🟣 E · PRE-COMPUTED RESULTS - Model Performance tab (nothing is re-evaluated)",
     "Optional. Each file feeds one table/chart; a missing file only blanks that block.",
     [(["results/evaluation/metrics.csv"], "AUPRC/AUROC per model & horizon, Brier/ECE", False, "table blank"),
      (["results/evaluation/bootstrap_ci_6h.csv"], "Stay-level bootstrap 95% CIs (6 h AUPRC)", False, "block blank"),
      (["results/evaluation/bootstrap_differences_6h.csv"], "Paired bootstrap differences", False, "block blank"),
      (["results/evaluation/ablations.csv"], "Modality-subset ablation", False, "block blank"),
      (["results/evaluation/missing_modality.csv"], "Reported held-out robustness (Robustness tab reference)", False, "block blank"),
      (["results/evaluation/stress_tests.csv"], "Noise / blackout stress tests", False, "block blank"),
      (["results/evaluation/lead_time_summary.csv"], "Lead-time summary", False, "block blank"),
      (["results/alarms/alarm_threshold_sweep.csv"], "Sensitivity-vs-alarm-burden curve (validation sweep)", False, "chart blank"),
      (["figures/calibration.png"], "Calibration figure", False, "image hidden"),
      (["figures/lead_time.png"], "Lead-time figure", False, "image hidden")]),
]

NOT_READ = [
    ("`data/raw/eicu_demo/*.csv.gz`", "Raw eICU tables - only steps 00-05 read them. **Never needed at runtime.**"),
    ("`src/00-05_*.py`, `06_build_features.py`, `src/features.py`", "Cohort, timeline, labels and feature construction that PRODUCE test.npz."),
    ("`data/interim/{cohort,split,labels,events,vital_clean}.csv`, `text_embeddings.npy`", "Intermediates between the steps above."),
    ("`data/processed/{train,validation}.npz`", "Training / model-selection tensors (same format as test.npz)."),
    ("`models/text_encoder.joblib`", "Frozen TF-IDF+SVD note encoder - already applied inside test.npz['text']."),
    ("`models/baseline/*`, `src/07-08_*.py`", "Baselines and training of the checkpoint."),
    ("`results/predictions/*.csv`, `results/evaluation/{calibrators,model_metadata,…}.csv`", "Offline prediction dumps / calibrators; the live view uses raw outputs."),
    ("`src/09-12_*.py`, `results/explanations/*`", "Offline evaluation, alarm sweep, explanation and bootstrap scripts that PRODUCE the CSVs in E."),
]

FLOW = ("UI choices (stay, hour, scenario)  = replay controls only, not clinical data\n"
        "        |\n"
        "        v\n"
        "[A] data/processed/test.npz --WindowData--> rows [a-23 ... a]   (phys | lab | text | mod | static)\n"
        "        |                                           |\n"
        "        |                       [A] models/multimodal/multimodal.pt  (+ [B] src/nets.py, src/common.py)\n"
        "        v                                           v\n"
        "                            p(1h), p(3h), p(6h)  --[C] alarm_metrics.csv: tau, k, cooldown--> alarm state\n"
        "\n"
        "In parallel, DISPLAY ONLY:  [D] timeline.csv / lab_clean.csv / note_clean.csv  ->  vitals cards, lab & note panels\n"
        "Static, PRE-COMPUTED:       [E] results/**/*.csv, figures/*.png                 ->  Model Performance tab")


def render_files():
    d = get_data()
    dm = d.dims
    meta = read_json("data/processed/metadata.json") or {}
    st.markdown("#### Data & files: what is input, what is support")
    st.markdown("**Two layers.** *UI selections* (stay, hour, scenario) only choose which rows to replay. The *model input* is the "
                "five tensors below, cut from `test.npz` as the 24 rows ending at the chosen hour. The dashboard never asks for "
                "clinical values and never reads raw eICU tables. Status below is checked live against this repository.")
    st.code(FLOW, language="text")

    for title, note, items in FILE_GROUPS:
        st.markdown(f"##### {title}")
        st.caption(note)
        rows = []
        for alts, role, required, degrade in items:
            found = next((a for a in alts if (ROOT / a).exists()), None)
            if found:
                status, size = "✅ found", _fmt_size(ROOT / found)
            else:
                status, size = ("❌ MISSING - app cannot run" if required else f"➖ absent → {degrade}"), "-"
            rows.append({"File": found or " | ".join(alts), "Role in the app": role, "Status": status, "Size": size})
        stretch(st.dataframe, pd.DataFrame(rows), hide_index=True)

    st.markdown("##### The five model tensors (all inside `test.npz`, exposed by `nets.WindowData`)")
    ln = [str(x) for x in (meta.get("lab_names") or [])]
    tens = pd.DataFrame([
        {"Tensor": "phys", "Window shape": f"{LOOKBACK} × {dm['d_p']}",
         "Per hourly row": f"{dm['d_p'] // 5} channels ({', '.join(PHYS_CH)}) × [z-mean, z-min, z-max, observed flag, hours-since-observed/24]; TRAIN-fit scaling"},
        {"Tensor": "lab", "Window shape": f"{LOOKBACK} × {dm['d_l']}",
         "Per hourly row": f"{dm['d_l'] // 5} lab variables{' (' + ', '.join(ln) + ')' if ln else ''} × [latest z, hours-since, 24 h slope, log-count, ever-measured]"},
        {"Tensor": "text", "Window shape": f"{LOOKBACK} × {dm['d_t']}",
         "Per hourly row": f"{dm['d_t'] - 2}-d frozen TF-IDF→SVD note embedding + [hours-since-note/72, log-count]; notes visible at documentation time (noteEnteredOffset)"},
        {"Tensor": "mod", "Window shape": f"{LOOKBACK} × {d.mod.shape[1]}",
         "Per hourly row": "availability mask [physiology observed this hour, any lab so far, new note this hour]"},
        {"Tensor": "static", "Window shape": f"{dm['d_s']}",
         "Per hourly row": f"{d.static.shape[1]} stored admission features (age, weight/height + missing flags, pre-ICU hours, sex, unit type) + log1p(hour)/6 appended at window time"},
    ])
    stretch(st.dataframe, tens, hide_index=True)
    st.caption(f"Replay set: {len(d.stay_ids)} test stays, {len(d):,} anchors. `test.npz` also holds the index arrays "
               "`stay_ids, row_start, n_rows, a_stay, a_hour, a_min` and labels `y, next_onset` "
               "(labels/onsets are shown only in the retrospective view, never fed to the model).")

    with st.expander("Files the app does NOT read at runtime (upstream / offline) - needed only to regenerate A, C, E"):
        st.markdown("\n".join(f"- {a} - {b}" for a, b in NOT_READ))

    st.markdown("##### Reproducing or scaling this deployment")
    st.markdown(
        "1. **Minimum runnable set = A + B** (+ C for the frozen alarm). D adds readability, E fills the performance tab.\n"
        "2. **`multimodal.pt` and `test.npz` must come from the same pipeline run** - dimensions are checked at startup and a mismatch stops the app.\n"
        "3. **New patients / a new cohort:** regenerate through `src/03 → 06`. The physiology, lab and static scalers are fitted on TRAIN inside "
        "`06_build_features.py` and are **not saved to disk** (only `models/text_encoder.joblib` is), so new stays can only be featurised "
        "consistently by re-running 06 on the same cohort/split (deterministic, SEED 42) or after persisting those scalers.\n"
        "4. **Streamlit Cloud:** commit A, B, C and E; commit `deployment/demo_data/timeline_test.csv` (slim, test stays only) plus the small "
        "`lab_clean.csv` / `note_clean.csv` for D. Nothing under `data/raw/` is needed.")


# ============================================================================ main
def main():
    st.title("The ICU Monitor Dilemma")
    st.markdown("##### Leakage-Controlled Multimodal Forecasting of ICU Physiological Deterioration")
    st.caption("Research Demonstration")
    st.warning("Research demonstration only. This prototype is not a clinical decision-support system and must not be used "
               "for patient care.")

    missing = [(p, w) for p, w in ((MODEL_PATH, "Model checkpoint not found"),
                                   (TEST_NPZ, "No deployment-ready demonstration data found")) if not p.exists()]
    if missing:
        for p, w in missing:
            st.error(f"{w}:\n`{p}`")
        st.info("Commit `models/multimodal/multimodal.pt` and `data/processed/test.npz` (+ `metadata.json`) to the repository.")
        st.stop()
    try:
        model, data = get_model(), get_data()
        if any(int(model.dims[k]) != int(data.dims[k]) for k in ("d_p", "d_l", "d_t", "d_s")):
            st.error(f"Checkpoint dimensions {dict(model.dims)} do not match the processed data {data.dims}. "
                     "Re-run the pipeline so that both come from the same run.")
            st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load model / data: `{type(exc).__name__}: {exc}`")
        st.stop()
    if len(stay_table()) == 0:
        st.error("The replay data contain no valid anchors.")
        st.stop()

    ctx = sidebar()
    tabs = st.tabs(["Live Forecast", "Robustness", "Model Performance", "Methodology", "Data & Files"])
    with tabs[0]:
        render_live(ctx)
    with tabs[1]:
        render_robustness(ctx)
    with tabs[2]:
        render_performance()
    with tabs[3]:
        render_methodology(ctx)
    with tabs[4]:
        render_files()

    if st.session_state.get("playing"):
        if ctx["pos"] >= ctx["n"] - 1:
            st.session_state["playing"] = False
            st.rerun()
        else:
            time.sleep(1.0 / ctx["speed"])
            st.session_state["_advance"] = True
            st.rerun()


main()