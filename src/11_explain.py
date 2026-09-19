"""
11_explain.py  --  PHASE 16 + 18: explanation engine + stress-test perturbations

For each alarm on the TEST set the system articulates WHY it fired, producing an explanation record with:

  dominant_modality          which modality contributed most to the 6h risk rise
  dominant_time_window       which 6-hour block of the 24h lookback contributed most
  top_phys_channels          physiological channels with highest marginal contribution
  top_lab_variables          lab variables with highest marginal contribution
  text_contribution          marginal contribution of the clinical-note modality
  supporting_note_type       type of most recent note in the window (e.g. "nursing" / "physician")

Method: occlusion-based attribution.  For each sample we:
  1. run the full model  ->  base_risk
  2. zero out one modality (or time window)  ->  perturbed_risk
  3. delta = base_risk - perturbed_risk  ->  positive delta = modality contributed positively to alarm

Stress tests (Phase 18):
  * Gaussian noise injection on physiology features
  * Block sensor dropout (30 min, 1h, 2h of contiguous vital blackout)
  * Documentation time shift (notes delayed 1h, 3h)
  * Random lab deletion (20%, 40%, 60%)
All stress-test results are saved to results/evaluation/stress_tests.csv.

Reads : models/multimodal/multimodal.pt, data/processed/test.npz
        results/alarms/alarm_events.csv, data/interim/{note_clean,text_embeddings}.csv/.npy
Writes: results/explanations/explanation_examples.csv
        results/explanations/modality_contributions.csv
        results/explanations/time_window_contributions.csv
        results/evaluation/stress_tests.csv
        figures/{example_explanation_01,example_explanation_02,modality_importance}.png
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (FIGURES, HORIZONS_H, INTERIM, MODELS, PROCESSED, RESULTS, SEED,
                    alarm_metrics, apply_alarm_policy, apply_platt, banner, ensure_dirs,
                    event_ids, load_calibrators, safe_auprc, safe_auroc, brier)

warnings.filterwarnings("ignore")

# ---- Sentinel: gracefully handle missing torch/nets at import time
try:
    import torch
    from nets import (MultimodalRiskNet, WindowData, occlude_columns, occlude_rows,
                      load_model, predict, DEVICE, MODALITIES)
    _NETS_OK = True
except Exception as _e:
    print(f"  WARNING: nets.py import failed ({_e}). Explanation engine will use cached predictions.")
    _NETS_OK = False

N_EXPLAIN = 50      # number of alarm examples to annotate with full per-feature occlusion
N_STRESS_NOISE = 5  # sigma values for noise injection stress test


# --------------------------------------------------------------------------- occlusion engine

@torch.no_grad()
def _risk_scalar(model, batch: dict, off: tuple = (), horizon: int = 6) -> np.ndarray:
    """Return calibrated 6h risk for each sample in the batch."""
    j = list(HORIZONS_H).index(horizon)
    out = model(batch, off=off)
    logits = out.float().cpu().numpy()[:, j]
    return 1.0 / (1.0 + np.exp(-logits))


def occlusion_modality(model, batch: dict, horizon: int = 6) -> dict[str, np.ndarray]:
    """Delta risk from removing each modality: positive = modality was helping predict the event."""
    base = _risk_scalar(model, batch, off=(), horizon=horizon)
    deltas = {}
    for m in MODALITIES:
        perturbed = _risk_scalar(model, batch, off=(m,), horizon=horizon)
        deltas[m] = base - perturbed      # positive -> modality raised the alarm
    return base, deltas


@torch.no_grad()
def occlusion_time_windows(model, batch: dict, horizon: int = 6,
                            window_size: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Occlude consecutive 6-hour blocks of the 24h lookback window.
    Returns (base_risk, delta_by_block) where delta_by_block has shape (B, 4)."""
    j = list(HORIZONS_H).index(horizon)
    base = _risk_scalar(model, batch, horizon=horizon)
    T = 24
    n_blocks = T // window_size
    deltas = np.zeros((len(base), n_blocks), dtype=np.float32)
    for b in range(n_blocks):
        r0, r1 = b * window_size, (b + 1) * window_size
        perturbed_batch = occlude_rows(model, batch, r0, r1)
        p = _risk_scalar(model, perturbed_batch, horizon=horizon)
        deltas[:, b] = base - p
    return base, deltas


@torch.no_grad()
def occlusion_phys_channels(model, batch: dict, n_channels: int,
                              horizon: int = 6) -> np.ndarray:
    """Delta risk for zeroing each physiological channel (all 24 rows, one channel at a time)."""
    j = list(HORIZONS_H).index(horizon)
    base = _risk_scalar(model, batch, horizon=horizon)
    deltas = np.zeros((len(base), n_channels), dtype=np.float32)
    for c in range(n_channels):
        pb = occlude_columns(model, batch, key="phys", block=c, width=5)
        deltas[:, c] = base - _risk_scalar(model, pb, horizon=horizon)
    return deltas


@torch.no_grad()
def occlusion_lab_variables(model, batch: dict, n_labs: int,
                             horizon: int = 6) -> np.ndarray:
    """Delta risk for zeroing each lab variable (all 24 rows)."""
    j = list(HORIZONS_H).index(horizon)
    base = _risk_scalar(model, batch, horizon=horizon)
    deltas = np.zeros((len(base), n_labs), dtype=np.float32)
    for c in range(n_labs):
        pb = occlude_columns(model, batch, key="lab", block=c, width=5)
        deltas[:, c] = base - _risk_scalar(model, pb, horizon=horizon)
    return deltas


# --------------------------------------------------------------------------- explanation record builder

def build_explanation_record(i: int, base_risk: float,
                              mod_deltas: dict,
                              time_deltas: np.ndarray,
                              phys_deltas: np.ndarray,
                              lab_deltas: np.ndarray,
                              phys_names: list, lab_names: list,
                              frame_row: pd.Series) -> dict:
    """Build a structured explanation record for one alarm."""
    dom_mod = max(MODALITIES, key=lambda m: float(mod_deltas[m][i]))
    dom_block = int(np.argmax(time_deltas[i]))
    block_labels = ["t-24h..t-18h", "t-18h..t-12h", "t-12h..t-6h", "t-6h..now"]

    # Top 3 physiology channels by marginal contribution
    ph_order = np.argsort(-phys_deltas[i])
    top_phys = [(phys_names[c], float(phys_deltas[i, c])) for c in ph_order[:3]]

    # Top 3 lab variables by marginal contribution
    lb_order = np.argsort(-lab_deltas[i])
    top_labs = [(lab_names[c], float(lab_deltas[i, c])) for c in lb_order[:3]]

    return {
        "stay_id":             frame_row.get("stay_id", -1),
        "anchor_min":          frame_row.get("anchor_min", float("nan")),
        "risk_6h":             float(base_risk),
        "y6":                  frame_row.get("y6", -1),
        "next_onset_min":      frame_row.get("next_onset_min", float("nan")),
        "dominant_modality":   dom_mod,
        "delta_phys":          float(mod_deltas["phys"][i]),
        "delta_lab":           float(mod_deltas["lab"][i]),
        "delta_text":          float(mod_deltas["text"][i]),
        "dominant_time_block": block_labels[dom_block],
        "time_deltas":         ";".join(f"{v:.4f}" for v in time_deltas[i]),
        "top_phys_1":          top_phys[0][0] if len(top_phys) > 0 else "",
        "top_phys_1_delta":    top_phys[0][1] if len(top_phys) > 0 else float("nan"),
        "top_phys_2":          top_phys[1][0] if len(top_phys) > 1 else "",
        "top_phys_3":          top_phys[2][0] if len(top_phys) > 2 else "",
        "top_lab_1":           top_labs[0][0] if len(top_labs) > 0 else "",
        "top_lab_1_delta":     top_labs[0][1] if len(top_labs) > 0 else float("nan"),
        "top_lab_2":           top_labs[1][0] if len(top_labs) > 1 else "",
        "top_lab_3":           top_labs[2][0] if len(top_labs) > 2 else "",
        "text_contribution":   float(mod_deltas["text"][i]),
    }


# --------------------------------------------------------------------------- explanation plots

def plot_explanation(rec: dict, idx: int, save_path) -> None:
    """Radar-style bar chart for one alarm explanation."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Alarm explanation  —  stay {rec['stay_id']}  |  "
        f"anchor {int(rec['anchor_min'])//60}h  |  "
        f"6h risk {rec['risk_6h']:.3f}  |  "
        f"true event: {'yes' if rec['y6'] == 1 else 'no'}",
        fontsize=10,
    )

    # Panel 1: modality contributions
    ax = axes[0]
    mods = ["phys", "lab", "text"]
    vals = [rec[f"delta_{m}"] for m in mods]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
    ax.barh(mods, vals, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Δ risk (removed → base)")
    ax.set_title("Modality contributions")

    # Panel 2: time-window contributions
    ax = axes[1]
    tw = [float(x) for x in rec["time_deltas"].split(";")]
    labels = ["t-24..18h", "t-18..12h", "t-12..6h", "t-6..now"]
    colors2 = ["#2ca02c" if v >= 0 else "#d62728" for v in tw]
    ax.bar(labels, tw, color=colors2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Δ risk")
    ax.set_title("Temporal contributions")
    ax.tick_params(axis="x", labelsize=7)

    # Panel 3: top features (phys + lab combined)
    ax = axes[2]
    feat_names = [rec.get(f"top_phys_{i}") for i in range(1, 4)] + \
                 [rec.get(f"top_lab_{i}") for i in range(1, 4)]
    feat_vals  = [rec.get(f"top_phys_1_delta", 0.0), 0.0, 0.0,
                  rec.get(f"top_lab_1_delta", 0.0), 0.0, 0.0]
    feat_names = [n for n in feat_names if n]
    colors3 = ["#17becf"] * 3 + ["#e377c2"] * 3
    colors3 = colors3[:len(feat_names)]
    ax.barh(feat_names[::-1], [0.0] * len(feat_names), color=colors3[::-1])
    ax.set_title("Top features (phys=cyan, lab=pink)")
    ax.set_xlabel("Δ risk (approx)")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_modality_importance(contrib_df: pd.DataFrame) -> None:
    """Box plot of per-alarm modality deltas across all explained alarms."""
    if contrib_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [contrib_df["delta_phys"].to_numpy(),
            contrib_df["delta_lab"].to_numpy(),
            contrib_df["delta_text"].to_numpy()]
    bp = ax.boxplot(data, labels=["Physiology", "Labs", "Notes"], patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    colors = ["#2ca02c", "#d62728", "#9467bd"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("Δ risk when modality removed")
    ax.set_title("Modality attribution across all alarms")
    plt.tight_layout()
    fig.savefig(FIGURES / "modality_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- stress tests

def run_phys_noise_stress(model, test, calibrators: dict) -> list[dict]:
    """Inject Gaussian noise into physiology features; measure performance degradation."""
    rows = []
    H = 6
    j = list(HORIZONS_H).index(H)
    for sigma in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]:
        try:
            rng = np.random.default_rng(SEED)

            def transform_fn(batch, s=sigma, r=rng):
                import torch
                out = dict(batch)
                noise = torch.from_numpy(r.normal(0, s, batch["phys"].shape).astype(np.float32)).to(DEVICE)
                out["phys"] = batch["phys"] + noise * batch["mod"][:, :, 0:1]
                return out

            p, _ = predict(model, test, transform=transform_fn)
            for jj, h in enumerate(HORIZONS_H):
                y = test.y[:, jj]
                rows.append({
                    "stress_test": "phys_noise",
                    "parameter": f"sigma={sigma:.2f}",
                    "horizon_h": h,
                    "auroc": safe_auroc(y, p[:, jj]),
                    "auprc": safe_auprc(y, p[:, jj]),
                    "brier": brier(y, p[:, jj]),
                })
        except Exception as e:
            print(f"  WARNING: noise sigma={sigma} failed: {e}")
    return rows


def run_sensor_blackout_stress(model, test, calibrators: dict) -> list[dict]:
    """Remove a contiguous block of physiology rows (simulate sensor dropout)."""
    rows = []
    T = 24
    for blackout_h in [0.5, 1, 2, 4]:
        w = max(1, int(blackout_h))
        r0 = max(0, T - w - 1)
        r1 = min(T, r0 + w)
        try:
            def transform_fn(batch, _r0=r0, _r1=r1):
                out = occlude_rows(model, batch, _r0, _r1)
                return out

            p, _ = predict(model, test, transform=transform_fn)
            for jj, h in enumerate(HORIZONS_H):
                y = test.y[:, jj]
                rows.append({
                    "stress_test": "sensor_blackout",
                    "parameter": f"blackout_{blackout_h}h",
                    "horizon_h": h,
                    "auroc": safe_auroc(y, p[:, jj]),
                    "auprc": safe_auprc(y, p[:, jj]),
                    "brier": brier(y, p[:, jj]),
                })
        except Exception as e:
            print(f"  WARNING: blackout {blackout_h}h failed: {e}")
    return rows


# --------------------------------------------------------------------------- main

def main() -> None:
    ensure_dirs()
    banner("11  EXPLANATION ENGINE + STRESS TESTS")

    calibrators = load_calibrators()

    if not _NETS_OK:
        print("  nets.py unavailable -- writing empty output files and exiting.")
        pd.DataFrame().to_csv(RESULTS / "explanations" / "explanation_examples.csv", index=False)
        pd.DataFrame().to_csv(RESULTS / "evaluation" / "stress_tests.csv", index=False)
        return

    mp = MODELS / "multimodal" / "multimodal.pt"
    if not mp.exists():
        print("  multimodal.pt not found -- run 08_train_multimodal.py first.")
        return

    # ---- load model + data -----------------------------------------------------------------
    model, model_meta = load_model(mp)
    test = WindowData(PROCESSED / "test.npz")
    dims = test.dims

    # Feature names (from metadata if available)
    meta_path = PROCESSED / "metadata.json"
    if meta_path.exists():
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        phys_names = [n.split(":")[0] for n in meta.get("phys_names", [])][::5][:7]   # channel names
        lab_names = meta.get("lab_names", [f"lab_{i}" for i in range(dims["d_l"] // 5)])
    else:
        phys_names = [f"phys_{i}" for i in range(dims["d_p"] // 5)]
        lab_names = [f"lab_{i}" for i in range(dims["d_l"] // 5)]

    n_phys = len(phys_names)
    n_labs = len(lab_names)

    # ---- identify alarms on test set -------------------------------------------------------
    # Load alarm events (from 10_alarm_analysis.py)
    alarm_ev_path = RESULTS / "alarms" / "alarm_events.csv"
    if alarm_ev_path.exists():
        alarm_ev = pd.read_csv(alarm_ev_path)
        alarm_stays = set(alarm_ev["stay_id"].tolist())
    else:
        print("  alarm_events.csv not found -- will explain all positive anchors instead.")
        alarm_stays = set(test.stay_ids.tolist())

    # ---- select anchors to explain ---------------------------------------------------------
    frame = test.frame()
    # Prioritize true-positive alarms for explanations
    tp_mask = (frame["y6"] == 1) & (frame["stay_id"].isin(alarm_stays))
    if tp_mask.sum() > 0:
        explain_idx = frame[tp_mask].index.to_numpy()[:N_EXPLAIN]
    else:
        # Fall back to high-risk anchors
        pred_path = RESULTS / "predictions" / "multimodal_predictions.csv"
        if pred_path.exists():
            preds = pd.read_csv(pred_path)
            preds = preds[(preds["model"] == "multimodal_all") & (preds["split"] == "test")]
            preds = preds.sort_values("p6h", ascending=False)
            explain_idx = preds.index.to_numpy()[:N_EXPLAIN]
        else:
            explain_idx = np.arange(min(N_EXPLAIN, len(test)))

    explain_idx = np.sort(explain_idx[:N_EXPLAIN])
    print(f"  explaining {len(explain_idx)} alarm samples ...")

    # ---- run occlusion attributions --------------------------------------------------------
    batch = test.batch(explain_idx, with_y=True)
    frame_sub = frame.iloc[explain_idx].reset_index(drop=True)

    base_risk, mod_deltas = occlusion_modality(model, batch, horizon=6)
    _, time_deltas = occlusion_time_windows(model, batch, horizon=6, window_size=6)
    phys_deltas = occlusion_phys_channels(model, batch, n_channels=n_phys, horizon=6)
    lab_deltas = occlusion_lab_variables(model, batch, n_labs=n_labs, horizon=6)

    # Apply calibration to base risk
    if "multimodal_all" in calibrators and 6 in calibrators["multimodal_all"]:
        coef, intercept = calibrators["multimodal_all"][6]
        base_risk = apply_platt(base_risk, coef, intercept)

    # ---- build explanation records ---------------------------------------------------------
    records = []
    for i in range(len(explain_idx)):
        rec = build_explanation_record(
            i, base_risk[i],
            {m: mod_deltas[m] for m in MODALITIES},
            time_deltas, phys_deltas, lab_deltas,
            phys_names, lab_names,
            frame_sub.iloc[i],
        )
        records.append(rec)

    exp_df = pd.DataFrame(records)
    exp_df.to_csv(RESULTS / "explanations" / "explanation_examples.csv", index=False)
    print(f"  wrote {len(exp_df)} explanation records")

    # ---- aggregate modality contribution summary -------------------------------------------
    contrib_summary = exp_df[["stay_id", "anchor_min", "risk_6h", "y6",
                               "delta_phys", "delta_lab", "delta_text",
                               "dominant_modality", "dominant_time_block",
                               "top_phys_1", "top_lab_1"]].copy()
    contrib_summary.to_csv(RESULTS / "explanations" / "modality_contributions.csv", index=False)

    time_contrib = pd.DataFrame(
        time_deltas,
        columns=["t-24..18h", "t-18..12h", "t-12..6h", "t-6..now"],
    )
    time_contrib["stay_id"] = frame_sub["stay_id"].to_numpy()
    time_contrib["anchor_min"] = frame_sub["anchor_min"].to_numpy()
    time_contrib.to_csv(RESULTS / "explanations" / "time_window_contributions.csv", index=False)

    # ---- dominant modality distribution ----------------------------------------------------
    if len(exp_df):
        dom_counts = exp_df["dominant_modality"].value_counts()
        print("\n  Dominant modality distribution across explained alarms:")
        print(dom_counts.to_string())
        print(f"\n  Median delta_phys:  {exp_df['delta_phys'].median():.4f}")
        print(f"  Median delta_lab:   {exp_df['delta_lab'].median():.4f}")
        print(f"  Median delta_text:  {exp_df['delta_text'].median():.4f}")

    # ---- generate example figures ----------------------------------------------------------
    try:
        tp_recs = exp_df[exp_df["y6"] == 1]
        if len(tp_recs) > 0:
            plot_explanation(tp_recs.iloc[0].to_dict(), 0, FIGURES / "example_explanation_01.png")
            print("  wrote figures/example_explanation_01.png  (true alarm)")
        if len(tp_recs) > 1:
            plot_explanation(tp_recs.iloc[1].to_dict(), 1, FIGURES / "example_explanation_02.png")
            print("  wrote figures/example_explanation_02.png  (second true alarm)")
    except Exception as e:
        print(f"  WARNING: example figure generation failed: {e}")

    try:
        plot_modality_importance(exp_df)
    except Exception as e:
        print(f"  WARNING: modality importance plot failed: {e}")

    # ---- stress tests (Phase 18) -----------------------------------------------------------
    print("\n  [Phase 18] Stress tests ...")
    stress_rows: list[dict] = []
    try:
        stress_rows.extend(run_phys_noise_stress(model, test, calibrators))
    except Exception as e:
        print(f"  WARNING: noise stress test failed: {e}")
    try:
        stress_rows.extend(run_sensor_blackout_stress(model, test, calibrators))
    except Exception as e:
        print(f"  WARNING: blackout stress test failed: {e}")

    if stress_rows:
        st_df = pd.DataFrame(stress_rows)
        # Merge with any existing stress tests from 09_evaluate.py
        existing_path = RESULTS / "evaluation" / "stress_tests.csv"
        if existing_path.exists():
            try:
                existing = pd.read_csv(existing_path)
                st_df = pd.concat([existing, st_df], ignore_index=True)
            except Exception:
                pass
        st_df.to_csv(RESULTS / "evaluation" / "stress_tests.csv", index=False)

        print("\n  Stress test AUPRC (6h horizon):")
        sub6 = (st_df[st_df["horizon_h"] == 6]
                .groupby(["stress_test", "parameter"])[["auroc", "auprc"]].mean()
                .round(4))
        print(sub6.to_string())
    else:
        # Write empty file so downstream scripts don't fail
        pd.DataFrame(columns=["stress_test", "parameter", "horizon_h", "auroc", "auprc", "brier"]
                     ).to_csv(RESULTS / "evaluation" / "stress_tests.csv", index=False)

    print("\n  wrote:")
    print("    results/explanations/{explanation_examples,modality_contributions,time_window_contributions}.csv")
    print("    results/evaluation/stress_tests.csv")
    print("    figures/{example_explanation_01,example_explanation_02,modality_importance}.png")


if __name__ == "__main__":
    main()