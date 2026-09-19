"""
10_alarm_analysis.py  --  PHASE 15: alarm controller + PHASE 14: missing-modality robustness

Part A: Alarm Controller
  At each hourly anchor, the model outputs a 6h risk score p(t).  A clinical alarm fires when:
    1. p(t) > τ  for k consecutive anchors  (persistence filter; avoids flicker)
    2. at least C minutes elapsed since the previous alarm in the same stay  (cooldown)

  τ is selected on VALIDATION (never on test).  Then frozen and applied once to test.

  Metrics reported:
    * alarms per patient-day  (burden metric that clinicians actually care about)
    * false alarms per patient-day
    * alarm precision
    * event sensitivity (fraction of true deterioration events that had >= 1 alarm in the 6h window)
    * warning time distribution (onset_time - first_true_alarm_time)

Part B: Missing-Modality Robustness
  The same trained multimodal model is evaluated with one (then two) modalities zeroed out.
  This tests Q4 from the whitepaper: "withstand the chaos of missing sensors and erratic lab orders."

  Perturbations tested:
    * drop_lab_20pct   -- randomly drop 20% of lab observations per stay
    * drop_lab_40pct
    * drop_lab_60pct
    * off_lab          -- remove entire lab modality
    * off_text         -- remove entire text modality
    * off_phys         -- remove entire physiology modality
    * delayed_notes_1h -- shift note availability forward by 60 min (tests documentation-timing dependence)
    * delayed_notes_3h -- shift note availability forward by 180 min

Reads : results/predictions/multimodal_predictions.csv,  results/evaluation/calibrators.csv
        models/multimodal/multimodal.pt,  data/processed/test.npz
        data/interim/{note_clean,text_embeddings}.csv/.npy
Writes: results/alarms/alarm_metrics.csv, results/alarms/alarm_events.csv,
        results/alarms/alarm_threshold_sweep.csv
        results/evaluation/missing_modality.csv
        figures/{alarm_tradeoff,missing_modality}.png
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


# --------------------------------------------------------------------------- loading helpers

def load_multimodal_preds(split: str = "test") -> pd.DataFrame:
    p = RESULTS / "predictions" / "multimodal_predictions.csv"
    if not p.exists():
        raise FileNotFoundError("multimodal_predictions.csv missing -- run 08_train_multimodal.py first")
    df = pd.read_csv(p)
    return df[(df["model"] == "multimodal_all") & (df["split"] == split)].reset_index(drop=True)


def load_multimodal_val_preds() -> pd.DataFrame:
    return load_multimodal_preds("validation")


def load_model_safe():
    try:
        from nets import load_model, WindowData, predict
        from features import Layout, text_features
        return load_model, WindowData, predict, Layout, text_features, True
    except Exception as e:
        print(f"  WARNING: could not load nets module ({e}). Stress tests will use cached predictions.")
        return None, None, None, None, None, False


# --------------------------------------------------------------------------- Part A: alarm controller

def threshold_sweep(val_df: pd.DataFrame, calibrators: dict,
                    k: int = 2, cooldown_min: float = 60.0) -> pd.DataFrame:
    """Sweep τ on VALIDATION predictions; return sensitivity / burden table."""
    H = 6   # use 6h horizon as the operational risk score
    j = list(HORIZONS_H).index(H)
    p = val_df[f"p{H}h"].to_numpy()

    # Apply Platt calibration if available
    if "multimodal_all" in calibrators and H in calibrators["multimodal_all"]:
        coef, intercept = calibrators["multimodal_all"][H]
        p = apply_platt(p, coef, intercept)

    y6 = val_df["y6"].to_numpy()
    stay = val_df["stay_id"].to_numpy()
    t = val_df["anchor_min"].to_numpy()
    next_onset = val_df["next_onset_min"].to_numpy()
    ev = event_ids(stay, y6, next_onset)

    rows = []
    for tau in np.arange(0.05, 0.96, 0.05):
        above = p > tau
        alarms = apply_alarm_policy(stay, t, above, k=k, cooldown_min=cooldown_min)
        m = alarm_metrics(t, alarms, ev, next_onset)
        rows.append({"tau": round(float(tau), 3), **m})
    return pd.DataFrame(rows)


def select_threshold(sweep_df: pd.DataFrame, target_sens: float = 0.70) -> float:
    """Choose highest τ that still achieves target event_sensitivity (or best if none reach it)."""
    ok = sweep_df[sweep_df["event_sensitivity"] >= target_sens]
    if len(ok) == 0:
        # fall back: highest sensitivity point
        best = sweep_df.sort_values("event_sensitivity", ascending=False).iloc[0]
    else:
        best = ok.sort_values("tau", ascending=False).iloc[0]
    return float(best["tau"])


def run_alarm_analysis(test_df: pd.DataFrame, calibrators: dict,
                       tau: float, k: int = 2, cooldown_min: float = 60.0) -> tuple:
    """Apply alarm policy with frozen τ to test predictions."""
    H = 6
    p = test_df[f"p{H}h"].to_numpy()
    if "multimodal_all" in calibrators and H in calibrators["multimodal_all"]:
        coef, intercept = calibrators["multimodal_all"][H]
        p = apply_platt(p, coef, intercept)

    y6 = test_df["y6"].to_numpy()
    stay = test_df["stay_id"].to_numpy()
    t = test_df["anchor_min"].to_numpy()
    next_onset = test_df["next_onset_min"].to_numpy()
    ev = event_ids(stay, y6, next_onset)
    above = p > tau
    alarms = apply_alarm_policy(stay, t, above, k=k, cooldown_min=cooldown_min)

    metrics = alarm_metrics(t, alarms, ev, next_onset)
    metrics["tau"] = tau
    metrics["k"] = k
    metrics["cooldown_min"] = cooldown_min
    metrics["model"] = "multimodal_all"
    metrics["split"] = "test"

    # per-event records
    ev_rows = []
    for i, (s, ti, al, ev_id) in enumerate(zip(stay.tolist(), t.tolist(), alarms.tolist(), ev.tolist())):
        if al and ev_id >= 0:
            next_o = float(next_onset[i])
            lead = (next_o - ti) / 60.0 if np.isfinite(next_o) else float("nan")
            ev_rows.append({"stay_id": int(s), "anchor_min": float(ti),
                            "event_id": int(ev_id), "lead_h": lead, "split": "test"})
    return metrics, pd.DataFrame(ev_rows), alarms, ev


def plot_alarm_tradeoff(sweep_df: pd.DataFrame, tau_star: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.plot(sweep_df["tau"], sweep_df["event_sensitivity"], "b-o", markersize=3)
    ax.axvline(tau_star, color="red", ls="--", label=f"τ*={tau_star:.2f}")
    ax.set_xlabel("Threshold τ"); ax.set_ylabel("Event sensitivity")
    ax.set_title("Sensitivity vs Threshold")
    ax.legend()

    ax = axes[1]
    ax.plot(sweep_df["tau"], sweep_df["false_alarms_per_patient_day"], "r-o", markersize=3)
    ax.axvline(tau_star, color="red", ls="--")
    ax.set_xlabel("Threshold τ"); ax.set_ylabel("False alarms / patient-day")
    ax.set_title("False Alarm Burden vs Threshold")

    ax = axes[2]
    ax.plot(sweep_df["false_alarms_per_patient_day"], sweep_df["event_sensitivity"], "k-o", markersize=3)
    for _, row in sweep_df.iloc[::3].iterrows():
        ax.annotate(f"{row['tau']:.2f}", (row["false_alarms_per_patient_day"], row["event_sensitivity"]),
                    fontsize=6)
    ax.set_xlabel("False alarms / patient-day"); ax.set_ylabel("Event sensitivity")
    ax.set_title("Alarm ROC Curve")

    plt.tight_layout()
    fig.savefig(FIGURES / "alarm_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- Part B: missing-modality robustness

def compute_missing_robustness(model, test, calibrators: dict) -> list[dict]:
    """Evaluate model performance with each modality progressively dropped."""
    from nets import predict, MODALITIES
    rows = []
    H = 6
    j = list(HORIZONS_H).index(H)

    perturbations = [
        ("full_model",          ()),
        ("off_lab",             ("lab",)),
        ("off_text",            ("text",)),
        ("off_phys",            ("phys",)),
        ("off_lab+text",        ("lab", "text")),
        ("off_phys+text",       ("phys", "text")),
        ("off_phys+lab",        ("phys", "lab")),
    ]

    for label, off in perturbations:
        try:
            p, _ = predict(model, test, off=off)
            if "multimodal_all" in calibrators and H in calibrators["multimodal_all"]:
                coef, intercept = calibrators["multimodal_all"][H]
                p[:, j] = apply_platt(p[:, j], coef, intercept)
            for jj, h in enumerate(HORIZONS_H):
                y = test.y[:, jj]
                rows.append({
                    "perturbation": label,
                    "modalities_off": ",".join(off) if off else "none",
                    "horizon_h": h,
                    "auroc": safe_auroc(y, p[:, jj]),
                    "auprc": safe_auprc(y, p[:, jj]),
                    "brier": brier(y, p[:, jj]),
                })
        except Exception as e:
            print(f"  WARNING: perturbation {label} failed: {e}")
    return rows


def plot_missing_modality(mm_df: pd.DataFrame) -> None:
    if mm_df.empty:
        return
    sub = mm_df[mm_df["horizon_h"] == 6].copy()
    sub = sub.sort_values("auprc", ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1f77b4" if r["perturbation"] == "full_model" else "#d62728" for _, r in sub.iterrows()]
    ax.barh(sub["perturbation"], sub["auprc"], color=colors)
    ax.set_xlabel("AUPRC (6h horizon)")
    ax.set_title("Missing-Modality Robustness")
    plt.tight_layout()
    fig.savefig(FIGURES / "missing_modality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_lab_dropout_stress(model, test, calibrators: dict) -> list[dict]:
    """Drop random fractions of lab observations to test degradation."""
    from nets import predict, DEVICE
    import torch
    rows = []
    H = 6
    j = list(HORIZONS_H).index(H)

    for drop_frac in (0.20, 0.40, 0.60):
        try:
            rng = np.random.default_rng(SEED)

            # Override lab features with randomly zeroed rows
            def transform_fn(batch):
                out = dict(batch)
                mask = torch.from_numpy(rng.random(batch["lab"].shape[:2]).astype(np.float32) >= drop_frac
                                        ).unsqueeze(-1).to(DEVICE)
                # Zero out lab and set ever-measured flag to 0 for dropped rows
                out["lab"] = batch["lab"] * mask
                m = out["mod"].clone()
                m[:, :, 1] = m[:, :, 1] * mask.squeeze(-1)
                out["mod"] = m
                return out

            p, _ = predict(model, test, transform=transform_fn)
            for jj, h in enumerate(HORIZONS_H):
                y = test.y[:, jj]
                rows.append({
                    "perturbation": f"lab_dropout_{int(drop_frac*100)}pct",
                    "modalities_off": f"lab_random_{int(drop_frac*100)}pct",
                    "horizon_h": h,
                    "auroc": safe_auroc(y, p[:, jj]),
                    "auprc": safe_auprc(y, p[:, jj]),
                    "brier": brier(y, p[:, jj]),
                })
        except Exception as e:
            print(f"  WARNING: lab dropout {drop_frac:.0%} failed: {e}")
    return rows


def run_note_delay_stress(model, test_data, notes: pd.DataFrame,
                          emb: np.ndarray, calibrators: dict) -> list[dict]:
    """Artificially delay note availability by delta_min and re-build text features."""
    from nets import WindowData, predict
    from features import Layout, text_features
    rows = []
    H = 6
    j = list(HORIZONS_H).index(H)

    layout = Layout(test_data.stay_ids, test_data.n_rows)

    for delay_min in (60, 180):
        try:
            tf, _ = text_features(layout, notes, emb, delay_min=float(delay_min))
            # Replace text in the test WindowData
            # row mask for test stays
            arr = dict(test_data.a)
            arr["text"] = tf  # same shape as test_data.text
            modified = WindowData(arrays=arr)
            p, _ = predict(model, modified)
            for jj, h in enumerate(HORIZONS_H):
                y = test_data.y[:, jj]
                rows.append({
                    "perturbation": f"note_delay_{delay_min}min",
                    "modalities_off": f"text_delayed_{delay_min}min",
                    "horizon_h": h,
                    "auroc": safe_auroc(y, p[:, jj]),
                    "auprc": safe_auprc(y, p[:, jj]),
                    "brier": brier(y, p[:, jj]),
                })
        except Exception as e:
            print(f"  WARNING: note delay {delay_min} min failed: {e}")
    return rows


# --------------------------------------------------------------------------- main

def main() -> None:
    ensure_dirs()
    banner("10  ALARM ANALYSIS + MISSING-MODALITY ROBUSTNESS")

    calibrators = load_calibrators()

    # ---- Part A: Alarm Controller ----------------------------------------------------------
    print("\n  [Part A]  Alarm controller")

    # Threshold selection on VALIDATION
    try:
        val_df = load_multimodal_val_preds()
        sweep_df = threshold_sweep(val_df, calibrators, k=2, cooldown_min=60.0)
        sweep_df.to_csv(RESULTS / "alarms" / "alarm_threshold_sweep.csv", index=False)
        tau_star = select_threshold(sweep_df, target_sens=0.70)
        print(f"  threshold selected on validation:  τ* = {tau_star:.3f}")
    except Exception as e:
        print(f"  WARNING: threshold sweep failed ({e}). Using τ = 0.30 as fallback.")
        sweep_df = pd.DataFrame()
        tau_star = 0.30

    # Apply frozen threshold to TEST
    try:
        test_df = load_multimodal_preds("test")
        am, ev_df, alarms, ev = run_alarm_analysis(test_df, calibrators, tau=tau_star)
        pd.DataFrame([am]).to_csv(RESULTS / "alarms" / "alarm_metrics.csv", index=False)
        ev_df.to_csv(RESULTS / "alarms" / "alarm_events.csv", index=False)

        print(f"\n  Alarm metrics on TEST (τ={tau_star:.3f}, k=2, cooldown=60min):")
        for k, v in am.items():
            if isinstance(v, float):
                print(f"    {k:<35} {v:.4f}")
            else:
                print(f"    {k:<35} {v}")
    except Exception as e:
        print(f"  WARNING: alarm analysis on test failed: {e}")
        am = {}

    if not sweep_df.empty:
        try:
            plot_alarm_tradeoff(sweep_df, tau_star)
        except Exception as e:
            print(f"  WARNING: alarm tradeoff plot failed: {e}")

    # ---- Part B: Missing-Modality Robustness -----------------------------------------------
    print("\n  [Part B]  Missing-modality robustness")
    load_model_fn, WindowDataCls, predict_fn, Layout, text_features_fn, nets_ok = load_model_safe()

    mm_rows: list[dict] = []
    if nets_ok:
        mp = MODELS / "multimodal" / "multimodal.pt"
        if mp.exists():
            try:
                model, _ = load_model_fn(mp)
                test = WindowDataCls(PROCESSED / "test.npz")

                mm_rows.extend(compute_missing_robustness(model, test, calibrators))
                mm_rows.extend(run_lab_dropout_stress(model, test, calibrators))

                # Note-delay stress test (needs raw note table)
                note_path = INTERIM / "note_clean.csv"
                emb_path = INTERIM / "text_embeddings.npy"
                if note_path.exists() and emb_path.exists():
                    notes = pd.read_csv(note_path)
                    notes = notes[notes["stay_id"].isin(set(test.stay_ids.tolist()))]
                    emb = np.load(emb_path)
                    # Align emb rows with the loaded note subset
                    all_notes = pd.read_csv(note_path)
                    note_mask = all_notes["stay_id"].isin(set(test.stay_ids.tolist()))
                    emb_sub = emb[note_mask.to_numpy()]
                    mm_rows.extend(run_note_delay_stress(model, test, notes.reset_index(drop=True),
                                                         emb_sub, calibrators))
            except Exception as e:
                print(f"  WARNING: model-based robustness tests failed: {e}")
        else:
            print("  WARNING: multimodal.pt not found -- skipping model-based robustness tests")
    else:
        # Fall back to cached ablation predictions
        abl_path = RESULTS / "predictions" / "ablation_predictions.csv"
        if abl_path.exists():
            abl = pd.read_csv(abl_path)
            for name, g in abl.groupby("model"):
                label = name.replace("ablation_", "off_")
                for h in HORIZONS_H:
                    sub = g[(g["split"] == "test") & (g["horizon_h"] == h)] if "horizon_h" in g else \
                          g[g["split"] == "test"]
                    if len(sub) == 0:
                        continue
                    for jj, hh in enumerate(HORIZONS_H):
                        y = sub[f"y{hh}"].to_numpy()
                        p = sub[f"p{hh}h"].to_numpy()
                        mm_rows.append({
                            "perturbation": label, "modalities_off": name,
                            "horizon_h": hh,
                            "auroc": safe_auroc(y, p),
                            "auprc": safe_auprc(y, p),
                            "brier": brier(y, p),
                        })

    if mm_rows:
        mm_df = pd.DataFrame(mm_rows)
        mm_df.to_csv(RESULTS / "evaluation" / "missing_modality.csv", index=False)
        try:
            plot_missing_modality(mm_df)
        except Exception as e:
            print(f"  WARNING: missing modality plot failed: {e}")

        print("\n  Missing-modality robustness (AUPRC, 6h horizon):")
        sub6 = mm_df[mm_df["horizon_h"] == 6][["perturbation", "auprc"]].sort_values("auprc", ascending=False)
        print(sub6.to_string(index=False))

    print("\n  wrote:")
    print("    results/alarms/{alarm_metrics,alarm_events,alarm_threshold_sweep}.csv")
    print("    results/evaluation/missing_modality.csv")
    print("    figures/{alarm_tradeoff,missing_modality}.png")


if __name__ == "__main__":
    main()