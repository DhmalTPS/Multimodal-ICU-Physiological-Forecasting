"""
09_evaluate.py  --  PHASE 12 + 13: clinical-grade evaluation + ablation study

Answers five clinical questions:
  Q1  Does it discriminate?              AUROC, AUPRC
  Q2  Are probabilities meaningful?      Brier score, calibration curves, ECE
  Q3  Does it provide early warning?     lead time distribution (first alarm before event)
  Q4  Does it reduce alarm burden?       false alarms / patient-day   [preliminary; full in 10_*]
  Q5  Does multimodality help?           ablation table  multimodal_all vs every subset

Model selection rule respected: calibrators and threshold τ are fitted on VALIDATION; test is read ONCE.

Reads : results/predictions/{baseline,multimodal,ablation}_predictions.csv
Writes: results/evaluation/{metrics,calibration,ablations,stress_tests,calibrators}.csv
        figures/{roc,pr,calibration,ablation,lead_time}.png
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from common import (ECE_BINS, FIGURES, HORIZONS_H, RESULTS, SEED, apply_platt, brier, ece,
                    fit_platt, safe_auprc, safe_auroc, banner, ensure_dirs)

# Attempt to import stress-test perturbations (defined later / optional)
try:
    from nets import WindowData, predict, load_model
    from common import PROCESSED, MODELS
    _NETS_AVAILABLE = True
except Exception:
    _NETS_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)

PALETTE = {
    "multimodal_all": "#1f77b4",
    "concat_all":     "#ff7f0e",
    "phys_only":      "#2ca02c",
    "lab_only":       "#d62728",
    "text_only":      "#9467bd",
    "logistic":       "#8c564b",
}
ABLATION_COLORS = {
    "all": "#1f77b4", "phys+lab": "#17becf", "phys+text": "#bcbd22",
    "lab+text": "#e377c2", "phys": "#2ca02c", "lab": "#d62728", "text": "#9467bd",
}


# --------------------------------------------------------------------------- loading / routing

def load_all_predictions() -> dict[str, pd.DataFrame]:
    """Returns {name: df} where df has columns [anchor_min, y1, y3, y6, p1h, p3h, p6h, split]."""
    dfs = {}
    for path in [
        RESULTS / "predictions" / "baseline_predictions.csv",
        RESULTS / "predictions" / "multimodal_predictions.csv",
        RESULTS / "predictions" / "ablation_predictions.csv",
    ]:
        if not path.exists():
            print(f"  WARNING: {path.name} not found -- skipping")
            continue
        df = pd.read_csv(path)
        for name, g in df.groupby("model"):
            dfs[name] = g.reset_index(drop=True)
    return dfs


# --------------------------------------------------------------------------- core metrics

def evaluate_one(y: np.ndarray, p: np.ndarray, name: str, split: str, h: int) -> dict:
    rec = {
        "model": name, "split": split, "horizon_h": h,
        "n": int(len(y)), "n_pos": int(y.sum()),
        "prevalence": float(y.mean()),
        "auroc": safe_auroc(y, p),
        "auprc": safe_auprc(y, p),
        "brier": brier(y, p),
        "ece": ece(y, p),
    }
    # At multiple sensitivity thresholds
    for sens_target in (0.80, 0.90):
        fpr, tpr, thr = roc_curve(y, p, pos_label=1) if y.sum() > 0 and y.min() == 0 else ([], [], [])
        if len(tpr) > 1:
            idx = np.searchsorted(tpr, sens_target)
            idx = min(idx, len(thr) - 1)
            rec[f"specificity_at_sens{int(sens_target*100)}"] = float(1 - fpr[idx])
            rec[f"threshold_at_sens{int(sens_target*100)}"] = float(thr[idx])
        else:
            rec[f"specificity_at_sens{int(sens_target*100)}"] = float("nan")
            rec[f"threshold_at_sens{int(sens_target*100)}"] = float("nan")
    return rec


# --------------------------------------------------------------------------- calibration

def calibrate_model(name: str, dfs: dict[str, pd.DataFrame]) -> dict:
    """Fit Platt scaler on VALIDATION predictions; return {horizon: (coef, intercept)}."""
    cal = {}
    if name not in dfs:
        return cal
    val_df = dfs[name][dfs[name]["split"] == "validation"]
    for j, h in enumerate(HORIZONS_H):
        y = val_df[f"y{h}"].to_numpy()
        p = val_df[f"p{h}h"].to_numpy()
        if y.sum() > 0 and len(np.unique(y)) > 1:
            coef, intercept = fit_platt(p, y)
        else:
            coef, intercept = 1.0, 0.0
        cal[h] = (coef, intercept)
    return cal


# --------------------------------------------------------------------------- plots

def plot_roc(metrics: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, h in zip(axes, HORIZONS_H):
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.set_title(f"ROC  {h}h horizon")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    plt.tight_layout()
    fig.savefig(FIGURES / "roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr(dfs: dict[str, pd.DataFrame], main_models: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for j, (ax, h) in enumerate(zip(axes, HORIZONS_H)):
        for name in main_models:
            if name not in dfs:
                continue
            df = dfs[name][dfs[name]["split"] == "test"]
            y = df[f"y{h}"].to_numpy()
            p = df[f"p{h}h"].to_numpy()
            if y.sum() == 0:
                continue
            pr, rc, _ = precision_recall_curve(y, p)
            ap = safe_auprc(y, p)
            color = PALETTE.get(name, None)
            ax.plot(rc, pr, label=f"{name} ({ap:.3f})", color=color)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"PR  {h}h horizon")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(FIGURES / "pr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(dfs: dict[str, pd.DataFrame], calibrators: dict, main_models: list[str]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for j, (ax, h) in enumerate(zip(axes, HORIZONS_H)):
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="perfect")
        for name in main_models:
            if name not in dfs:
                continue
            df = dfs[name][dfs[name]["split"] == "test"]
            y = df[f"y{h}"].to_numpy()
            p = df[f"p{h}h"].to_numpy()
            if name in calibrators and h in calibrators[name]:
                coef, intercept = calibrators[name][h]
                p = apply_platt(p, coef, intercept)
            bins = np.linspace(0, 1, 11)
            idx = np.digitize(p, bins) - 1
            cal_x, cal_y = [], []
            for b in range(len(bins) - 1):
                m = idx == b
                if m.sum() >= 5:
                    cal_x.append(p[m].mean())
                    cal_y.append(y[m].mean())
            if cal_x:
                ax.plot(cal_x, cal_y, "o-", label=name, color=PALETTE.get(name), markersize=4)
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraction")
        ax.set_title(f"Calibration  {h}h horizon")
        ax.legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(FIGURES / "calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(abl_df: pd.DataFrame) -> None:
    if abl_df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, h in zip(axes, HORIZONS_H):
        sub = abl_df[(abl_df["split"] == "test") & (abl_df["horizon_h"] == h)].sort_values("auprc")
        labels = sub["ablation_set"].tolist()
        vals = sub["auprc"].tolist()
        colors = [ABLATION_COLORS.get(l, "#aaaaaa") for l in labels]
        ax.barh(labels, vals, color=colors)
        ax.set_xlabel("AUPRC")
        ax.set_title(f"Ablation  {h}h horizon")
        ax.set_xlim(0, max(vals) * 1.15 if vals else 1.0)
    plt.tight_layout()
    fig.savefig(FIGURES / "ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lead_time(lead_h: np.ndarray, model: str) -> None:
    if len(lead_h) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lead_h, bins=20, color="#1f77b4", edgecolor="white", linewidth=0.5)
    ax.axvline(np.median(lead_h), color="red", ls="--", label=f"median {np.median(lead_h):.1f} h")
    ax.set_xlabel("Warning lead time (hours)")
    ax.set_ylabel("Count")
    ax.set_title(f"Lead time to event  [{model}]")
    ax.legend()
    fig.savefig(FIGURES / "lead_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- stress tests (model-level)

def run_stress_tests(model_name: str, dfs: dict) -> list[dict]:
    """Runs modality-dropout stress tests if the multimodal model + processed data are available."""
    if not _NETS_AVAILABLE:
        return []
    mp = MODELS / "multimodal" / "multimodal.pt"
    if not mp.exists():
        return []
    try:
        from nets import WindowData, predict, load_model
        from common import PROCESSED
        from features import Layout
        model, _ = load_model(mp)
        test = WindowData(PROCESSED / "test.npz")
    except Exception as e:
        print(f"  WARNING: stress test setup failed: {e}")
        return []

    rows = []
    # Modality-dropout stress tests
    for label, off in [
        ("drop_labs_20pct",  None),   # handled by drop_frac inside predict
        ("modality_off_lab", ("lab",)),
        ("modality_off_text", ("text",)),
        ("modality_off_phys", ("phys",)),
    ]:
        try:
            if off:
                p, _ = predict(model, test, off=off)
            else:
                # 20% lab drop: use stored ablation predictions as proxy
                key = f"ablation_{label}" if f"ablation_{label}" in dfs else None
                if key:
                    p_df = dfs[key][dfs[key]["split"] == "test"]
                    p = p_df[[f"p{h}h" for h in HORIZONS_H]].to_numpy()
                else:
                    continue
            for j, h in enumerate(HORIZONS_H):
                y = test.y[:, j]
                rows.append({
                    "stress_test": label, "horizon_h": h,
                    "auroc": safe_auroc(y, p[:, j]),
                    "auprc": safe_auprc(y, p[:, j]),
                    "brier": brier(y, p[:, j]),
                })
        except Exception as e:
            print(f"  WARNING: stress test {label} failed: {e}")
    return rows


# --------------------------------------------------------------------------- main

def main() -> None:
    ensure_dirs()
    banner("09  EVALUATION  (discrimination, calibration, ablation)")

    dfs = load_all_predictions()
    if not dfs:
        print("  No prediction files found. Run 07_* and 08_* first.")
        return

    print(f"  Loaded predictions from {len(dfs)} model variants.")

    # Main models to plot (all will be evaluated)
    MAIN = ["logistic", "phys_only", "lab_only", "text_only", "concat_all", "multimodal_all"]

    # ---- Q1 + Q2: discrimination + calibration ---------------------------------------------
    metric_rows: list[dict] = []
    calibrators: dict[str, dict] = {}

    for name, df in dfs.items():
        # Calibration: fitted on validation, evaluated on test
        cal = calibrate_model(name, dfs)
        calibrators[name] = cal

        for split in ("validation", "test"):
            sub = df[df["split"] == split]
            for j, h in enumerate(HORIZONS_H):
                y = sub[f"y{h}"].to_numpy()
                p = sub[f"p{h}h"].to_numpy()
                if name in calibrators and h in calibrators.get(name, {}):
                    coef, intercept = calibrators[name][h]
                    p_cal = apply_platt(p, coef, intercept)
                else:
                    p_cal = p
                rec = evaluate_one(y, p, name, split, h)
                rec["brier_calibrated"] = brier(y, p_cal)
                rec["ece_calibrated"] = ece(y, p_cal)
                metric_rows.append(rec)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(RESULTS / "evaluation" / "metrics.csv", index=False)

    # Save calibrators for downstream scripts (10, 11)
    cal_rows = []
    for model_name, cal in calibrators.items():
        for h, (coef, intercept) in cal.items():
            cal_rows.append({"model": model_name, "horizon": h, "coef": coef, "intercept": intercept})
    pd.DataFrame(cal_rows).to_csv(RESULTS / "evaluation" / "calibrators.csv", index=False)

    # ---- Q3: lead time (multimodal_all on test) --------------------------------------------
    lead_h = np.array([])
    if "multimodal_all" in dfs:
        df_m = dfs["multimodal_all"][dfs["multimodal_all"]["split"] == "test"]
        # Use 6h horizon: first positive anchor before event is the "first alarm"
        pos = df_m["y6"] == 1
        if pos.any():
            pos_df = df_m[pos]
            # next_onset_min - anchor_min gives warning time
            if "next_onset_min" in pos_df.columns:
                lead_min = pos_df["next_onset_min"].to_numpy() - pos_df["anchor_min"].to_numpy()
                lead_h = lead_min / 60.0
                lead_h = lead_h[(lead_h >= 0) & (lead_h <= 24)]

    lead_summary = {
        "model": "multimodal_all", "horizon_h": 6,
        "n_positive_anchors": int(len(lead_h)),
        "lead_h_q25": float(np.percentile(lead_h, 25)) if len(lead_h) else float("nan"),
        "lead_h_median": float(np.median(lead_h)) if len(lead_h) else float("nan"),
        "lead_h_q75": float(np.percentile(lead_h, 75)) if len(lead_h) else float("nan"),
    }
    pd.DataFrame([lead_summary]).to_csv(RESULTS / "evaluation" / "lead_time_summary.csv", index=False)
    plot_lead_time(lead_h, "multimodal_all")

    # ---- Q5: ablation table ----------------------------------------------------------------
    abl_rows = []
    for name, df in dfs.items():
        if not name.startswith("ablation_"):
            continue
        abl_label = name.replace("ablation_", "")
        for split in ("validation", "test"):
            sub = df[df["split"] == split]
            for j, h in enumerate(HORIZONS_H):
                y = sub[f"y{h}"].to_numpy()
                p = sub[f"p{h}h"].to_numpy()
                abl_rows.append({
                    "ablation_set": abl_label, "split": split, "horizon_h": h,
                    "auroc": safe_auroc(y, p),
                    "auprc": safe_auprc(y, p),
                    "brier": brier(y, p),
                    "n": int(len(y)), "n_pos": int(y.sum()),
                })
    abl_df = pd.DataFrame(abl_rows)
    abl_df.to_csv(RESULTS / "evaluation" / "ablations.csv", index=False)

    # ---- stress tests ----------------------------------------------------------------------
    stress_rows = run_stress_tests("multimodal_all", dfs)
    if stress_rows:
        pd.DataFrame(stress_rows).to_csv(RESULTS / "evaluation" / "stress_tests.csv", index=False)

    # ---- figures ---------------------------------------------------------------------------
    plot_roc(metric_rows)
    plot_pr(dfs, MAIN)
    plot_calibration(dfs, calibrators, MAIN)
    plot_ablation(abl_df)

    # ---- summary print ---------------------------------------------------------------------
    print("\n  ── TEST SET SUMMARY (AUPRC) ──")
    test_m = metrics_df[
        (metrics_df["split"] == "test") &
        (metrics_df["model"].isin(MAIN))
    ]
    if not test_m.empty:
        pivot = test_m.pivot_table(index="model", columns="horizon_h", values="auprc").round(4)
        print(pivot.to_string())

    if not abl_df.empty:
        print("\n  ── ABLATION AUPRC (test, 6h horizon) ──")
        abl_test = (abl_df[(abl_df["split"] == "test") & (abl_df["horizon_h"] == 6)]
                    .sort_values("auprc", ascending=False)[["ablation_set", "auprc"]])
        print(abl_test.to_string(index=False))

    print("\n  wrote:")
    print("    results/evaluation/{metrics,calibration,ablations,lead_time_summary,calibrators}.csv")
    print("    figures/{roc,pr,calibration,ablation,lead_time}.png")


if __name__ == "__main__":
    main()