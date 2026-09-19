"""
08_train_multimodal.py  --  PHASE 11: final multimodal architecture

Architecture (frozen):
  - Physiological encoder: TCN over 24 hourly rows  (physiology branch)
  - Laboratory encoder:    MLP per hourly row        (irregular lab branch)
  - Text branch:           frozen note embeddings projected to d  (no fine-tuning)
  - Causal cross-modal attention:  phys <- text,  phys <- lab,  lab <- text
  - Missingness-aware reliability gate:  Z_t = Σ_k m_k α_k Z_k  (renorm. over active modalities)
  - GRU longitudinal patient state (initialised from static features)
  - 3 sigmoid heads: 1h / 3h / 6h  (multi-task BCE loss, equal lambda)

Also trains ablation variants by disabling modality combinations via `off=` at inference time
(same weights, different modality masks) -- ablation predictions are saved for 09_evaluate.py.

Model selection: best validation mean AUPRC; test set never touched here.

Reads : data/processed/{train,validation,test}.npz
Writes: models/multimodal/multimodal.pt
        results/predictions/multimodal_predictions.csv   (all modalities)
        results/predictions/ablation_predictions.csv     (per-modality subsets)
        results/evaluation/multimodal_training_log.csv
        results/evaluation/model_metadata.csv
"""
import numpy as np
import pandas as pd

from common import (HORIZONS_H, MODELS, PROCESSED, RESULTS, SEED, banner, ensure_dirs,
                    safe_auprc, safe_auroc, brier, set_seed)
from nets import (MultimodalRiskNet, WindowData, fit_model, predict, save_checkpoint,
                  MODALITIES, DEVICE)

# Ablation subsets:  (label, modalities_to_KEEP)
# The "off=" parameter in predict() takes the modalities to DISABLE;
# we compute off = all_modalities - keep
ABLATION_SETS = [
    ("phys",         ("phys",)),
    ("lab",          ("lab",)),
    ("text",         ("text",)),
    ("phys+lab",     ("phys", "lab")),
    ("phys+text",    ("phys", "text")),
    ("lab+text",     ("lab", "text")),
    ("all",          MODALITIES),          # full model -- no ablation
]


def load_splits():
    train = WindowData(PROCESSED / "train.npz")
    val = WindowData(PROCESSED / "validation.npz")
    test = WindowData(PROCESSED / "test.npz")
    return train, val, test


def quick_metrics(y: np.ndarray, p: np.ndarray, split: str, model: str) -> list[dict]:
    rows = []
    for j, h in enumerate(HORIZONS_H):
        rows.append({
            "model": model, "split": split, "horizon_h": h,
            "auroc": safe_auroc(y[:, j], p[:, j]),
            "auprc": safe_auprc(y[:, j], p[:, j]),
            "brier": brier(y[:, j], p[:, j]),
            "n": int(len(y)),
            "n_pos": int(y[:, j].sum()),
        })
    return rows


def main() -> None:
    ensure_dirs()
    banner("08  MULTIMODAL MODEL")

    train, val, test = load_splits()
    dims = train.dims
    print(f"  dims: {dims}")
    print(f"  train anchors: {len(train):,}  val: {len(val):,}  test: {len(test):,}")
    print(f"  device: {DEVICE}")

    set_seed(SEED)

    # ---- build model -----------------------------------------------------------------------
    model = MultimodalRiskNet(
        dims=dims,
        use=MODALITIES,
        D=64,           # per-modality representation dimension
        H=96,           # GRU hidden size
        heads=4,        # cross-modal attention heads
        drop=0.20,
        mod_drop=0.15,  # training-time modality dropout (builds missing-modality robustness)
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  MultimodalRiskNet: {n_params:,} parameters")

    # ---- train (model selection on validation AUPRC; test is untouched) --------------------
    training_meta = fit_model(
        model, train, val,
        name="multimodal",
        epochs=40,
        bs=512,
        lr=1e-3,
        wd=1e-4,
        patience=6,
        seed=SEED,
        verbose=True,
    )
    save_checkpoint(MODELS / "multimodal" / "multimodal.pt", model, training_meta)
    print(f"\n  best epoch: {training_meta['best_epoch']}  "
          f"val mean-AUPRC: {training_meta['best_val_mean_auprc']:.4f}")

    # ---- save training log -----------------------------------------------------------------
    log_df = pd.DataFrame(training_meta.get("log", []))
    log_df.to_csv(RESULTS / "evaluation" / "multimodal_training_log.csv", index=False)

    # ---- full model predictions (all modalities) -------------------------------------------
    all_metrics: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    for sp, data in (("validation", val), ("test", test)):
        p, _ = predict(model, data)
        all_metrics.extend(quick_metrics(data.y, p, sp, "multimodal_all"))
        f = data.frame()
        for j, h in enumerate(HORIZONS_H):
            f[f"p{h}h"] = p[:, j]
        f["model"] = "multimodal_all"
        f["split"] = sp
        pred_frames.append(f)

    pd.concat(pred_frames, ignore_index=True).to_csv(
        RESULTS / "predictions" / "multimodal_predictions.csv", index=False)

    # ---- ablation predictions --------------------------------------------------------------
    # Using the SAME trained model weights but masking out modalities at inference time.
    # This is valid because (a) the model was trained with modality dropout, and
    # (b) the ablation evaluation is on the test set with fixed weights (no re-training).
    abl_frames: list[pd.DataFrame] = []
    for label, keep in ABLATION_SETS:
        off = tuple(m for m in MODALITIES if m not in keep)
        for sp, data in (("validation", val), ("test", test)):
            p, _ = predict(model, data, off=off)
            all_metrics.extend(quick_metrics(data.y, p, sp, f"ablation_{label}"))
            f = data.frame()
            for j, h in enumerate(HORIZONS_H):
                f[f"p{h}h"] = p[:, j]
            f["model"] = f"ablation_{label}"
            f["modalities_kept"] = ",".join(keep)
            f["modalities_off"] = ",".join(off) if off else "none"
            f["split"] = sp
            abl_frames.append(f)

    pd.concat(abl_frames, ignore_index=True).to_csv(
        RESULTS / "predictions" / "ablation_predictions.csv", index=False)

    # ---- metrics summary -------------------------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(RESULTS / "evaluation" / "multimodal_quick_metrics.csv", index=False)

    print("\n  Validation AUPRC  (full model + ablations):")
    vsum = (metrics_df[metrics_df["split"] == "validation"]
            .pivot_table(index="model", columns="horizon_h", values="auprc")
            .round(4))
    print(vsum.to_string())

    # ---- model metadata record (no config file; all info in one csv) -----------------------
    meta_rows = [{
        "model": "multimodal",
        "arch": "MultimodalRiskNet",
        "n_params": n_params,
        "D": 64, "H": 96, "heads": 4, "drop": 0.20, "mod_drop": 0.15,
        "epochs_trained": training_meta["best_epoch"],
        "best_val_mean_auprc": training_meta["best_val_mean_auprc"],
        "seed": SEED,
        "training_split": "train",
        "selection_split": "validation",
        "horizons_h": str(list(HORIZONS_H)),
        "lookback_h": 24,
        "dims": str(dims),
    }]
    pd.DataFrame(meta_rows).to_csv(
        RESULTS / "evaluation" / "model_metadata.csv", index=False)

    print("\n  wrote:")
    print("    models/multimodal/multimodal.pt")
    print("    results/predictions/{multimodal,ablation}_predictions.csv")
    print("    results/evaluation/{multimodal_training_log,multimodal_quick_metrics,model_metadata}.csv")


if __name__ == "__main__":
    main()