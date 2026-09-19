"""
07_train_baselines.py  --  PHASE 10: baseline models

Trains five baselines on the processed tensors (no look at the test set during model selection):

  B0  logistic regression on flat features (collapses temporal shape -- the anti-pattern the PS criticises)
  B1  physiology-only ConcatGRU
  B2  lab-only ConcatGRU
  B3  text-only ConcatGRU
  B4  simple concatenation multimodal ConcatGRU (all modalities, naive fusion)

All baselines share the same WindowData tensors; modality masking ("off=") disables unavailable streams.
Model selection: best validation mean AUPRC (over three horizons). Test set is never touched here.

Reads : data/processed/{train,validation,test}.npz, data/processed/metadata.json
Writes: models/baseline/{name}.pt (or .joblib for LR)
        results/predictions/baseline_predictions.csv
        results/evaluation/baseline_training_log.csv
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (HORIZONS_H, MODELS, PROCESSED, RESULTS, SEED, banner, ensure_dirs,
                    safe_auprc, safe_auroc, brier, set_seed)
from nets import (ConcatGRU, WindowData, fit_model, flat_features, predict,
                  save_checkpoint, MODALITIES, DEVICE)


# --------------------------------------------------------------------------- helpers

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


def save_predictions(df: pd.DataFrame, model_name: str, preds: dict) -> pd.DataFrame:
    """Append model predictions to the running predictions dataframe."""
    for sp, (y, p, frame) in preds.items():
        for j, h in enumerate(HORIZONS_H):
            col = f"p{h}h"
            frame[col] = p[:, j]
        frame["model"] = model_name
        frame["split"] = sp
        df = pd.concat([df, frame], ignore_index=True)
    return df


# --------------------------------------------------------------------------- B0: logistic regression

def train_logistic(train: WindowData, val: WindowData, test: WindowData) -> tuple:
    """Flat-feature logistic regression baseline -- intentionally collapses temporal shape."""
    print("  [B0] logistic regression on flat features ...")
    set_seed(SEED)

    X_tr = flat_features(train)
    X_va = flat_features(val)
    X_te = flat_features(test)

    models, p_val_all, p_test_all = [], [], []
    for j, h in enumerate(HORIZONS_H):
        y_tr = train.y[:, j]
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=0.1, max_iter=2000, random_state=SEED, class_weight="balanced")),
        ])
        pipe.fit(X_tr, y_tr)
        models.append(pipe)
        p_val_all.append(pipe.predict_proba(X_va)[:, 1])
        p_test_all.append(pipe.predict_proba(X_te)[:, 1])

    p_val = np.stack(p_val_all, axis=1)
    p_test = np.stack(p_test_all, axis=1)
    joblib.dump(models, MODELS / "baseline" / "logistic.joblib")

    # Return (y, p) for each split
    return {
        "validation": (val.y, p_val, val.frame()),
        "test": (test.y, p_test, test.frame()),
    }


# --------------------------------------------------------------------------- B1-B4: GRU baselines

BASELINE_CONFIGS = [
    ("phys_only",    ("phys",),                    {}),
    ("lab_only",     ("lab",),                     {}),
    ("text_only",    ("text",),                    {}),
    ("concat_all",   MODALITIES,                   {}),   # naive concatenation, all modalities
]


def train_gru_baseline(name: str, use: tuple, kw: dict,
                       train: WindowData, val: WindowData, test: WindowData,
                       dims: dict) -> tuple:
    print(f"  [{name}] ConcatGRU  use={use} ...")
    set_seed(SEED)
    model = ConcatGRU(dims, use=use, H=64, drop=0.2, mod_drop=0.0, **kw)
    training_meta = fit_model(
        model, train, val,
        name=name,
        epochs=30,
        bs=512,
        lr=1e-3,
        wd=1e-4,
        patience=5,
        seed=SEED,
        verbose=True,
    )
    save_checkpoint(MODELS / "baseline" / f"{name}.pt", model, training_meta)

    p_val, _ = predict(model, val)
    p_test, _ = predict(model, test)

    return training_meta, {
        "validation": (val.y, p_val, val.frame()),
        "test": (test.y, p_test, test.frame()),
    }


# --------------------------------------------------------------------------- main

def main() -> None:
    ensure_dirs()
    banner("07  BASELINE MODELS")

    train, val, test = load_splits()
    dims = train.dims
    print(f"  dims: {dims}")
    print(f"  train anchors: {len(train):,}  val: {len(val):,}  test: {len(test):,}")

    all_metrics: list[dict] = []
    all_preds = pd.DataFrame()
    training_logs: list[dict] = []

    # ---- B0: logistic regression -----------------------------------------------------------
    lr_preds = train_logistic(train, val, test)
    for sp, (y, p, frame) in lr_preds.items():
        all_metrics.extend(quick_metrics(y, p, sp, "logistic"))
    all_preds = save_predictions(all_preds, "logistic", lr_preds)

    # ---- B1-B4: GRU baselines --------------------------------------------------------------
    for name, use, kw in BASELINE_CONFIGS:
        meta, preds = train_gru_baseline(name, use, kw, train, val, test, dims)
        for sp, (y, p, frame) in preds.items():
            all_metrics.extend(quick_metrics(y, p, sp, name))
        all_preds = save_predictions(all_preds, name, preds)
        # flatten the per-epoch log
        for row in meta.get("log", []):
            row["model"] = name
            training_logs.append(row)

    # ---- save outputs ----------------------------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics)
    print("\n  Validation AUPRC summary:")
    vsum = (metrics_df[metrics_df["split"] == "validation"]
            .pivot_table(index="model", columns="horizon_h", values="auprc")
            .round(4))
    print(vsum.to_string())

    # Write predictions with model column so 09_evaluate can route them
    out_path = RESULTS / "predictions" / "baseline_predictions.csv"
    all_preds.to_csv(out_path, index=False)
    print(f"\n  wrote {out_path}  ({len(all_preds):,} rows)")

    # Quick metrics for the report
    metrics_df.to_csv(RESULTS / "evaluation" / "baseline_quick_metrics.csv", index=False)

    if training_logs:
        pd.DataFrame(training_logs).to_csv(
            RESULTS / "evaluation" / "baseline_training_log.csv", index=False)


if __name__ == "__main__":
    main()