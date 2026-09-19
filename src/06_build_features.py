"""
06_build_features.py  --  PHASES 6, 7, 9: laboratory + text representations and the final model tensors

For every ICU stay we build hourly "state rows" (see features.py); a sample for anchor hour a is rows a-23..a.

  physiology  hourly mean/min/max, observed flag, hours-since-observation          (from data/interim/timeline.csv)
  laboratory  latest value, freshness, 24 h slope, 24 h count, ever-measured flag  (labResultOffset = availability)
  text        frozen embedding of notes visible at their DOCUMENTATION time (noteEnteredOffset), NOT noteOffset
  static      admission-time features only (no outcome / discharge information)
  masks       modality availability per row  [physiology observed, any lab so far, new note this hour]

Leakage controls implemented here
  * scalers / lab-name selection / text-encoder vocabulary are fitted on TRAIN stays only, then frozen;
  * a truncation test proves causality: deleting every observation after 60*g minutes leaves row g unchanged.

Usage : python src/06_build_features.py [--text-encoder tfidf | hf:emilyalsentzer/Bio_ClinicalBERT]
Reads : data/interim/{cohort,split,labels,timeline}.csv + raw lab/note tables
Writes: data/interim/{lab_clean,note_clean}.csv, data/interim/text_embeddings.npy, models/text_encoder.joblib,
        data/processed/{train,validation,test}.npz, data/processed/metadata.json,
        results/features/feature_summary.csv
"""
import argparse
import sys

import joblib
import numpy as np
import pandas as pd

from common import (HORIZONS_H, INTERIM, MODELS, N_LABS, PROCESSED, RESULTS, SEED, TEXT_DIM,
                    banner, ensure_dirs, read_table, save_json, set_seed)
from features import (HFEncoder, Layout, TfidfSvdEncoder, fit_lab_scaler, fit_phys_scaler,
                      fit_static_scaler, lab_feature_names, lab_features, phys_feature_names,
                      phys_features, phys_raw, select_lab_names, static_features, text_features,
                      hourly_phys_table)


# --------------------------------------------------------------------------- data loading helpers

def load_labs(coh: pd.DataFrame, a_max: dict) -> pd.DataFrame:
    """Load and clean lab table, restricted to cohort stays and anchored time windows."""
    lab = read_table("lab", usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"])
    lab = lab.rename(columns={
        "patientunitstayid": "stay_id",
        "labresultoffset": "t",
        "labname": "name",
        "labresult": "value",
    })
    lab["value"] = pd.to_numeric(lab["value"], errors="coerce")
    lab["t"] = pd.to_numeric(lab["t"], errors="coerce")
    lab = lab.dropna(subset=["value", "t", "name"])
    lab = lab[lab["stay_id"].isin(set(coh["stay_id"]))].copy()
    lab["name"] = lab["name"].astype(str).str.strip().str.lower()
    # Only keep observations up to the last anchor for that stay (causal ceiling)
    lab = lab[lab["t"] <= 60.0 * lab["stay_id"].map(a_max)]
    # Remove implausible negative offsets (lab drawn before ICU admission)
    lab = lab[lab["t"] >= -60.0]
    return lab.sort_values(["stay_id", "t"]).reset_index(drop=True)


def load_notes(coh: pd.DataFrame, a_max: dict) -> pd.DataFrame:
    """Load notes; availability time = noteEnteredOffset (documentation time).
    Falls back to noteOffset only when documentation time is missing.
    Critical: at anchor t, we include ONLY notes with avail_min <= 60*anchor_hour."""
    n = read_table("note", usecols=["patientunitstayid", "noteoffset", "noteenteredoffset",
                                    "notetype", "notevalue", "notetext"])
    n = n.rename(columns={"patientunitstayid": "stay_id"})

    # Resolve text column (some eICU mirrors use notevalue instead of notetext)
    if "notetext" in n.columns:
        txt = n["notetext"].copy()
    else:
        txt = pd.Series([None] * len(n), index=n.index)

    if "notevalue" in n.columns:
        txt = txt.fillna(n["notevalue"])

    n["text"] = (txt.astype(str)
                 .str.replace(r"\s+", " ", regex=True)
                 .str.strip()
                 .str.slice(0, 2000))
    n.loc[txt.isna() | (txt.astype(str).str.strip() == ""), "text"] = ""

    n["note_offset"] = pd.to_numeric(n["noteoffset"], errors="coerce")
    # LEAKAGE CONTROL: availability time is documentation time, not clinical event time
    n["avail_min"] = pd.to_numeric(n.get("noteenteredoffset", pd.Series(dtype=float)),
                                   errors="coerce").fillna(n["note_offset"])

    n = n[
        n["stay_id"].isin(set(coh["stay_id"])) &
        (n["text"].str.len() >= 3) &
        n["avail_min"].notna()
    ].copy()

    # Restrict to notes available up to the last anchor of each stay
    n = n[n["avail_min"] <= 60.0 * n["stay_id"].map(a_max)]
    n["note_type"] = n["notetype"].astype(str) if "notetype" in n.columns else ""

    return (n[["stay_id", "note_offset", "avail_min", "note_type", "text"]]
            .sort_values(["stay_id", "avail_min"])
            .reset_index(drop=True))


# --------------------------------------------------------------------------- causality / leakage test

def truncation_test(layout: Layout, hourly: pd.DataFrame, ph_scaler: dict,
                    lab: pd.DataFrame, lab_scaler: dict,
                    notes: pd.DataFrame, emb: np.ndarray,
                    full: tuple, n_checks: int = 20) -> int:
    """Causality proof: rebuild one row from data truncated at that row's time; compare to the full build.

    The contract: if we delete every observation with t > 60*g, the feature row g must be identical
    (within floating-point tolerance) to the row computed from the complete dataset.
    This rules out future-leakage in all three modalities simultaneously.
    """
    rng = np.random.default_rng(SEED)
    phys_f, lab_f, text_f = full
    ok = 0

    # Only test stays that have at least 25 rows (enough to pick a non-trivial interior row)
    cand = [(s, n) for s, n in zip(layout.stay_ids.tolist(), layout.n_rows.tolist()) if n > 25]
    if not cand:
        print("  truncation test: no stays with >25 rows -- skipped")
        return 0

    chosen = rng.choice(len(cand), size=min(n_checks, len(cand)), replace=False)
    for i in chosen:
        sid, R = cand[int(i)]
        g = int(rng.integers(24, R))          # anchor hour to check (must have full 24h lookback)

        sub = Layout(np.array([sid]), np.array([g + 1]))

        # Truncate each modality at 60*g minutes
        h_sub = hourly[(hourly["stay_id"] == sid) & (hourly["hour"] <= g)]
        p_sub, _ = phys_features(sub, phys_raw(sub, h_sub), ph_scaler)

        lab_sub = lab[(lab["stay_id"] == sid) & (lab["t"] <= 60.0 * g)]
        l_sub, _ = lab_features(sub, lab_sub, lab_scaler)

        note_mask = (notes["stay_id"] == sid) & (notes["avail_min"] <= 60.0 * g)
        emb_sub = emb[note_mask.to_numpy()]
        t_sub, _ = text_features(sub, notes[note_mask], emb_sub)

        # Compare with full-dataset row
        pos_in_layout = int(layout.pos_of(np.array([sid]))[0])
        r = int(layout.row_start[pos_in_layout]) + g

        for name, a, b in (("phys", p_sub[g], phys_f[r]),
                            ("lab",  l_sub[g], lab_f[r]),
                            ("text", t_sub[g], text_f[r])):
            if not np.allclose(a, b, atol=1e-5):
                raise AssertionError(
                    f"CAUSALITY VIOLATION in {name}: stay {sid}, hour {g}\n"
                    f"  truncated row: {a[:5]}\n  full-data row: {b[:5]}"
                )
        ok += 1

    return ok


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Build model-ready tensors from interim tables.")
    ap.add_argument("--text-encoder", default="tfidf",
                    help="'tfidf' (offline, default) or 'hf:<model_name>' e.g. hf:emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--n-labs", type=int, default=N_LABS,
                    help=f"Number of lab variables to keep (default: {N_LABS})")
    ap.add_argument("--skip-truncation-test", action="store_true",
                    help="Skip the causality truncation test (faster, for debugging only)")
    args = ap.parse_args()

    ensure_dirs()
    set_seed(SEED)
    banner("06  FEATURES + TENSORS")

    # ---- load cohort / split / labels -------------------------------------------------------
    coh = pd.read_csv(INTERIM / "cohort.csv").sort_values("stay_id").reset_index(drop=True)
    split_ser = pd.read_csv(INTERIM / "split.csv").set_index("stay_id")["split"]
    labels = pd.read_csv(INTERIM / "labels.csv")

    # Restrict cohort to stays that actually have labels (should be identical, but defensive)
    coh = coh[coh["stay_id"].isin(set(labels["stay_id"]))].reset_index(drop=True)

    # Maximum anchor hour per stay (used to bound data loading)
    a_max = labels.groupby("stay_id")["anchor_hour"].max().to_dict()

    # Build the global row layout: one entry per stay, length = (max_anchor + 1) rows
    layout = Layout(
        coh["stay_id"].to_numpy(),
        np.array([a_max[s] + 1 for s in coh["stay_id"]])
    )

    sp_of_stay = coh["stay_id"].map(split_ser).to_numpy()
    train_stays = set(coh.loc[sp_of_stay == "train", "stay_id"])

    print(f"  stays: {len(coh):,}   hourly rows (total): {layout.total:,}   "
          f"train stays: {len(train_stays):,}")

    # ---- physiology -------------------------------------------------------------------------
    print("\n  [1/5] building physiology features ...")
    hourly = pd.read_csv(INTERIM / "timeline.csv")
    vals = phys_raw(layout, hourly)
    ph_scaler = fit_phys_scaler(vals, layout, train_stays)
    phys_f, phys_any = phys_features(layout, vals, ph_scaler)
    assert np.isfinite(phys_f).all(), "non-finite values in physiology features"
    print(f"  physiology  : {phys_f.shape[1]} features/row  "
          f"(~{int(phys_any.mean() * 100)}% rows with any observation)")

    # ---- laboratory -------------------------------------------------------------------------
    print("  [2/5] building laboratory features ...")
    lab = load_labs(coh, a_max)
    lab_names = select_lab_names(lab, train_stays, args.n_labs)
    lab = lab[lab["name"].isin(lab_names)].reset_index(drop=True)
    lab_scaler = fit_lab_scaler(lab, lab_names, train_stays)
    lab_f, lab_any = lab_features(layout, lab, lab_scaler)
    assert np.isfinite(lab_f).all(), "non-finite values in laboratory features"
    lab.to_csv(INTERIM / "lab_clean.csv", index=False)
    print(f"  laboratory  : {len(lab_names)} variables × 5 stats = {lab_f.shape[1]} features/row")
    print(f"    selected labs: {lab_names}")

    # ---- text -------------------------------------------------------------------------------
    print("  [3/5] building text features ...")
    notes = load_notes(coh, a_max)
    is_train_note = notes["stay_id"].isin(train_stays).to_numpy()
    enc_info = {"kind": "none", "dim": TEXT_DIM}

    if is_train_note.sum() >= 20:
        if args.text_encoder.startswith("hf:"):
            enc = HFEncoder(args.text_encoder[3:])
        else:
            enc = TfidfSvdEncoder()

        # LEAKAGE CONTROL: fit vocabulary / scaler on TRAIN notes only
        train_texts = notes.loc[is_train_note, "text"].drop_duplicates().tolist()
        enc.fit(train_texts)
        emb = enc.transform(notes["text"].tolist())
        joblib.dump(enc, MODELS / "text_encoder.joblib")
        enc_info = {"kind": enc.kind, "dim": TEXT_DIM}
        print(f"  text encoder: {enc.kind}  fitted on {len(train_texts):,} unique train notes")
    else:
        print(f"  WARNING: only {is_train_note.sum()} usable training notes -- "
              "text modality will be zeros (mask 0). At least 20 are needed.")
        emb = np.zeros((len(notes), TEXT_DIM), dtype=np.float32)
        enc_info = {"kind": "none_insufficient_data", "dim": TEXT_DIM}

    notes.to_csv(INTERIM / "note_clean.csv", index=False)
    np.save(INTERIM / "text_embeddings.npy", emb)
    text_f, text_any = text_features(layout, notes, emb)
    assert np.isfinite(text_f).all(), "non-finite values in text features"
    print(f"  text        : {len(notes):,} notes  →  {text_f.shape[1]} features/row  "
          f"(~{int(text_any.mean() * 100)}% rows with a new note)")

    # ---- static -----------------------------------------------------------------------------
    print("  [4/5] building static features ...")
    st_scaler = fit_static_scaler(coh, train_stays)
    static, st_names = static_features(coh, st_scaler)
    assert np.isfinite(static).all(), "non-finite values in static features"
    print(f"  static      : {static.shape[1]} features/stay  [{', '.join(st_names)}]")

    # ---- modality mask ----------------------------------------------------------------------
    # mod[r, 0] = physiology observed in this hour
    # mod[r, 1] = any lab result available for this stay up to this hour
    # mod[r, 2] = a new note was documented in this hour
    mod = np.stack([phys_any, lab_any, text_any], axis=1).astype(np.uint8)

    # ---- causality / leakage test -----------------------------------------------------------
    if not args.skip_truncation_test:
        print("  [5/5] running truncation (causality) test ...")
        n_ok = truncation_test(
            layout, hourly, ph_scaler,
            lab, lab_scaler,
            notes, emb,
            (phys_f, lab_f, text_f),
            n_checks=20,
        )
        print(f"  truncation test passed on {n_ok} random (stay, hour) pairs -- "
              "features depend only on data <= anchor time  ✓")
    else:
        print("  [5/5] truncation test SKIPPED (--skip-truncation-test)")

    # ---- write per-split .npz files ---------------------------------------------------------
    print("\n  writing split tensors ...")
    labv = labels[labels["valid"] == 1]
    meta_split = {}

    for sp in ["train", "validation", "test"]:
        s_idx = np.flatnonzero(sp_of_stay == sp)
        stays = coh["stay_id"].to_numpy()[s_idx]
        n_rows = layout.n_rows[s_idx]

        # Boolean mask over ALL rows selecting rows that belong to this split
        rmask = np.isin(layout.row_stay, stays)

        # Per-split anchor table (valid anchors only, sorted)
        a = labv[labv["stay_id"].isin(set(stays))].sort_values(["stay_id", "anchor_min"])

        # Index of the stay in the per-split stay array
        a_stay = np.searchsorted(stays, a["stay_id"].to_numpy())
        assert np.all(stays[a_stay] == a["stay_id"].to_numpy()), \
            f"stay id mismatch in split '{sp}'"

        # Sanity: every anchor hour must be within the allocated rows
        row_start_local = np.r_[0, np.cumsum(n_rows)[:-1]].astype(np.int64)
        assert np.all(a["anchor_hour"].to_numpy() <= n_rows[a_stay] - 1), \
            f"anchor hour out of range in split '{sp}'"
        assert a["anchor_hour"].min() >= 24, \
            f"anchor below lookback horizon in split '{sp}'"

        y = a[[f"y{h}" for h in HORIZONS_H]].to_numpy().astype(np.uint8)

        np.savez_compressed(
            PROCESSED / f"{sp}.npz",
            # stay-level arrays
            stay_ids=stays,
            row_start=row_start_local,
            n_rows=n_rows,
            # row-level feature arrays (only rows belonging to this split)
            phys=phys_f[rmask],
            lab=lab_f[rmask],
            text=text_f[rmask],
            mod=mod[rmask],
            # static (one row per stay)
            static=static[s_idx],
            # anchor arrays
            a_stay=a_stay.astype(np.int64),
            a_hour=a["anchor_hour"].to_numpy().astype(np.int64),
            a_min=a["anchor_min"].to_numpy().astype(np.float64),
            y=y,
            next_onset=a["next_onset_min"].to_numpy().astype(np.float64),
        )

        meta_split[sp] = {
            "n_stays": int(len(stays)),
            "n_rows": int(rmask.sum()),
            "n_anchors": int(len(a)),
            **{f"pos_{h}h": int(y[:, j].sum()) for j, h in enumerate(HORIZONS_H)},
            **{f"prevalence_{h}h": round(float(y[:, j].mean()), 4) for j, h in enumerate(HORIZONS_H)},
        }
        print(f"  {sp:<12} stays {len(stays):>5,}  anchors {len(a):>7,}  "
              f"pos(1/3/6h) {meta_split[sp]['pos_1h']}/{meta_split[sp]['pos_3h']}/{meta_split[sp]['pos_6h']}")

    # ---- metadata.json ----------------------------------------------------------------------
    meta = {
        "seed": SEED,
        "horizons_h": list(HORIZONS_H),
        "lookback_rows": 24,
        "grid_min": 60,
        "dims": {
            "d_p": int(phys_f.shape[1]),
            "d_l": int(lab_f.shape[1]),
            "d_t": int(text_f.shape[1]),
            "d_s": int(static.shape[1] + 1),   # +1 for log1p anchor hour appended at batch time
        },
        "phys_names": phys_feature_names(),
        "lab_names": lab_names,
        "lab_feature_names": lab_feature_names(lab_names),
        "static_names": st_names + ["log1p_anchor_hour"],
        "text_encoder": enc_info,
        "n_labs_selected": len(lab_names),
        "modality_availability": {
            "phys_pct_rows": float(round(phys_any.mean() * 100, 2)),
            "lab_pct_rows": float(round(lab_any.mean() * 100, 2)),
            "text_pct_rows": float(round(text_any.mean() * 100, 2)),
        },
        "leakage_note": (
            "scalers, lab-name selection and text vocabulary were fitted on TRAIN stays only; "
            "normalization parameters from training are applied to validation and test."
        ),
        "splits": meta_split,
    }
    save_json(meta, PROCESSED / "metadata.json")

    # ---- feature summary --------------------------------------------------------------------
    tr_mask = np.isin(layout.row_stay, np.array(list(train_stays)))
    rows = []
    for i, ch in enumerate(["hr", "rr", "spo2", "temp", "sbp", "dbp", "map"]):
        # obs flag is index 3 in the 5-value block
        obs_flag = phys_f[tr_mask, i * 5 + 3]
        rows.append({
            "block": "phys", "feature": ch,
            "pct_rows_observed": float(round(100 * obs_flag.mean(), 2)),
        })
    for i, n in enumerate(lab_names):
        # ever-measured flag is index 4 in the 5-value block
        ever_flag = lab_f[tr_mask, i * 5 + 4]
        rows.append({
            "block": "lab", "feature": n,
            "pct_rows_observed": float(round(100 * ever_flag.mean(), 2)),
        })
    rows.append({
        "block": "text", "feature": "note_in_window",
        "pct_rows_observed": float(round(100 * text_any[tr_mask].mean(), 2)),
    })
    for n in st_names:
        rows.append({"block": "static", "feature": n, "pct_rows_observed": float("nan")})

    pd.DataFrame(rows).to_csv(RESULTS / "features" / "feature_summary.csv", index=False)

    print("\n  wrote:")
    print("    data/processed/{train,validation,test}.npz")
    print("    data/processed/metadata.json")
    print("    data/interim/{lab_clean,note_clean}.csv")
    print("    data/interim/text_embeddings.npy")
    print("    results/features/feature_summary.csv")


if __name__ == "__main__":
    main()