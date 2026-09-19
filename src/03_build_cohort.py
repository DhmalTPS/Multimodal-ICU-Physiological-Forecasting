"""
03_build_cohort.py  --  PHASE 4: cohort criteria and the patient-level split

Cohort rules (applied in order, flow counts are written out):
    1. valid stay identifier and a known ICU discharge time
    2. ICU length of stay > 24 h  (we need a full 24 h look-back and at least one anchor)
    3. known ICU discharge status (Alive / Expired)
    4. plausible timeline: hospital admission not after ICU admission
    5. >= 12 vital-sign observations during the first 24 h (sufficient observation history)
    6. locked-endpoint episodes have plausible timing (0 <= onset <= ICU discharge)

Split: 70 / 15 / 15 by *patient* (uniquepid -- a patient can have several ICU stays), stratified on "ever has an
event", performed BEFORE any normalisation or model fitting.  All anchors of a stay/patient stay in one split.

Reads : raw tables, results/audit/locked_task.json
Writes: data/interim/cohort.csv, data/interim/split.csv,
        results/cohort/cohort_summary.csv, results/cohort/split_summary.csv
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (LOOKBACK_H, MIN_VITAL_OBS_FIRST24, RESULTS, SEED, INTERIM, SeriesIndex, anchor_hours, banner,
                    build_events, ensure_dirs, load_clean_vitals, load_locked_task, load_patient)


def strat_split(df: pd.DataFrame, test_size: float):
    y = df["has_event"].astype(int)
    strat = y if y.value_counts().min() >= 2 and y.nunique() > 1 else None
    return train_test_split(df, test_size=test_size, random_state=SEED, stratify=strat)


def main() -> None:
    ensure_dirs()
    banner("03  COHORT + PATIENT-LEVEL SPLIT")
    task = load_locked_task()
    endpoint = task["endpoint"]
    pat = load_patient()
    flow = [("all ICU stays in patient table", pat)]

    c = pat[pat["stay_id"].notna() & pat["unitdischargeoffset"].notna()]
    flow.append(("valid id + ICU discharge time", c))
    c = c[c["unitdischargeoffset"] > LOOKBACK_H * 60]
    flow.append((f"ICU LOS > {LOOKBACK_H} h", c))
    c = c[c["unitdischargestatus"].astype(str).str.strip().str.lower().isin(["alive", "expired"])]
    flow.append(("known ICU discharge status", c))
    c = c[~(c["hospitaladmitoffset"] > 0)]
    flow.append(("plausible hospital/ICU timeline", c))

    vit = load_clean_vitals(set(c["stay_id"]))
    first24 = vit[(vit["t"] >= 0) & (vit["t"] <= LOOKBACK_H * 60)].groupby("stay_id").size()
    c = c.assign(n_vitals_24h=c["stay_id"].map(first24).fillna(0).astype(int))
    c = c[c["n_vitals_24h"] >= MIN_VITAL_OBS_FIRST24]
    flow.append((f">= {MIN_VITAL_OBS_FIRST24} vital observations in first 24 h", c))

    sidx = SeriesIndex(vit)
    ev = build_events(endpoint, c, sidx)
    u = dict(zip(c["stay_id"], c["unitdischargeoffset"]))
    bad = ev[(ev["onset"] < 0) | (ev["onset"] > ev["stay_id"].map(u) + 1)]["stay_id"].unique()
    c = c[~c["stay_id"].isin(bad)]
    ev = ev[ev["stay_id"].isin(c["stay_id"])]
    flow.append(("plausible event timing", c))

    # a stay 'has an event' if an onset occurs after the first anchor time and before ICU discharge
    has = ev[ev["onset"] > LOOKBACK_H * 60].groupby("stay_id").size()
    c = c.assign(has_event=c["stay_id"].map(has).fillna(0).gt(0).astype(int),
                 pre_icu_h=np.log1p(np.clip(-c["hospitaladmitoffset"].fillna(0) / 60.0, 0, 720)),
                 n_anchors=[len(anchor_hours(x)) for x in c["unitdischargeoffset"]])
    c = c.sort_values("stay_id").reset_index(drop=True)

    # ---- patient-level split (grouped by uniquepid)
    grp = c.groupby("uniquepid").agg(has_event=("has_event", "max")).reset_index()
    train_g, tmp = strat_split(grp, 0.30)
    val_g, test_g = strat_split(tmp, 0.50)
    which = {**{p: "train" for p in train_g["uniquepid"]}, **{p: "validation" for p in val_g["uniquepid"]},
             **{p: "test" for p in test_g["uniquepid"]}}
    split = c[["stay_id", "uniquepid"]].assign(split=c["uniquepid"].map(which))
    assert split["split"].notna().all()
    assert split.groupby("uniquepid")["split"].nunique().max() == 1, "a patient leaked across splits"

    keep = ["stay_id", "uniquepid", "patienthealthsystemstayid", "age_num", "gender", "unittype", "admissionweight",
            "admissionheight", "hospitaladmitoffset", "pre_icu_h", "unitdischargeoffset", "unitdischargestatus",
            "hospitaldischargestatus", "n_vitals_24h", "has_event", "n_anchors"]
    c[keep].to_csv(INTERIM / "cohort.csv", index=False)
    split.to_csv(INTERIM / "split.csv", index=False)

    fl = pd.DataFrame([{"step": n, "n_stays": len(d), "n_patients": d["uniquepid"].nunique() if "uniquepid" in d else np.nan}
                       for n, d in flow])
    fl["excluded_vs_previous"] = (-fl["n_stays"].diff()).fillna(0).astype(int)
    fl.to_csv(RESULTS / "cohort" / "cohort_summary.csv", index=False)

    m = c.merge(split[["stay_id", "split"]], on="stay_id")
    ss = m.groupby("split").agg(n_stays=("stay_id", "size"), n_patients=("uniquepid", "nunique"),
                                n_stays_with_event=("has_event", "sum"), n_candidate_anchors=("n_anchors", "sum"),
                                median_los_h=("unitdischargeoffset", lambda x: float(np.median(x) / 60.0)),
                                icu_mortality=("unitdischargestatus",
                                               lambda x: float((x.astype(str).str.lower() == "expired").mean()))).reset_index()
    ss["event_stay_rate"] = ss["n_stays_with_event"] / ss["n_stays"]
    ss = ss.set_index("split").loc[["train", "validation", "test"]].reset_index()
    ss.to_csv(RESULTS / "cohort" / "split_summary.csv", index=False)
    print(fl.to_string(index=False))
    print("\n" + ss.to_string(index=False))
    if (ss["n_stays_with_event"] < 5).any():
        print("\n  WARNING: fewer than 5 event-stays in some split -- validation/test estimates will be very noisy.")


if __name__ == "__main__":
    main()