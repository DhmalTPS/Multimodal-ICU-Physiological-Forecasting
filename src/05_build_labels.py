"""
05_build_labels.py  --  PHASE 8: multi-horizon labels (kept separate from feature construction on purpose)

For every hourly anchor t (t >= 24 h, ICU still running):

        y_H(t) = 1  iff an event ONSET lies in (t, t + H]       H in {1, 3, 6} hours

* the label looks only into the future (t, t+H]; features (06_*) look only at <= t;
* anchors inside an ongoing event episode are flagged valid=0 (persistence is not forecasting);
* invariants are asserted: positives have onset strictly after the anchor and within the horizon; anchors are
  strictly before ICU discharge and at least 24 h after ICU admission.

Reads : data/interim/{cohort,split,vital_clean}.csv, results/audit/locked_task.json
Writes: data/interim/events.csv, data/interim/labels.csv, results/cohort/label_summary.csv
"""
import numpy as np
import pandas as pd

from common import (HORIZONS_H, INTERIM, LOOKBACK_H, RESULTS, SeriesIndex, anchor_hours, banner, build_events,
                    ensure_dirs, load_locked_task, load_patient, make_labels)


def main() -> None:
    ensure_dirs()
    banner("05  LABELS")
    task = load_locked_task()
    endpoint = task["endpoint"]
    coh = pd.read_csv(INTERIM / "cohort.csv")
    split = pd.read_csv(INTERIM / "split.csv")
    vit = pd.read_csv(INTERIM / "vital_clean.csv")
    pat = load_patient()
    pat = pat[pat["stay_id"].isin(set(coh["stay_id"]))]
    ev = build_events(endpoint, pat, SeriesIndex(vit))
    ev.to_csv(INTERIM / "events.csv", index=False)
    by = {s: g.sort_values("start") for s, g in ev.groupby("stay_id")}

    frames = []
    for sid, u in zip(coh["stay_id"], coh["unitdischargeoffset"]):
        a = anchor_hours(u)
        if len(a) == 0:
            continue
        g = by.get(sid)
        st, on, en = ((g["start"].to_numpy(), g["onset"].to_numpy(), g["end"].to_numpy())
                      if g is not None else (np.empty(0), np.empty(0), np.empty(0)))
        lab = make_labels(a, st, on, en)
        # ---- invariants ---------------------------------------------------------------------------
        assert lab["anchor_min"].min() >= LOOKBACK_H * 60 and lab["anchor_min"].max() < u
        for h in HORIZONS_H:
            m = lab[f"y{h}"] == 1
            assert np.all(lab["next_onset_min"][m] > lab["anchor_min"][m]), f"stay {sid}: event not after anchor"
            assert np.all(lab["next_onset_min"][m] <= lab["anchor_min"][m] + 60 * h), f"stay {sid}: beyond horizon"
        assert np.all(lab["y6"] >= lab["y3"]) and np.all(lab["y3"] >= lab["y1"]), "horizon nesting violated"
        d = pd.DataFrame({"stay_id": sid, **{k: lab[k] for k in ["anchor_hour", "anchor_min", "y1", "y3", "y6",
                                                                  "next_onset_min", "valid"]}})
        frames.append(d)
    labels = pd.concat(frames, ignore_index=True)
    labels["valid"] = labels["valid"].astype(int)
    labels.to_csv(INTERIM / "labels.csv", index=False)

    m = labels.merge(split[["stay_id", "split"]], on="stay_id")
    rows = []
    for sp in ["train", "validation", "test"]:
        d = m[m["split"] == sp]
        v = d[d["valid"] == 1]
        rec = {"split": sp, "n_anchors": len(d), "n_valid_anchors": len(v),
               "n_excluded_in_episode": len(d) - len(v), "n_stays": d["stay_id"].nunique()}
        for h in HORIZONS_H:
            rec[f"pos_{h}h"] = int(v[f"y{h}"].sum())
            rec[f"prevalence_{h}h"] = float(v[f"y{h}"].mean()) if len(v) else np.nan
        rows.append(rec)
    summ = pd.DataFrame(rows)
    summ.to_csv(RESULTS / "cohort" / "label_summary.csv", index=False)
    print(f"  endpoint: {endpoint}\n  episode types:\n{ev['etype'].value_counts().to_string()}\n")
    print(summ.round(4).to_string(index=False))
    print("\n  all label invariants passed (onset strictly after anchor, inside horizon, anchor < ICU discharge)")


if __name__ == "__main__":
    main()