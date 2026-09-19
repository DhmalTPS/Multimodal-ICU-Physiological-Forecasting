"""
04_build_timeline.py  --  PHASE 5: the physiological branch

* converts vitalPeriodic (5-min) + vitalAperiodic (non-invasive BP) to numeric,
* removes physiologically impossible values (kept as MISSING -- nothing is interpolated),
* keeps every timestamp,
* aggregates to the hourly grid: for each stay and hour g, mean/min/max/count over the bin (60(g-1), 60g].
  The bin (60(g-1), 60g] is fully known at time 60*g -> row g is causal for an anchor at 60*g.

Honest note: the public eICU demo has periodic/aperiodic vital measurements, not raw hundreds-of-Hz waveforms.
This branch is a "high-frequency physiological time-series" encoder; the encoder interface (TCN over hourly
rows) can later be swapped for a raw-waveform front-end without touching the downstream fusion.

Reads : cohort.csv + raw vitals
Writes: data/interim/vital_clean.csv   long cleaned observations of cohort stays
        data/interim/timeline.csv      hourly per-stay table
        results/features/vital_summary.csv
"""
import numpy as np
import pandas as pd

from common import INTERIM, PHYS_CHANNELS, PLAUSIBLE, RESULTS, anchor_hours, banner, ensure_dirs, load_clean_vitals
from features import hourly_phys_table


def main() -> None:
    ensure_dirs()
    banner("04  PHYSIOLOGICAL TIMELINE")
    coh = pd.read_csv(INTERIM / "cohort.csv")
    stays = set(coh["stay_id"])
    vit = load_clean_vitals(stays)
    u = dict(zip(coh["stay_id"], coh["unitdischargeoffset"]))
    a_max = {s: int(anchor_hours(x)[-1]) for s, x in u.items()}

    vit = vit[vit["t"] <= vit["stay_id"].map(u) + 1].reset_index(drop=True)        # nothing after ICU discharge
    vit.to_csv(INTERIM / "vital_clean.csv", index=False)

    tl_in = vit[vit["t"] <= 60.0 * vit["stay_id"].map(a_max)]                       # nothing after the last anchor
    hourly = hourly_phys_table(tl_in)
    hourly = hourly[hourly["hour"] <= hourly["stay_id"].map(a_max)]
    hourly.to_csv(INTERIM / "timeline.csv", index=False)

    n_rows = sum(a_max.values()) + 0
    rows = []
    for ch in PLAUSIBLE:
        x = vit[ch].dropna().to_numpy()
        obs_hours = int(((hourly[f"{ch}_count"] > 0) & (hourly["hour"] >= 1)).sum())
        rows.append({"channel": ch, "n_observations": len(x),
                     "pct_stays_with_channel": 100 * hourly.loc[hourly[f"{ch}_count"] > 0, "stay_id"].nunique() / len(stays),
                     "pct_stay_hours_observed": 100 * obs_hours / max(n_rows, 1),
                     "mean": float(x.mean()) if len(x) else np.nan, "sd": float(x.std()) if len(x) else np.nan,
                     "p01": float(np.percentile(x, 1)) if len(x) else np.nan,
                     "p99": float(np.percentile(x, 99)) if len(x) else np.nan,
                     "plausible_range": f"{PLAUSIBLE[ch][0]}-{PLAUSIBLE[ch][1]}"})
    summ = pd.DataFrame(rows)
    summ.to_csv(RESULTS / "features" / "vital_summary.csv", index=False)
    print(f"  cohort stays: {len(stays):,}   cleaned observations: {len(vit):,}   hourly rows: {len(hourly):,}")
    print(summ.round(2).to_string(index=False))
    print("\n  wrote data/interim/{vital_clean,timeline}.csv, results/features/vital_summary.csv")


if __name__ == "__main__":
    main()