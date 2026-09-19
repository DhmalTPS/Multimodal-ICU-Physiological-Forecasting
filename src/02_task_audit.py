"""
02_task_audit.py  --  PHASE 3: what prediction task can this dataset support WITHOUT cheating?

For every candidate endpoint we build its episode table, generate the hourly anchors (t >= 24 h, ICU still
running) and apply the feasibility gate:

    sufficient positive events?   reliable timestamp?   enough lead time?   no obvious leakage?

Events are ONSETS.  An anchor that already lies inside an ongoing episode is excluded, so the model has to
forecast the transition to collapse, not detect a collapse that is already on the monitor.

Reads : data/raw/*
Writes: results/audit/task_candidates.csv   one row per endpoint with counts, lead times, gate verdict
        results/audit/locked_task.json      the frozen endpoint used by every later script
Usage : python src/02_task_audit.py [--endpoint composite_collapse]     (override; must still pass the gate)
"""
import argparse

import numpy as np
import pandas as pd

from common import (CANDIDATES, GATE_MIN_IN_ICU_FRACTION, GATE_MIN_MEDIAN_LEAD_H, GATE_MIN_POS_ANCHORS_1H,
                    GATE_MIN_POS_ANCHORS_6H, GATE_MIN_POS_STAYS, HORIZONS_H, HYPOTENSION, HYPOXEMIA,
                    LOOKBACK_H, MERGE_GAP_MIN, RESULTS, SeriesIndex, anchor_hours, banner, build_events,
                    ensure_dirs, load_clean_vitals, load_patient, make_labels, save_json)

DEFINITIONS = {
    "composite_collapse": (f"first onset of: sustained hypotension (MAP<{HYPOTENSION['thr']:.0f} mmHg >= {HYPOTENSION['min_dur']} min) "
                           f"OR sustained hypoxemia (SpO2<{HYPOXEMIA['thr']:.0f}% >= {HYPOXEMIA['min_dur']} min) OR ICU death"),
    "icu_death": "death in the ICU (event time = unitDischargeOffset)",
    "hospital_death": "in-hospital death (event time = hospitalDischargeOffset; often AFTER ICU monitoring ends)",
    "severe_hypotension": f"MAP<{HYPOTENSION['thr']:.0f} mmHg sustained >= {HYPOTENSION['min_dur']} min",
    "severe_hypoxemia": f"SpO2<{HYPOXEMIA['thr']:.0f}% sustained >= {HYPOXEMIA['min_dur']} min",
}
LEAK_RISK = {"composite_collapse": "medium", "icu_death": "low", "hospital_death": "high",
             "severe_hypotension": "medium", "severe_hypoxemia": "medium"}
LEAK_NOTE = {"medium": "label derived from monitored vitals; mitigated by onset-only labels + in-episode anchor exclusion",
             "low": "", "high": "event time can fall outside the ICU observation window"}


def audit_endpoint(endpoint, pat, sidx):
    ev = build_events(endpoint, pat, sidx)
    by_stay = {s: g.sort_values("start") for s, g in ev.groupby("stay_id")}
    uoff = dict(zip(pat["stay_id"], pat["unitdischargeoffset"]))
    n_valid = 0
    pos = {h: 0 for h in HORIZONS_H}
    pos_stays = set()
    lead, n_events_after_24h, n_in_icu = [], 0, 0
    for sid, u in uoff.items():
        a = anchor_hours(u)
        g = by_stay.get(sid)
        if g is not None:
            n_in_icu += int((g["onset"] <= u + 1).sum())
            lead += [(o - LOOKBACK_H * 60) / 60.0 for o in g["onset"] if o > LOOKBACK_H * 60 and (o <= u + 1)]
        if len(a) == 0:
            continue
        st, on, en = ((g["start"].to_numpy(), g["onset"].to_numpy(), g["end"].to_numpy())
                      if g is not None else (np.empty(0), np.empty(0), np.empty(0)))
        lab = make_labels(a, st, on, en)
        v = lab["valid"]
        n_valid += int(v.sum())
        for h in HORIZONS_H:
            pos[h] += int((lab[f"y{h}"][v] == 1).sum())
        if (lab["y6"][v] == 1).any():
            pos_stays.add(sid)
        # ---- invariant: positives are strictly in the future and inside the horizon
        for h in HORIZONS_H:
            m = v & (lab[f"y{h}"] == 1)
            assert np.all(lab["next_onset_min"][m] > lab["anchor_min"][m]), "event not strictly after anchor"
            assert np.all(lab["next_onset_min"][m] <= lab["anchor_min"][m] + h * 60), "event beyond horizon"
    n_ev = len(ev)
    lead = np.asarray(lead)
    rec = {
        "endpoint": endpoint, "definition": DEFINITIONS[endpoint], "n_episodes_total": n_ev,
        "n_events_after_24h": int(len(lead)), "n_valid_anchors": n_valid,
        **{f"pos_anchors_{h}h": pos[h] for h in HORIZONS_H},
        **{f"prevalence_{h}h": pos[h] / max(n_valid, 1) for h in HORIZONS_H},
        "n_positive_stays_6h": len(pos_stays),
        "frac_events_inside_icu_window": n_in_icu / max(n_ev, 1),
        "lead_h_q25": float(np.percentile(lead, 25)) if len(lead) else np.nan,
        "lead_h_median": float(np.median(lead)) if len(lead) else np.nan,
        "lead_h_q75": float(np.percentile(lead, 75)) if len(lead) else np.nan,
        "leakage_risk": LEAK_RISK[endpoint],
    }
    reasons = []
    if rec["n_positive_stays_6h"] < GATE_MIN_POS_STAYS:
        reasons.append(f"positive stays {rec['n_positive_stays_6h']} < {GATE_MIN_POS_STAYS}")
    if rec["pos_anchors_6h"] < GATE_MIN_POS_ANCHORS_6H:
        reasons.append(f"6h positive anchors {rec['pos_anchors_6h']} < {GATE_MIN_POS_ANCHORS_6H}")
    if rec["pos_anchors_1h"] < GATE_MIN_POS_ANCHORS_1H:
        reasons.append(f"1h positive anchors {rec['pos_anchors_1h']} < {GATE_MIN_POS_ANCHORS_1H}")
    if rec["frac_events_inside_icu_window"] < GATE_MIN_IN_ICU_FRACTION:
        reasons.append(f"only {rec['frac_events_inside_icu_window']:.0%} of events inside ICU window (timestamp unreliable)")
    if not np.isfinite(rec["lead_h_median"]) or rec["lead_h_median"] < GATE_MIN_MEDIAN_LEAD_H:
        reasons.append(f"median lead from first anchor {rec['lead_h_median']:.1f} h < {GATE_MIN_MEDIAN_LEAD_H} h")
    rec["gate_failures"] = "; ".join(reasons)
    rec["status"] = "keep" if not reasons else "reject"
    rec["leakage_note"] = LEAK_NOTE[rec["leakage_risk"]]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", choices=CANDIDATES, default=None, help="override the automatic choice")
    args = ap.parse_args()
    ensure_dirs()
    banner("02  TASK AUDIT  (freeze the prediction problem)")
    pat = load_patient()
    vit = load_clean_vitals(set(pat["stay_id"]))
    sidx = SeriesIndex(vit)
    print(f"  stays: {len(pat):,}   vital observations: {len(vit):,}")

    recs = []
    for ep in CANDIDATES:
        r = audit_endpoint(ep, pat, sidx)
        recs.append(r)
        print(f"  {ep:<20} anchors={r['n_valid_anchors']:>7,}  pos(1h/3h/6h)="
              f"{r['pos_anchors_1h']:>6,}/{r['pos_anchors_3h']:>6,}/{r['pos_anchors_6h']:>6,}  "
              f"pos-stays={r['n_positive_stays_6h']:>4}  -> {r['status']}  {r['gate_failures']}")
    df = pd.DataFrame(recs)
    df.to_csv(RESULTS / "audit" / "task_candidates.csv", index=False)

    keep = df[df["status"] == "keep"]["endpoint"].tolist()
    if args.endpoint:
        choice = args.endpoint
        if choice not in keep:
            print(f"  WARNING: override '{choice}' FAILS the feasibility gate: "
                  f"{df.set_index('endpoint').loc[choice, 'gate_failures']}")
    elif keep:
        choice = keep[0]                       # CANDIDATES is already in priority order
    else:
        raise SystemExit("No candidate endpoint passes the feasibility gate. Inspect results/audit/task_candidates.csv.")
    row = df.set_index("endpoint").loc[choice]
    save_json({
        "endpoint": choice, "definition": DEFINITIONS[choice], "horizons_h": list(HORIZONS_H),
        "lookback_h": LOOKBACK_H, "grid_min": 60, "merge_gap_min": MERGE_GAP_MIN,
        "hypotension": HYPOTENSION, "hypoxemia": HYPOXEMIA,
        "counts": {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in
                   ["n_valid_anchors", "pos_anchors_1h", "pos_anchors_3h", "pos_anchors_6h", "n_positive_stays_6h"]},
        "rule": "anchors inside an ongoing episode are excluded; labels are event ONSETS in (t, t+H]",
    }, RESULTS / "audit" / "locked_task.json")
    print(f"\n  >>> LOCKED ENDPOINT: {choice}\n      {DEFINITIONS[choice]}")
    print("      (this decision must not change casually -- all later stages read results/audit/locked_task.json)")


if __name__ == "__main__":
    main()