"""
01_schema_audit.py  --  PHASE 2: what is actually in the tables?  Is multimodality feasible?

Reads : data/raw/*
Writes: results/audit/schema_summary.csv        rows / columns / patients / time range per table
        results/audit/missingness.csv           % missing per column per table
        results/audit/modality_coverage.csv     per ICU stay: which modalities exist and how many observations
        results/audit/modality_combinations.csv how many stays have which modality combination
        results/audit/lab_names.csv             laboratory variables ranked by coverage
        results/audit/note_documentation_delay.csv   noteEnteredOffset - noteOffset (leakage-relevant lag)

Key eICU facts checked here: all offsets are minutes from ICU admission; `noteoffset` is the clinical time of
a note but `noteenteredoffset` is when it was DOCUMENTED -- only the latter is usable as availability time.
"""
import numpy as np
import pandas as pd

from common import RESULTS, banner, ensure_dirs, find_raw, load_patient

# table -> time columns
TABLES = {
    "patient": ["unitdischargeoffset", "hospitaldischargeoffset", "hospitaladmitoffset"],
    "vitalperiodic": ["observationoffset"],
    "vitalaperiodic": ["observationoffset"],
    "lab": ["labresultoffset"],
    "note": ["noteoffset", "noteenteredoffset"],
    "nursecharting": ["nursingchartoffset"],
    "nurseassessment": ["nurseassessoffset"],
    "physicalexam": ["physicalexamoffset"],
}
# tables that define a modality and the time column defining *availability*
MODALITY_TABLES = {"vitalperiodic": "observationoffset", "vitalaperiodic": "observationoffset",
                   "lab": "labresultoffset", "note": "noteenteredoffset"}


def scan(name: str, timecols):
    p = find_raw(name)
    if p is None:
        return None
    rows, nulls, stays, tmin, tmax, cols = 0, None, set(), {}, {}, []
    per_stay, per_stay24 = [], []
    for ch in pd.read_csv(p, chunksize=500_000, low_memory=False):
        ch.columns = [c.lower() for c in ch.columns]
        cols = list(ch.columns)
        rows += len(ch)
        nn = ch.isna().sum()
        nulls = nn if nulls is None else nulls.add(nn, fill_value=0)
        if "patientunitstayid" in ch:
            stays.update(ch["patientunitstayid"].dropna().unique().tolist())
        for tc in timecols:
            if tc in ch:
                v = pd.to_numeric(ch[tc], errors="coerce")
                if v.notna().any():
                    tmin[tc] = min(tmin.get(tc, np.inf), v.min())
                    tmax[tc] = max(tmax.get(tc, -np.inf), v.max())
        if name in MODALITY_TABLES and MODALITY_TABLES[name] in ch:
            v = pd.to_numeric(ch[MODALITY_TABLES[name]], errors="coerce")
            per_stay.append(ch.groupby("patientunitstayid").size())
            per_stay24.append(ch[(v <= 1440)].groupby("patientunitstayid").size())
    ps = pd.concat(per_stay).groupby(level=0).sum() if per_stay else None
    ps24 = pd.concat(per_stay24).groupby(level=0).sum() if per_stay24 else None
    return dict(rows=rows, cols=cols, nulls=nulls, stays=stays, tmin=tmin, tmax=tmax, per_stay=ps, per_stay24=ps24)


def main() -> None:
    ensure_dirs()
    banner("01  SCHEMA / MISSINGNESS / MODALITY COVERAGE AUDIT")
    schema, miss, per_stay = [], [], {}
    for name, tcols in TABLES.items():
        r = scan(name, tcols)
        if r is None:
            schema.append({"table": name, "present": False})
            continue
        print(f"  {name:<16} rows={r['rows']:>10,}  columns={len(r['cols']):>3}  stays={len(r['stays']):>6,}")
        rec = {"table": name, "present": True, "rows": r["rows"], "columns": len(r["cols"]),
               "patients": len(r["stays"]), "column_names": "|".join(r["cols"])}
        for tc in tcols:
            if tc in r["tmin"]:
                rec[f"{tc}_min"], rec[f"{tc}_max"] = r["tmin"][tc], r["tmax"][tc]
        schema.append(rec)
        for c, n in r["nulls"].items():
            miss.append({"table": name, "column": c, "n_missing": int(n),
                         "pct_missing": round(100.0 * n / max(r["rows"], 1), 3)})
        if name in MODALITY_TABLES:
            per_stay[name] = (r["per_stay"], r["per_stay24"])
    pd.DataFrame(schema).to_csv(RESULTS / "audit" / "schema_summary.csv", index=False)
    pd.DataFrame(miss).to_csv(RESULTS / "audit" / "missingness.csv", index=False)

    # ---- modality coverage per stay
    pat = load_patient()
    cov = pd.DataFrame({"stay_id": pat["stay_id"].to_numpy()}).set_index("stay_id")

    def col(name, k):
        s = per_stay.get(name, (None, None))[k]
        return s.reindex(cov.index).fillna(0).astype(int) if s is not None else 0

    cov["n_vitals"] = col("vitalperiodic", 0) + col("vitalaperiodic", 0)
    cov["n_labs"] = col("lab", 0)
    cov["n_notes"] = col("note", 0)
    cov["n_vitals_first24h"] = col("vitalperiodic", 1) + col("vitalaperiodic", 1)
    cov["n_labs_first24h"] = col("lab", 1)
    cov["n_notes_first24h"] = col("note", 1)
    cov["vital_available"] = (cov["n_vitals"] > 0).astype(int)
    cov["lab_available"] = (cov["n_labs"] > 0).astype(int)
    cov["note_available"] = (cov["n_notes"] > 0).astype(int)
    cov = cov.reset_index()[["stay_id", "vital_available", "lab_available", "note_available", "n_vitals", "n_labs",
                             "n_notes", "n_vitals_first24h", "n_labs_first24h", "n_notes_first24h"]]
    cov.to_csv(RESULTS / "audit" / "modality_coverage.csv", index=False)
    combo = (cov.assign(combo=lambda d: d.vital_available.map({1: "V", 0: "-"}) + d.lab_available.map({1: "L", 0: "-"})
                        + d.note_available.map({1: "N", 0: "-"}))
             .groupby("combo").size().rename("n_stays").reset_index())
    combo["pct"] = (100 * combo["n_stays"] / combo["n_stays"].sum()).round(2)
    combo.to_csv(RESULTS / "audit" / "modality_combinations.csv", index=False)
    print("\nModality combinations (V=vitals, L=labs, N=notes):\n" + combo.to_string(index=False))
    print(f"\nStays with all three modalities: {(cov[['vital_available','lab_available','note_available']].sum(1) == 3).mean():.1%}")

    # ---- lab variables ranking
    if find_raw("lab") is not None:
        from common import read_table
        lab = read_table("lab", usecols=["patientunitstayid", "labname", "labresult"])
        lab["labresult"] = pd.to_numeric(lab["labresult"], errors="coerce")
        lab = lab.dropna(subset=["labresult"])
        ln = (lab.groupby("labname").agg(n_results=("labresult", "size"), n_stays=("patientunitstayid", "nunique"))
              .sort_values("n_stays", ascending=False).reset_index())
        ln.to_csv(RESULTS / "audit" / "lab_names.csv", index=False)
        print("\nTop laboratory variables by stay coverage:\n" + ln.head(12).to_string(index=False))

    # ---- documentation delay (leakage-relevant)
    if find_raw("note") is not None:
        from common import read_table
        nt = read_table("note", usecols=["noteoffset", "noteenteredoffset"])
        d = (pd.to_numeric(nt["noteenteredoffset"], errors="coerce") - pd.to_numeric(nt["noteoffset"], errors="coerce")).dropna()
        if len(d):
            q = d.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).rename_axis("quantile").reset_index(name="delay_min")
            q.loc[len(q)] = ["frac_entered_after_event", float((d > 0).mean())]
            q.to_csv(RESULTS / "audit" / "note_documentation_delay.csv", index=False)
            print(f"\nNotes documented AFTER their clinical time: {(d > 0).mean():.1%} "
                  f"(median lag {d.median():.0f} min) -> availability time = noteenteredoffset, not noteoffset")
    print("\nWrote results/audit/{schema_summary,missingness,modality_coverage,modality_combinations,lab_names}.csv")


if __name__ == "__main__":
    main()