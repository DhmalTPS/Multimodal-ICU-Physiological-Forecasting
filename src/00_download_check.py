"""
00_download_check.py  --  PHASE 1: do I actually possess the data required to run the project?

Checks (no preprocessing whatsoever):
  1. required raw files exist (patient, vitalPeriodic, vitalAperiodic, lab, note);
  2. compressed files open and parse;
  3. required columns exist;
  4. row counts are non-zero;
  5. raw files were not modified since the previous run (SHA-256 fingerprint; also compared against an
     official SHA256SUMS.txt if you placed one in data/raw/);
  6. dataset version is recorded.

Reads : data/raw/*
Writes: results/audit/raw_integrity.csv, results/audit/raw_checksums.csv, results/audit/dataset_version.txt
"""
import hashlib
import sys
from datetime import datetime

import pandas as pd

from common import RAW, RESULTS, banner, ensure_dirs, find_raw

REQUIRED = {
    "patient": ["patientunitstayid", "unitdischargeoffset", "unitdischargestatus", "hospitaldischargestatus", "age", "gender"],
    "vitalperiodic": ["patientunitstayid", "observationoffset", "heartrate", "respiration", "sao2", "temperature",
                      "systemicsystolic", "systemicdiastolic", "systemicmean"],
    "vitalaperiodic": ["patientunitstayid", "observationoffset", "noninvasivesystolic", "noninvasivediastolic",
                       "noninvasivemean"],
    "lab": ["patientunitstayid", "labresultoffset", "labname", "labresult"],
    "note": ["patientunitstayid", "noteoffset", "noteenteredoffset", "notetype", "notetext"],
}
OPTIONAL = {"nursecharting": ["patientunitstayid"], "nurseassessment": ["patientunitstayid"],
            "physicalexam": ["patientunitstayid"]}


def sha256(path, block=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path, first_col) -> int:
    n = 0
    for chunk in pd.read_csv(path, usecols=lambda c: c.lower() == first_col, chunksize=500_000, low_memory=False):
        n += len(chunk)
    return n


def main() -> int:
    ensure_dirs()
    banner("00  RAW DATA INTEGRITY CHECK")
    prev = RESULTS / "audit" / "raw_checksums.csv"
    prev_hash = dict(pd.read_csv(prev)[["file", "sha256"]].itertuples(index=False)) if prev.exists() else {}
    official = {}
    sums = next(RAW.rglob("SHA256SUMS.txt"), None)
    if sums is not None:
        for line in sums.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                official[parts[1].split("/")[-1]] = parts[0]

    rows, hashes, failed = [], [], False
    for required, tables in ((True, REQUIRED), (False, OPTIONAL)):
        for name, cols in tables.items():
            p = find_raw(name)
            rec = {"table": name, "required": required, "found": p is not None, "opens": False, "columns_ok": False,
                   "missing_columns": "", "n_rows": 0, "size_mb": 0.0, "modified_since_last_run": "n/a",
                   "matches_official_sha256": "n/a", "status": "MISSING"}
            if p is not None:
                rec["size_mb"] = round(p.stat().st_size / 1e6, 2)
                try:
                    head = pd.read_csv(p, nrows=5, low_memory=False)
                    rec["opens"] = True
                    have = {c.lower() for c in head.columns}
                    miss = [c for c in cols if c not in have]
                    if name == "note" and "notetext" in miss and "notevalue" in have:
                        miss.remove("notetext")                      # notevalue is an acceptable fallback
                    rec["missing_columns"] = ",".join(miss)
                    rec["columns_ok"] = not miss
                    rec["n_rows"] = count_rows(p, cols[0])
                    digest = sha256(p)
                    hashes.append({"file": p.name, "sha256": digest})
                    if p.name in prev_hash:
                        rec["modified_since_last_run"] = "YES" if prev_hash[p.name] != digest else "no"
                    if p.name in official:
                        rec["matches_official_sha256"] = "yes" if official[p.name] == digest else "NO"
                    ok = rec["columns_ok"] and rec["n_rows"] > 0 and rec["modified_since_last_run"] != "YES" \
                        and rec["matches_official_sha256"] != "NO"
                    rec["status"] = "OK" if ok else "PROBLEM"
                except Exception as e:                                  # noqa: BLE001
                    rec["status"] = f"UNREADABLE: {e}"
            if required and rec["status"] != "OK":
                failed = True
            rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "audit" / "raw_integrity.csv", index=False)
    pd.DataFrame(hashes).to_csv(RESULTS / "audit" / "raw_checksums.csv", index=False)
    with open(RESULTS / "audit" / "dataset_version.txt", "w") as f:
        f.write("Dataset            : eICU Collaborative Research Database Demo (expected v2.0.1, open access)\n")
        f.write("Source             : https://physionet.org/content/eicu-crd-demo/2.0.1/\n")
        f.write(f"Checked at         : {datetime.now().isoformat(timespec='seconds')}\n")
        f.write("Physiology branch  : periodic/aperiodic vital signs (NOT raw hundreds-of-Hz waveforms)\n\n")
        for h in hashes:
            f.write(f"{h['sha256']}  {h['file']}\n")
    print(out.to_string(index=False))
    if failed:
        print("\nFAILED: a required table is missing/unreadable/incomplete. Download the eICU demo into data/raw/.")
        return 1
    print("\nAll required tables present and readable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())