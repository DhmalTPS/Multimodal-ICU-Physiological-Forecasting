"""
run_all.py  --  Sequential execution of the full ICU deterioration forecasting pipeline.

This file contains NO model logic.  It simply executes the numbered scripts in dependency order.
Every script remains individually runnable for debugging partial pipelines.

Usage:
    python run_all.py                        # full pipeline
    python run_all.py --from 06             # restart from feature building
    python run_all.py --dry-run             # print steps without executing
    python run_all.py --skip 08 09 10 11    # skip specific steps (e.g. skip model training)

Exit codes: 0 = success, 1 = a required step failed.
"""

import sys, functools
print = functools.partial(print, flush=True)

import argparse
import subprocess
import sys
import time
from pathlib import Path

PIPELINE = [
    ("00", "src/00_download_check.py",   "RAW DATA INTEGRITY CHECK",         True),
    ("01", "src/01_schema_audit.py",     "SCHEMA / MODALITY AUDIT",          True),
    ("02", "src/02_task_audit.py",       "TASK AUDIT + ENDPOINT FREEZE",     True),
    ("03", "src/03_build_cohort.py",     "COHORT + PATIENT SPLIT",           True),
    ("04", "src/04_build_timeline.py",   "PHYSIOLOGICAL TIMELINE",           True),
    ("05", "src/05_build_labels.py",     "MULTI-HORIZON LABELS",             True),
    ("06", "src/06_build_features.py",   "FEATURE TENSORS",                  True),
    ("07", "src/07_train_baselines.py",  "BASELINE MODELS",                  True),
    ("08", "src/08_train_multimodal.py", "MULTIMODAL MODEL",                 True),
    ("09", "src/09_evaluate.py",         "EVALUATION + ABLATION",            True),
    ("10", "src/10_alarm_analysis.py",   "ALARM CONTROLLER + ROBUSTNESS",    True),
    ("11", "src/11_explain.py",          "EXPLANATIONS + STRESS TESTS",      True),
    ("12", "src/12_bootstrap_ci.py",     "BOOTSTRAP UNCERTAINTY",            True),
]


def run_step(step_id: str, script: str, label: str, required: bool,
             dry_run: bool = False) -> bool:
    """Execute one pipeline step; return True on success."""
    width = 70
    header = f"\n{'='*width}\nSTEP {step_id}  {label}\n{'='*width}"
    print(header)

    if dry_run:
        print(f"  [DRY RUN]  would run: python {script}")
        return True

    if not Path(script).exists():
        msg = f"  MISSING: {script}"
        if required:
            print(msg + "  (required -- aborting pipeline)")
            return False
        else:
            print(msg + "  (optional -- skipping)")
            return True

    t0 = time.time()
    result = subprocess.run([sys.executable, script], check=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        msg = f"  FAILED (exit {result.returncode}) after {elapsed:.1f}s"
        if required:
            print(msg + "  (required step -- aborting pipeline)")
            return False
        else:
            print(msg + "  (optional step -- continuing)")
            return True
    else:
        print(f"  OK  ({elapsed:.1f}s)")
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the full ICU deterioration forecasting pipeline.")
    ap.add_argument("--from", dest="from_step", default="00",
                    help="Start execution from this step number (e.g. --from 06)")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Step numbers to skip (e.g. --skip 08 09)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print steps without executing them")
    args = ap.parse_args()

    start_from = args.from_step.zfill(2)
    skip_set = {s.zfill(2) for s in (args.skip or [])}

    print("\n" + "=" * 70)
    print("  ICU DETERIORATION FORECASTING PIPELINE")
    print(f"  Starting from step: {start_from}")
    if skip_set:
        print(f"  Skipping steps: {skip_set}")
    if args.dry_run:
        print("  DRY RUN -- no scripts will be executed")
    print("=" * 70)

    t_total = time.time()
    for step_id, script, label, required in PIPELINE:
        if step_id < start_from:
            print(f"\n  step {step_id}  {label}  [skipped -- before --from {start_from}]")
            continue
        if step_id in skip_set:
            print(f"\n  step {step_id}  {label}  [skipped -- in --skip list]")
            continue

        success = run_step(step_id, script, label, required, dry_run=args.dry_run)
        if not success:
            print(f"\n  Pipeline aborted at step {step_id}.\n")
            return 1

    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print(f"  Pipeline completed in {elapsed/60:.1f} minutes.")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())