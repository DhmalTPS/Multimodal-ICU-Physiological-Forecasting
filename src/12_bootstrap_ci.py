
"""
12_bootstrap_ci.py

ICU Monitor Dilemma
-------------------

6-hour ICU-stay-level bootstrap uncertainty analysis.

This script operates ONLY on the already-generated test predictions.

It does NOT:
    - retrain models
    - modify features
    - modify labels
    - modify train/validation/test splits
    - modify the alarm policy
    - modify the existing prediction files

It computes:

1. 95% cluster-bootstrap CI for 6h AUPRC for:
       logistic
       phys_only
       concat_all
       multimodal_all

2. Paired cluster-bootstrap CI for:
       multimodal_all - phys_only
       multimodal_all - concat_all
       multimodal_all - logistic

Bootstrap unit:
    ICU stay

This is important because hourly anchors from the same ICU stay are
correlated. Resampling individual anchors would incorrectly treat them
as independent observations.

Input files
-----------
results/predictions/baseline_predictions.csv
results/predictions/multimodal_predictions.csv

Actual prediction schema
------------------------
stay_id
anchor_min
y1
y3
y6
next_onset_min
p1h
p3h
p6h
model
split

For the 6-hour analysis:
    label      = y6
    prediction = p6h
    split      = test

Outputs
-------
results/evaluation/bootstrap_ci_6h.csv
results/evaluation/bootstrap_differences_6h.csv
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

BASELINE_PATH = (
    ROOT
    / "results"
    / "predictions"
    / "baseline_predictions.csv"
)

MULTIMODAL_PATH = (
    ROOT
    / "results"
    / "predictions"
    / "multimodal_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "evaluation"
)

CI_OUTPUT_PATH = (
    OUTPUT_DIR
    / "bootstrap_ci_6h.csv"
)

DIFF_OUTPUT_PATH = (
    OUTPUT_DIR
    / "bootstrap_differences_6h.csv"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42
N_BOOT = 2000

STAY_COL = "stay_id"
LABEL_COL = "y6"
PRED_COL = "p6h"
MODEL_COL = "model"
SPLIT_COL = "split"

TEST_SPLIT = "test"

TARGET_MODELS = [
    "logistic",
    "phys_only",
    "concat_all",
    "multimodal_all",
]


# ============================================================================
# BASIC UTILITIES
# ============================================================================

def fail(message):
    """Raise a clear project-specific error."""
    raise ValueError(message)


def check_required_columns(df, required, source_name):
    """
    Verify that all expected columns exist.
    """
    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        fail(
            f"{source_name} is missing required columns:\n"
            f"  Missing: {missing}\n"
            f"  Available: {list(df.columns)}"
        )


def print_schema(name, df):
    """
    Print prediction-file schema.
    """
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(f"Shape: {df.shape}")

    print("Columns:")

    for i, col in enumerate(df.columns):
        print(f"  [{i:02d}] {col}")

    print()

    print("First rows:")

    print(
        df.head().to_string(
            index=False
        )
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_prediction_files():
    """
    Load the two existing prediction files.
    """

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline prediction file not found:\n"
            f"{BASELINE_PATH}"
        )

    if not MULTIMODAL_PATH.exists():
        raise FileNotFoundError(
            f"Multimodal prediction file not found:\n"
            f"{MULTIMODAL_PATH}"
        )

    print()
    print("Loading prediction files...")

    baseline = pd.read_csv(
        BASELINE_PATH
    )

    multimodal = pd.read_csv(
        MULTIMODAL_PATH
    )

    print_schema(
        "BASELINE PREDICTIONS",
        baseline,
    )

    print_schema(
        "MULTIMODAL PREDICTIONS",
        multimodal,
    )

    required_baseline = [
        STAY_COL,
        LABEL_COL,
        PRED_COL,
        MODEL_COL,
        SPLIT_COL,
    ]

    required_multimodal = [
        STAY_COL,
        LABEL_COL,
        PRED_COL,
        MODEL_COL,
        SPLIT_COL,
    ]

    check_required_columns(
        baseline,
        required_baseline,
        "baseline_predictions.csv",
    )

    check_required_columns(
        multimodal,
        required_multimodal,
        "multimodal_predictions.csv",
    )

    return baseline, multimodal


# ============================================================================
# TEST-SET FILTER
# ============================================================================

def select_test_rows(
    df,
    source_name,
):
    """
    Select only the test split.

    The prediction files already contain explicit split information.
    Therefore no horizon filtering is required because y6/p6h directly
    identify the 6-hour task.
    """

    split_values = (
        df[SPLIT_COL]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    out = df.loc[
        split_values == TEST_SPLIT
    ].copy()

    if len(out) == 0:
        fail(
            f"{source_name}: no rows found with "
            f"{SPLIT_COL} == '{TEST_SPLIT}'.\n"
            f"Available split values: "
            f"{sorted(split_values.unique().tolist())}"
        )

    print(
        f"{source_name}: selected "
        f"{len(out):,} test rows."
    )

    return out


# ============================================================================
# MODEL EXTRACTION
# ============================================================================

def select_model(
    df,
    model_name,
    source_name,
):
    """
    Select one model from a long-format prediction dataframe.
    """

    model_values = (
        df[MODEL_COL]
        .astype(str)
        .str.strip()
    )

    out = df.loc[
        model_values == model_name
    ].copy()

    if len(out) == 0:
        available = sorted(
            model_values.unique().tolist()
        )

        fail(
            f"{source_name}: model '{model_name}' "
            f"was not found.\n"
            f"Available models: {available}"
        )

    print(
        f"{source_name}: model={model_name}, "
        f"rows={len(out):,}"
    )

    return out


# ============================================================================
# PREDICTION CLEANING
# ============================================================================

def clean_prediction_frame(
    df,
    model_name,
):
    """
    Keep only valid rows for 6h AUPRC.
    """

    out = df.copy()

    # Convert label to numeric.
    out[LABEL_COL] = pd.to_numeric(
        out[LABEL_COL],
        errors="coerce",
    )

    # Convert prediction to numeric.
    out[PRED_COL] = pd.to_numeric(
        out[PRED_COL],
        errors="coerce",
    )

    # Remove invalid rows.
    out = out.dropna(
        subset=[
            STAY_COL,
            LABEL_COL,
            PRED_COL,
        ]
    ).copy()

    # Labels must be binary.
    unique_labels = set(
        out[LABEL_COL]
        .unique()
        .tolist()
    )

    if not unique_labels.issubset({0, 1}):
        fail(
            f"{model_name}: y6 contains values other than "
            f"0/1: {sorted(unique_labels)}"
        )

    out[LABEL_COL] = (
        out[LABEL_COL]
        .astype(int)
    )

    # Predictions must be finite.
    finite_mask = np.isfinite(
        out[PRED_COL].to_numpy(
            dtype=float
        )
    )

    out = out.loc[
        finite_mask
    ].copy()

    # Basic probability sanity check.
    if (
        (out[PRED_COL] < 0).any()
        or (out[PRED_COL] > 1).any()
    ):
        print(
            f"WARNING: {model_name} contains predictions "
            "outside [0, 1]. AUPRC can still be calculated, "
            "but this is unexpected for probability outputs."
        )

    n_rows = len(out)

    n_positive = int(
        out[LABEL_COL].sum()
    )

    n_stays = (
        out[STAY_COL]
        .nunique()
    )

    prevalence = (
        n_positive / n_rows
        if n_rows > 0
        else np.nan
    )

    print(
        f"{model_name}: "
        f"valid rows={n_rows:,}, "
        f"stays={n_stays:,}, "
        f"positives={n_positive:,}, "
        f"prevalence={prevalence:.6f}"
    )

    if n_rows == 0:
        fail(
            f"{model_name}: no valid rows remain."
        )

    if n_positive == 0:
        fail(
            f"{model_name}: no positive y6 examples."
        )

    if n_positive == n_rows:
        fail(
            f"{model_name}: all y6 examples are positive."
        )

    if n_stays < 10:
        fail(
            f"{model_name}: only {n_stays} ICU stays. "
            "Too few clusters for this bootstrap analysis."
        )

    return out


# ============================================================================
# AUPRC
# ============================================================================

def calculate_auprc(df):
    """
    Calculate average precision, used here as AUPRC.
    """

    y_true = (
        df[LABEL_COL]
        .to_numpy(
            dtype=int
        )
    )

    y_score = (
        df[PRED_COL]
        .to_numpy(
            dtype=float
        )
    )

    return float(
        average_precision_score(
            y_true,
            y_score,
        )
    )


# ============================================================================
# STAY-LEVEL BOOTSTRAP
# ============================================================================

def bootstrap_auprc_by_stay(
    df,
    n_boot=N_BOOT,
    seed=SEED,
):
    """
    Cluster bootstrap of AUPRC with ICU stay as the bootstrap unit.

    Procedure
    ---------
    Let S be the set of unique ICU stays.

    For each bootstrap replicate:
        1. Sample |S| stays from S with replacement.
        2. For every sampled stay, include ALL of its test anchors.
        3. Calculate AUPRC on the resulting bootstrap sample.

    This preserves within-stay temporal dependence.
    """

    rng = np.random.default_rng(
        seed
    )

    # ------------------------------------------------------------
    # Group every test row by ICU stay once.
    # ------------------------------------------------------------

    groups = {
        stay: group
        for stay, group
        in df.groupby(
            STAY_COL,
            sort=False,
        )
    }

    stays = np.asarray(
        list(groups.keys())
    )

    n_stays = len(stays)

    # ------------------------------------------------------------
    # Observed test AUPRC.
    # ------------------------------------------------------------

    observed = calculate_auprc(
        df
    )

    # ------------------------------------------------------------
    # Bootstrap distribution.
    # ------------------------------------------------------------

    bootstrap_scores = np.full(
        n_boot,
        np.nan,
        dtype=float,
    )

    for b in range(n_boot):

        sampled_stays = rng.choice(
            stays,
            size=n_stays,
            replace=True,
        )

        sampled_parts = [
            groups[stay]
            for stay in sampled_stays
        ]

        bootstrap_sample = pd.concat(
            sampled_parts,
            ignore_index=True,
        )

        # A bootstrap replicate can theoretically contain no positives.
        if (
            bootstrap_sample[LABEL_COL]
            .sum()
            == 0
        ):
            continue

        # It is also theoretically possible, though extremely unlikely,
        # for every sampled row to be positive.
        if (
            bootstrap_sample[LABEL_COL]
            .sum()
            == len(bootstrap_sample)
        ):
            continue

        bootstrap_scores[b] = (
            calculate_auprc(
                bootstrap_sample
            )
        )

    valid_scores = (
        bootstrap_scores[
            np.isfinite(
                bootstrap_scores
            )
        ]
    )

    invalid_count = (
        n_boot
        - len(valid_scores)
    )

    if invalid_count > 0:
        print(
            f"  Invalid bootstrap replicates: "
            f"{invalid_count:,}"
        )

    if len(valid_scores) < int(
        0.95 * n_boot
    ):
        fail(
            "More than 5% of bootstrap replicates "
            "were invalid. "
            f"Valid={len(valid_scores):,}, "
            f"requested={n_boot:,}."
        )

    # ------------------------------------------------------------
    # Percentile 95% confidence interval.
    # ------------------------------------------------------------

    ci_lower = float(
        np.percentile(
            valid_scores,
            2.5,
        )
    )

    ci_upper = float(
        np.percentile(
            valid_scores,
            97.5,
        )
    )

    return {
        "auprc": observed,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_boot": n_boot,
        "valid_bootstrap_replicates": len(
            valid_scores
        ),
        "n_stays": n_stays,
        "n_rows": len(df),
        "n_positive": int(
            df[LABEL_COL].sum()
        ),
        "prevalence": float(
            df[LABEL_COL].mean()
        ),
    }


# ============================================================================
# PAIRED STAY-LEVEL BOOTSTRAP
# ============================================================================

def paired_bootstrap_difference(
    df_a,
    df_b,
    model_a,
    model_b,
    n_boot=N_BOOT,
    seed=SEED,
):
    """
    Paired ICU-stay-level bootstrap.

    The SAME sampled ICU stays are used for both models.

    This is important because model A and model B make predictions on the
    same underlying clinical test observations.

    The statistic is:

        Delta = AUPRC(A) - AUPRC(B)

    The confidence interval is obtained from the bootstrap distribution
    of Delta.
    """

    # ------------------------------------------------------------
    # Group both models by stay.
    # ------------------------------------------------------------

    groups_a = {
        stay: group
        for stay, group
        in df_a.groupby(
            STAY_COL,
            sort=False,
        )
    }

    groups_b = {
        stay: group
        for stay, group
        in df_b.groupby(
            STAY_COL,
            sort=False,
        )
    }

    # ------------------------------------------------------------
    # We can only perform a paired comparison on stays present
    # in both prediction sets.
    # ------------------------------------------------------------

    common_stays = sorted(
        set(groups_a.keys())
        & set(groups_b.keys())
    )

    common_stays = np.asarray(
        common_stays
    )

    n_common_stays = len(
        common_stays
    )

    if n_common_stays < 10:
        fail(
            f"Paired comparison {model_a} - {model_b} "
            f"has only {n_common_stays} common ICU stays."
        )

    # ------------------------------------------------------------
    # Observed difference.
    # ------------------------------------------------------------

    # Restrict both datasets to exactly the same common-stay set
    # before computing the observed paired statistic.
    common_a = df_a[
        df_a[STAY_COL].isin(
            common_stays
        )
    ].copy()

    common_b = df_b[
        df_b[STAY_COL].isin(
            common_stays
        )
    ].copy()

    observed_a = calculate_auprc(
        common_a
    )

    observed_b = calculate_auprc(
        common_b
    )

    observed_delta = (
        observed_a
        - observed_b
    )

    # ------------------------------------------------------------
    # Bootstrap.
    # ------------------------------------------------------------

    rng = np.random.default_rng(
        seed
    )

    bootstrap_deltas = np.full(
        n_boot,
        np.nan,
        dtype=float,
    )

    for b in range(n_boot):

        # SAME sampled stays for A and B.
        sampled_stays = rng.choice(
            common_stays,
            size=n_common_stays,
            replace=True,
        )

        sample_a = pd.concat(
            [
                groups_a[stay]
                for stay in sampled_stays
            ],
            ignore_index=True,
        )

        sample_b = pd.concat(
            [
                groups_b[stay]
                for stay in sampled_stays
            ],
            ignore_index=True,
        )

        # Skip degenerate bootstrap samples.
        positives_a = int(
            sample_a[LABEL_COL].sum()
        )

        positives_b = int(
            sample_b[LABEL_COL].sum()
        )

        if (
            positives_a == 0
            or positives_b == 0
        ):
            continue

        if (
            positives_a == len(sample_a)
            or positives_b == len(sample_b)
        ):
            continue

        score_a = calculate_auprc(
            sample_a
        )

        score_b = calculate_auprc(
            sample_b
        )

        if (
            np.isfinite(score_a)
            and np.isfinite(score_b)
        ):
            bootstrap_deltas[b] = (
                score_a
                - score_b
            )

    valid_deltas = (
        bootstrap_deltas[
            np.isfinite(
                bootstrap_deltas
            )
        ]
    )

    invalid_count = (
        n_boot
        - len(valid_deltas)
    )

    if invalid_count > 0:
        print(
            f"  Invalid paired bootstrap replicates: "
            f"{invalid_count:,}"
        )

    if len(valid_deltas) < int(
        0.95 * n_boot
    ):
        fail(
            "More than 5% of paired bootstrap replicates "
            "were invalid. "
            f"Valid={len(valid_deltas):,}, "
            f"requested={n_boot:,}."
        )

    # ------------------------------------------------------------
    # Percentile CI for difference.
    # ------------------------------------------------------------

    ci_lower = float(
        np.percentile(
            valid_deltas,
            2.5,
        )
    )

    ci_upper = float(
        np.percentile(
            valid_deltas,
            97.5,
        )
    )

    return {
        "model_a": model_a,
        "model_b": model_b,
        "auprc_a": observed_a,
        "auprc_b": observed_b,
        "delta_auprc": observed_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_boot": n_boot,
        "valid_bootstrap_replicates": len(
            valid_deltas
        ),
        "n_common_stays": n_common_stays,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 80)
    print("ICU MONITOR DILEMMA")
    print("6-HOUR CLUSTER BOOTSTRAP UNCERTAINTY ANALYSIS")
    print("=" * 80)

    print()
    print(f"Project root : {ROOT}")
    print(f"Bootstrap B  : {N_BOOT}")
    print(f"Random seed  : {SEED}")
    print(f"Label        : {LABEL_COL}")
    print(f"Prediction   : {PRED_COL}")
    print(f"Test split   : {TEST_SPLIT}")
    print(f"Cluster      : {STAY_COL}")

    # ========================================================================
    # 1. LOAD
    # ========================================================================

    baseline, multimodal = (
        load_prediction_files()
    )

    # ========================================================================
    # 2. SELECT TEST SET
    # ========================================================================

    print()
    print("=" * 80)
    print("TEST-SET SELECTION")
    print("=" * 80)

    baseline_test = select_test_rows(
        baseline,
        "baseline_predictions.csv",
    )

    multimodal_test = select_test_rows(
        multimodal,
        "multimodal_predictions.csv",
    )

    # ========================================================================
    # 3. EXTRACT MODELS
    # ========================================================================

    print()
    print("=" * 80)
    print("MODEL EXTRACTION")
    print("=" * 80)

    model_frames = {}

    for model_name in [
        "logistic",
        "phys_only",
        "concat_all",
    ]:

        frame = select_model(
            baseline_test,
            model_name,
            "baseline_predictions.csv",
        )

        frame = clean_prediction_frame(
            frame,
            model_name,
        )

        model_frames[model_name] = (
            frame
        )

    multimodal_frame = select_model(
        multimodal_test,
        "multimodal_all",
        "multimodal_predictions.csv",
    )

    multimodal_frame = clean_prediction_frame(
        multimodal_frame,
        "multimodal_all",
    )

    model_frames[
        "multimodal_all"
    ] = multimodal_frame

    # ========================================================================
    # 4. CHECK TEST-STAY ALIGNMENT
    # ========================================================================

    print()
    print("=" * 80)
    print("TEST-STAY ALIGNMENT")
    print("=" * 80)

    stay_sets = {}

    for model_name, frame in model_frames.items():

        stays = set(
            frame[STAY_COL]
            .unique()
            .tolist()
        )

        stay_sets[model_name] = stays

        print(
            f"{model_name:16s}: "
            f"{len(stays):,} ICU stays"
        )

    reference_model = (
        "multimodal_all"
    )

    reference_stays = stay_sets[
        reference_model
    ]

    for model_name, stays in stay_sets.items():

        if model_name == reference_model:
            continue

        if stays != reference_stays:

            missing_from_model = (
                reference_stays - stays
            )

            extra_in_model = (
                stays - reference_stays
            )

            print()
            print(
                "WARNING: test ICU-stay sets differ."
            )

            print(
                f"Model: {model_name}"
            )

            print(
                f"  Missing stays: "
                f"{len(missing_from_model)}"
            )

            print(
                f"  Extra stays:   "
                f"{len(extra_in_model)}"
            )

            print(
                "Paired comparisons will use only "
                "the common ICU stays."
            )

        else:

            print(
                f"{model_name:16s}: "
                "exactly aligned with multimodal test stays."
            )

    # ========================================================================
    # 5. INDIVIDUAL MODEL BOOTSTRAP CIs
    # ========================================================================

    print()
    print("=" * 80)
    print("STAY-LEVEL BOOTSTRAP AUPRC")
    print("=" * 80)

    ci_results = []

    for model_name in [
        "logistic",
        "phys_only",
        "concat_all",
        "multimodal_all",
    ]:

        frame = model_frames[
            model_name
        ]

        print()
        print(
            f"Bootstrapping {model_name}..."
        )

        result = bootstrap_auprc_by_stay(
            frame,
            n_boot=N_BOOT,
            seed=SEED,
        )

        result["model"] = model_name
        result["horizon_hours"] = 6

        ci_results.append(
            result
        )

        print(
            f"  Observed AUPRC : "
            f"{result['auprc']:.6f}"
        )

        print(
            f"  95% CI         : "
            f"[{result['ci_lower']:.6f}, "
            f"{result['ci_upper']:.6f}]"
        )

        print(
            f"  ICU stays      : "
            f"{result['n_stays']:,}"
        )

        print(
            f"  Positive rows  : "
            f"{result['n_positive']:,}"
        )

        print(
            f"  Prevalence     : "
            f"{result['prevalence']:.6f}"
        )

    ci_df = pd.DataFrame(
        ci_results
    )

    ci_df = ci_df[
        [
            "model",
            "horizon_hours",
            "auprc",
            "ci_lower",
            "ci_upper",
            "n_boot",
            "valid_bootstrap_replicates",
            "n_stays",
            "n_rows",
            "n_positive",
            "prevalence",
        ]
    ]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ci_df.to_csv(
        CI_OUTPUT_PATH,
        index=False,
    )

    # ========================================================================
    # 6. PAIRED MODEL DIFFERENCES
    # ========================================================================

    print()
    print("=" * 80)
    print("PAIRED STAY-LEVEL BOOTSTRAP DIFFERENCES")
    print("=" * 80)

    comparisons = [
        (
            "multimodal_all",
            "phys_only",
        ),
        (
            "multimodal_all",
            "concat_all",
        ),
        (
            "multimodal_all",
            "logistic",
        ),
    ]

    difference_results = []

    for model_a, model_b in comparisons:

        print()
        print(
            f"Comparison: "
            f"{model_a} - {model_b}"
        )

        result = paired_bootstrap_difference(
            df_a=model_frames[
                model_a
            ],
            df_b=model_frames[
                model_b
            ],
            model_a=model_a,
            model_b=model_b,
            n_boot=N_BOOT,
            seed=SEED,
        )

        result[
            "comparison"
        ] = (
            f"{model_a}_minus_{model_b}"
        )

        result[
            "horizon_hours"
        ] = 6

        difference_results.append(
            result
        )

        print(
            f"  AUPRC ({model_a}) : "
            f"{result['auprc_a']:.6f}"
        )

        print(
            f"  AUPRC ({model_b}) : "
            f"{result['auprc_b']:.6f}"
        )

        print(
            f"  Difference        : "
            f"{result['delta_auprc']:+.6f}"
        )

        print(
            f"  95% CI            : "
            f"[{result['ci_lower']:+.6f}, "
            f"{result['ci_upper']:+.6f}]"
        )

        print(
            f"  Common ICU stays  : "
            f"{result['n_common_stays']:,}"
        )

    difference_df = pd.DataFrame(
        difference_results
    )

    difference_df = difference_df[
        [
            "comparison",
            "model_a",
            "model_b",
            "horizon_hours",
            "auprc_a",
            "auprc_b",
            "delta_auprc",
            "ci_lower",
            "ci_upper",
            "n_boot",
            "valid_bootstrap_replicates",
            "n_common_stays",
        ]
    ]

    difference_df.to_csv(
        DIFF_OUTPUT_PATH,
        index=False,
    )

    # ========================================================================
    # 7. HUMAN-READABLE SUMMARY
    # ========================================================================

    print()
    print("=" * 80)
    print("FINAL 6-HOUR AUPRC RESULTS")
    print("=" * 80)

    print()

    for _, row in ci_df.iterrows():

        print(
            f"{row['model']:16s} "
            f"AUPRC={row['auprc']:.4f} "
            f"95% CI="
            f"[{row['ci_lower']:.4f}, "
            f"{row['ci_upper']:.4f}]"
        )

    print()
    print("=" * 80)
    print("FINAL PAIRED DIFFERENCES")
    print("=" * 80)

    print()

    for _, row in difference_df.iterrows():

        print(
            f"{row['model_a']} - "
            f"{row['model_b']}: "
            f"Delta={row['delta_auprc']:+.4f} "
            f"95% CI="
            f"[{row['ci_lower']:+.4f}, "
            f"{row['ci_upper']:+.4f}]"
        )

    # ========================================================================
    # 8. INTERPRETATION FLAGS
    # ========================================================================

    print()
    print("=" * 80)
    print("STATISTICAL INTERPRETATION FLAGS")
    print("=" * 80)

    for _, row in difference_df.iterrows():

        lower = row["ci_lower"]
        upper = row["ci_upper"]

        if lower > 0:
            interpretation = (
                "CI entirely above zero"
            )

        elif upper < 0:
            interpretation = (
                "CI entirely below zero"
            )

        else:
            interpretation = (
                "CI contains zero"
            )

        print(
            f"{row['comparison']:40s}: "
            f"{interpretation}"
        )

    # ========================================================================
    # 9. OUTPUT PATHS
    # ========================================================================

    print()
    print("=" * 80)
    print("OUTPUTS")
    print("=" * 80)

    print()
    print(
        f"Model confidence intervals:\n"
        f"  {CI_OUTPUT_PATH}"
    )

    print()
    print(
        f"Paired model differences:\n"
        f"  {DIFF_OUTPUT_PATH}"
    )

    print()
    print("=" * 80)
    print("BOOTSTRAP ANALYSIS COMPLETE")
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print()
        print(
            "Bootstrap analysis interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 80)
        print("BOOTSTRAP ANALYSIS FAILED")
        print("=" * 80)

        print()
        print(str(exc))

        print()

        sys.exit(1)
