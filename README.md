# The ICU Monitor Dilemma

## Leakage-Controlled Multimodal Forecasting of ICU Physiological Deterioration

<!-- > **A note on math in this file.** GitHub's Markdown renderer supports LaTeX math written as `$...$` (inline) and `$$...$$` (display, each on its own line, with no blank line inside the block). It does **not** render the `\( ... \)` / `\[ ... \]` delimiter style — those show up as literal backslash-parenthesis text. Every equation below uses the `$` / `$$` form for this reason. -->

---

## 1. Problem

Intensive-care units continuously generate heterogeneous clinical data from bedside monitors, laboratory systems, and clinical documentation.

A conventional ICU monitor primarily reacts to threshold violations that have already occurred. This creates a fundamental operational problem: transient artifacts, sensor failures, physiological variability, and documentation delays can generate large numbers of alarms, while clinically meaningful deterioration may be difficult to distinguish from background noise.

The objective of this project is to construct a research prototype for **early forecasting of clinically meaningful physiological deterioration** from heterogeneous ICU data.

The system considers four information sources at time $t$:

$$
\mathcal X(t) = \left[ X^{(p)}(t),\ X^{(l)}(t),\ X^{(n)}(t),\ X^{(s)} \right]
$$

where $X^{(p)}(t)$ is physiological observations, $X^{(l)}(t)$ is laboratory observations, $X^{(n)}(t)$ is clinical notes, and $X^{(s)}$ is static stay-level information (age, sex, admission weight/height, ICU unit type).

The forecasting objective is

$$
p_h(t) = \Pr\big(Y_h(t)=1 \mid \mathcal F(t)\big)
$$

where $t$ is the prediction anchor, $h$ is a future prediction horizon, $Y_h(t)$ indicates whether deterioration occurs within that horizon, and $\mathcal F(t)$ denotes the information actually available at prediction time (not the full retrospective record).

The project deliberately separates three concerns that are easy to conflate:

$$
\text{data construction} \ \longrightarrow\ \text{risk prediction} \ \longrightarrow\ \text{alarm generation}
$$

rather than treating a high predicted risk and a clinical alarm as the same object.

---

## 2. Dataset

The experiments use the freely accessible **eICU Collaborative Research Database Demo**.

The project uses five core source tables — `patient`, `vitalPeriodic`, `vitalAperiodic`, `lab`, `note` — and audits three additional tables (`nurseCharting`, `nurseAssessment`, `physicalExam`) for schema and coverage information without depending on them.

### Raw-data audit

| Table | Rows | Required | Readable |
|---|---:|:---:|:---:|
| `patient` | 2,520 | Yes | Yes |
| `vitalPeriodic` | 1,634,960 | Yes | Yes |
| `vitalAperiodic` | 274,088 | Yes | Yes |
| `lab` | 434,660 | Yes | Yes |
| `note` | 24,758 | Yes | Yes |
| `nurseCharting` | 1,477,163 | No | Yes |
| `nurseAssessment` | 91,589 | No | Yes |
| `physicalExam` | 84,058 | No | Yes |

The raw-data integrity stage (`00_download_check.py`) verifies that required files exist, are readable, contain the expected columns, have non-zero row counts, and match official SHA-256 checksums.

Raw data are treated as immutable source data. All intermediate and generated outputs are constructed separately under `data/interim/`, `data/processed/`, `results/`, `figures/`, and `models/` — nothing is ever written back into `data/raw/`.

### Modality coverage

Among the 2,520 raw ICU stays:

| Available modalities | Stays | Percentage |
|---|---:|---:|
| None | 36 | 1.43% |
| Notes only | 9 | 0.36% |
| Labs only | 42 | 1.67% |
| Labs + notes | 53 | 2.10% |
| Vitals only | 13 | 0.52% |
| Vitals + notes | 18 | 0.71% |
| Vitals + labs | 172 | 6.83% |
| Vitals + labs + notes | 2,177 | 86.39% |

Although the modalities are heterogeneous and incomplete for a minority of stays, the large majority (86.4%) contain all three core dynamic modalities.

---

## 3. Research Questions

The central research question is:

> **Can a leakage-controlled temporal model use heterogeneous ICU physiology, laboratory measurements, clinical notes, and static information to forecast clinically meaningful deterioration several hours before it occurs, while remaining robust to missing or delayed modalities and producing an operationally interpretable alarm process?**

This decomposes into six narrower questions:

| # | Question |
|---|---|
| RQ1 | Does the available ICU data contain useful predictive information at all? |
| RQ2 | Which information source (physiology vs. labs vs. text) contributes the dominant predictive signal? |
| RQ3 | Does explicit multimodal fusion provide incremental value over statistical baselines, single-modality models, and naive concatenation? |
| RQ4 | How does performance change when one or more modalities are absent, delayed, or partially observed? |
| RQ5 | How does continuous risk translate into an alarm stream, and what is the sensitivity/warning-time/alarm-burden trade-off? |
| RQ6 | How robust is the system to physiological measurement noise and sensor blackout? |

---

## 4. Prediction Target

The endpoint was audited across five candidates (see `report/tables.md`, Table 3) before being frozen. The selected endpoint is **`composite_collapse`**: the first occurrence of sustained hypotension, sustained hypoxemia, or ICU death.

- **Hypotension** — the first sustained episode with $\text{MAP} < 55$ mmHg for at least 30 minutes.
- **Hypoxemia** — the first sustained episode with $\text{SpO}_2 < 88\%$ for at least 30 minutes.
- **ICU death** — death occurring during the ICU stay.

The composite event time is

$$
T^{\text{collapse}} = \min\big(T^{\text{hypotension}},\ T^{\text{hypoxemia}},\ T^{\text{death}}\big)
$$

and, for anchor $t$ and horizon $h$, the label is

$$
Y_h(t) = \mathbf{1}\big[\, t < T^{\text{collapse}} \le t+h \,\big], \qquad h \in \{1,3,6\}\ \text{hours}.
$$

An anchor that falls inside an already-active deterioration episode is excluded from prediction (the task is onset *forecasting*, not persistence detection). Every label is checked against three invariants: $T^{\text{collapse}} > t$, $T^{\text{collapse}} \le t+h$, and $t < T^{\text{ICU discharge}}$.

### Endpoint audit

| Candidate endpoint | 1h positives | 3h positives | 6h positives | Decision |
|---|---:|---:|---:|---|
| Composite collapse | 292 | 1,259 | 2,457 | **Selected** |
| ICU death | 74 | 216 | 418 | Retained as candidate |
| Hospital death | 51 | 167 | 371 | Rejected |
| Severe hypotension | 116 | 545 | 1,093 | Retained as candidate |
| Severe hypoxemia | 130 | 607 | 1,192 | Retained as candidate |

Hospital death was rejected as the primary endpoint because only 43% of hospital-death events occurred within the ICU observation window — the event timestamp is unreliable relative to the physiological record used to build $\mathcal F(t)$.

---

## 5. Temporal Setup

Physiological measurements arrive at high frequency, laboratory measurements are sparse and irregular, and notes are event-driven. The pipeline therefore constructs an **hourly prediction grid**.

For ICU stay $i$, define anchor times $t_{i,k} = t_{i,0} + 60k$ minutes. At anchor $t_{i,k}$, the feature constructor may use only information with timestamp $\le t_{i,k}$ — formally, the information filtration

$$
\mathcal F_{i,k} = \sigma\big( X_i(\tau) : \tau \le t_{i,k} \big),
$$

and the prediction is $\hat p_{i,k,h} = f_\theta(\mathcal F_{i,k})_h$ for $h \in \{1,3,6\}$ hours.

### Final cohort

Cohort construction begins with 2,520 ICU stays belonging to 1,841 patients. After eligibility filtering, the final cohort contains **1,565 ICU stays** belonging to **1,294 patients**.

| Stage | Stays | Patients |
|---|---:|---:|
| All ICU stays | 2,520 | 1,841 |
| Valid ID + ICU discharge time | 2,520 | 1,841 |
| ICU LOS > 24 h | 1,626 | 1,332 |
| Known ICU discharge status | 1,626 | 1,332 |
| Plausible timeline | 1,622 | 1,328 |
| ≥ 12 vital observations in first 24 h | 1,566 | 1,294 |
| Plausible event timing | **1,565** | **1,294** |

### Patient-level split

The train/validation/test partition is performed at the **patient** level (70/15/15, stratified on "ever has an event"), so no patient's ICU stays are split across partitions:

| Split | Stays | Patients | Event stays |
|---|---:|---:|---:|
| Train | 1,086 | 905 | 182 |
| Validation | 234 | 194 | 40 |
| Test | 245 | 195 | 40 |

---

## 6. Multimodal Architecture

Each clinical modality is treated as a distinct information-generating process. At hourly index $k$: $P_k \in \mathbb R^{35}$ (physiology), $L_k \in \mathbb R^{80}$ (laboratory), $N_k \in \mathbb R^{66}$ (text), and static information $S \in \mathbb R^{15}$. The dynamic multimodal representation therefore has $35+80+66=181$ features per hourly row.

```text
 Physiological data ──▶ Physiology encoder (causal TCN) ──▶ H⁽ᵖ⁾
 Laboratory data     ──▶ Lab encoder (MLP)                ──▶ H⁽ˡ⁾
 Clinical notes       ──▶ Text encoder (frozen embed. + proj.) ──▶ H⁽ⁿ⁾
 Static information   ──▶ Static projection                    ──▶ S'
                                    │
                                    ▼
                    Causal cross-modal attention + reliability gate
                                    │
                                    ▼
                          GRU longitudinal state
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
                   1 h             3 h             6 h  risk
```

Each modality is transformed into a learned representation, $H^{(p)} = f_p(P_{1:T})$, $H^{(l)} = f_l(L_{1:T})$, $H^{(n)} = f_n(N_{1:T})$, and fused as

$$
Z = f_{\text{fusion}}\big(H^{(p)}, H^{(l)}, H^{(n)}, S\big),
$$

with the final risk vector $\hat{\mathbf p} = (\hat p_1, \hat p_3, \hat p_6)$. The implemented multimodal model contains approximately **172,070 trainable parameters**. The full architecture — causal temporal convolutions, causal cross-modal attention with a missingness-aware reliability gate, and the GRU head — is defined exactly in `src/nets.py` and derived mathematically in `report/whitepaper.md`.

---

## 7. Missingness Handling

Missingness is treated as part of the clinical data-generating process, not as a nuisance to be imputed away.

- **Physiology** (heart rate, respiratory rate, SpO₂, temperature, systolic/diastolic BP, MAP) is comparatively dense: $d_p = 35$.
- **Laboratory** data are irregular and sparse: 16 variables × 5 temporal summary statistics each, $d_l = 16 \times 5 = 80$.
- **Clinical notes** are highly event-driven: character TF-IDF → SVD → $\mathbb R^{64}$, plus freshness and count, $d_t = 66$. Only about 2% of hourly rows contain a newly available note.

The trained system is explicitly evaluated under modality perturbation: full removal of a modality, combined modality removal, 20/40/60% laboratory dropout, and 60/180-minute note-availability delay. The principal finding is that removing physiology causes by far the largest performance degradation, indicating that physiological observations carry the dominant predictive signal in this cohort (see `report/tables.md`, Table 14).

---

## 8. Leakage Prevention

Leakage prevention is a first-class component of the project, not an afterthought:

1. **Patient-level splitting** — the train/validation/test split is performed by patient, not by hourly observation.
2. **Temporal feature truncation** — at anchor $t$, only observations with $\tau \le t$ may contribute to features.
3. **Clinical-note availability** — 98% of notes were documented after their clinical timestamp (median delay ≈ 12 minutes), so note availability is defined by the *documentation* time $t^{\text{entry}}$, never the clinical event time: a note contributes to an anchor only if $t^{\text{entry}} \le t$.
4. **Training-only representation fitting** — the TF-IDF/SVD text transform and every feature scaler are fit on training-split data only.
5. **Endpoint freezing** — candidate endpoints are audited before downstream modeling; the selected endpoint is frozen in `results/audit/locked_task.json` and every later stage reads it rather than re-deciding.
6. **Label causality** — event onset must be strictly after the anchor and within the forecast horizon.
7. **Truncation tests** — `06_build_features.py` rebuilds randomly chosen feature rows from data truncated at that row's own anchor time and asserts the result is numerically identical to the value computed from the full dataset, i.e. it empirically verifies

$$
\Phi\big(X_{\le t}\big) \ \text{does not depend on}\ X_{>t}.
$$

---

## 9. Model Hierarchy

| Model | Description |
|---|---|
| Logistic regression | Flat baseline, $\hat p = \sigma(w^\top x + b)$, collapses temporal shape |
| Physiology-only | ConcatGRU using only $P_{1:T}$ |
| Laboratory-only | ConcatGRU using only $L_{1:T}$ |
| Text-only | ConcatGRU using only $N_{1:T}$ |
| Naive concatenation | ConcatGRU on $[P_k; L_k; N_k]$, shared temporal model |
| **Proposed multimodal** | Separate modality encoders, causal cross-attention, missingness-aware reliability gate, GRU |

This hierarchy exists because a complex multimodal architecture is only scientifically useful when compared against simpler explanations of the same predictive signal.

---

## 10. Evaluation

The primary discrimination metric is **area under the precision-recall curve (AUPRC)**:

$$
\text{Precision}(\tau) = \frac{TP(\tau)}{TP(\tau)+FP(\tau)}, \qquad \text{Recall}(\tau) = \frac{TP(\tau)}{TP(\tau)+FN(\tau)},
$$

with AUPRC the area under the resulting precision–recall curve. This is preferred over AUROC for a highly imbalanced task: at the 6-hour horizon, test prevalence is $\pi = 322/12485 \approx 2.58\%$.

### Test AUPRC

| Model | 1h | 3h | 6h |
|---|---:|---:|---:|
| Logistic | 0.0125 | 0.0483 | 0.0880 |
| **Physiology-only** | **0.0918** | **0.1012** | **0.1266** |
| Laboratory-only | 0.0060 | 0.0278 | 0.0527 |
| Text-only | 0.0036 | 0.0136 | 0.0259 |
| Naive concatenation | 0.0480 | 0.0770 | 0.1034 |
| Multimodal | 0.0215 | 0.0833 | 0.1175 |

**Physiology provides the dominant predictive signal in this cohort.** The multimodal architecture is competitive but does not exceed physiology-only performance on the held-out test set.

### Uncertainty quantification

Hourly anchors within one ICU stay are correlated, so confidence intervals are computed with **ICU-stay-level cluster bootstrap resampling** ($B=2{,}000$ replicates) rather than treating every hourly anchor as independent.

| Model | AUPRC (6h) | 95% CI |
|---|---:|---:|
| Logistic | 0.0880 | [0.0453, 0.1509] |
| Physiology-only | 0.1266 | [0.0557, 0.2473] |
| Naive concatenation | 0.1034 | [0.0553, 0.1802] |
| Multimodal | 0.1175 | [0.0592, 0.1954] |

**Paired differences** (all bootstrap intervals below contain zero, so none of these comparisons is statistically distinguishable at this sample size):

| Comparison | Difference | 95% CI |
|---|---:|---:|
| Multimodal − Physiology-only | −0.0091 | [−0.0980, +0.0575] |
| Multimodal − Naive concatenation | +0.0141 | [−0.0496, +0.0885] |
| Multimodal − Logistic | +0.0294 | [−0.0145, +0.0862] |

---

## 11. Alarm Policy

Prediction and alarm generation are deliberately separated. The model produces a continuous risk estimate $p_t$; a controller converts it into an operational alarm. A bare threshold rule $A_t = \mathbf{1}[p_t \ge \tau]$ would react to isolated fluctuations, so the implemented controller additionally requires $p_t \ge \tau$ for $k=2$ consecutive hourly predictions, followed by a 60-minute cooldown after each alarm. The operating threshold $\tau = 0.05$ was selected on the *validation* split and frozen before evaluation on test.

| Quantity | Value |
|---|---:|
| Threshold $\tau$ | 0.05 |
| Consecutive predictions $k$ | 2 |
| Cooldown | 60 min |
| Patient-days | 520.21 |
| Total alarms | 5,950 |
| Alarms / patient-day | 11.44 |
| Alarm precision | 4.49% |
| Events | 73 |
| Events detected | 62 |
| Event sensitivity | 84.93% |
| Median warning time | 5.18 h |
| Q1 / Q3 warning time | 2.89 h / 5.63 h |

This operating point demonstrates the core trade-off explicitly: earlier, more sensitive detection comes at the cost of alarm burden. The system should **not** be read as deployment-ready at this operating point — that is precisely why prediction quality and alarm-policy quality are evaluated as separate objectives here.

---

## 12. Explainability

For a trained model $f$, the modality-contribution attribution is

$$
\Delta_m = f(x) - f(x_{\setminus m}),
$$

where $x_{\setminus m}$ is the input with modality $m$ removed or suppressed. This is a **model-attribution quantity, not a causal effect**: $\Delta_m \neq \text{clinical causal effect}$. Across 50 explained alarm samples, attribution is dominated by physiological information, consistent with the ablation and missing-modality results above.

---

## 13. Reproducibility

**Use a fresh virtual environment.** Do not assume an existing Python environment already has the correct package/PyTorch/CUDA versions for this project.

The repository provides `requirements.txt` for general Python dependencies and `install_torch.py` for the PyTorch/CUDA-specific install step.

### Prerequisites

- Python 3.11
- Access to the eICU Demo raw data (credentialed PhysioNet access)
- Sufficient disk space for raw/intermediate/generated artifacts
- Optional NVIDIA GPU for accelerated training (the pipeline runs on CPU, just slower)

### Windows (PowerShell)

```powershell
cd path\to\ICU_monitor_dilemma

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
python install_torch.py

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Linux / macOS

```bash
cd /path/to/ICU_monitor_dilemma

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python install_torch.py

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### Recreating a broken environment

```powershell
# Windows
deactivate
Remove-Item -Recurse -Force .\.venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python install_torch.py
```

```bash
# Linux / macOS
deactivate 2>/dev/null || true
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python install_torch.py
```

### Raw data placement

Place the raw eICU Demo download under:

```text
data/raw/eicu_demo/
```

including the source `.csv.gz` tables, `LICENSE.txt`, and `SHA256SUMS.txt`. The pipeline never writes to or deletes anything under `data/raw/`.

### Running the full pipeline

```powershell
# Windows
New-Item -ItemType Directory -Force -Path .\logs | Out-Null
python -u run_all.py 2>&1 |
    Tee-Object -FilePath ".\logs\run_all_full_$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss').log"
```

```bash
# Linux / macOS
mkdir -p logs
python -u run_all.py 2>&1 | tee "logs/run_all_full_$(date +%Y-%m-%d_%H-%M-%S).log"
```

`-u` (unbuffered stdout) keeps the log lines in the correct chronological order when piped through `tee`/`Tee-Object`.

Other useful invocations:

```bash
python run_all.py --dry-run           # show the execution plan without running anything
python run_all.py --from 06           # resume from feature construction onward
python run_all.py --skip 08 09 10 11 12   # skip model training / evaluation stages
```

Every numbered stage also runs standalone, e.g. `python src/06_build_features.py`, which is useful when debugging a single stage.

### Clean rerun (raw data untouched)

```bash
rm -rf data/interim/* data/processed/* results/* figures/* models/*
mkdir -p logs
python -u run_all.py 2>&1 | tee "logs/run_all_full_$(date +%Y-%m-%d_%H-%M-%S).log"
```

---

## 14. Project Structure

```text
ICU_monitor_dilemma/
├── data/
│   ├── raw/eicu_demo/           # immutable source snapshot (*.csv.gz, sqlite/, LICENSE.txt, SHA256SUMS.txt)
│   ├── interim/                 # vital_clean.csv, timeline.csv, lab_clean.csv, note_clean.csv, text_embeddings.npy
│   └── processed/               # train.npz, validation.npz, test.npz, metadata.json
├── models/
│   ├── baseline/
│   └── multimodal/multimodal.pt
├── results/
│   ├── audit/       cohort/     features/    predictions/
│   ├── evaluation/  alarms/     explanations/
├── figures/                     # roc.png, pr.png, calibration.png, ablation.png, alarm_tradeoff.png, ...
├── report/
│   ├── whitepaper.md
│   └── tables.md
├── src/
│   ├── 00_download_check.py … 12_bootstrap_ci.py
│   ├── common.py    features.py    nets.py
├── logs/
├── install_torch.py
├── run_all.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Pipeline dependency graph

```text
00 RAW DATA INTEGRITY
 └─▶ 01 SCHEMA / MODALITY AUDIT
      └─▶ 02 TASK AUDIT + ENDPOINT FREEZE
           └─▶ 03 COHORT + PATIENT SPLIT
                └─▶ 04 PHYSIOLOGICAL TIMELINE
                     └─▶ 05 MULTI-HORIZON LABELS
                          └─▶ 06 FEATURE TENSORS
                               ├─▶ 07 BASELINES ─┐
                               └─▶ 08 MULTIMODAL ┴─▶ 09 EVALUATION + ABLATION
                                                       ├─▶ 10 ALARM + ROBUSTNESS ─┐
                                                       └─▶ 11 EXPLANATION + STRESS┴─▶ 12 BOOTSTRAP CI
```

| Stage | Responsibility |
|---|---|
| `00` | Raw data integrity |
| `01` | Schema, missingness, modality coverage |
| `02` | Task audit and endpoint freeze |
| `03` | Cohort construction and patient-level split |
| `04` | Physiological temporal representation |
| `05` | Multi-horizon labels |
| `06` | Multimodal feature tensors + causal truncation test |
| `07` | Baseline models |
| `08` | Multimodal model |
| `09` | Predictive evaluation and ablation |
| `10` | Alarm controller and missing-modality robustness |
| `11` | Explainability and stress testing |
| `12` | Stay-clustered bootstrap uncertainty |

---

## 15. Deliverables

- **`report/whitepaper.md`** — full mathematical account: problem formulation, endpoint definition, causal feature construction, architecture, evaluation methodology, results, alarm analysis, explainability, limitations.
- **`report/tables.md`** — every publication-style result table referenced above, in one place.
- **`results/`, `figures/`, `models/`** — generated artifacts from the most recent full pipeline run (see directory structure above).
- **`logs/run_all_full_*.log`** — a persistent, timestamped record of every stage's console output, timing, and final status.

---

## 16. Current Experimental Interpretation

The principal result is **not** that multimodal fusion automatically improves deterioration prediction. Instead:

1. Physiological observations contain the dominant predictive signal in the current cohort.
2. The multimodal model is competitive but does not establish superiority over physiology-only modeling.
3. Text and laboratory modalities provide limited incremental signal under the present representation and cohort size — most plausibly because of their sparsity (2% of hourly rows carry a new note; only 225 unique training notes fit the text encoder) rather than an architectural limitation.
4. Removing physiological information causes by far the largest performance degradation.
5. The alarm operating point demonstrates a substantial sensitivity-vs-burden trade-off.
6. Stay-level bootstrap uncertainty is large enough that none of the point-estimate differences between models should be read as established superiority.

> **Under the present eICU Demo cohort, hourly temporal representation, endpoint definition, and model configuration, physiological observations provide the dominant predictive signal. The multimodal fusion model achieves competitive performance, but the available test cohort does not establish statistically distinguishable superiority over physiology-only or the other evaluated baselines.**

---

## 17. Limitations

1. The eICU **Demo** (2,520 stays) is used, not a large external clinical cohort (the full eICU-CRD has ~200,000 stays).
2. The final test cohort has 245 ICU stays and only 322 positive 6-hour anchors — statistical power is limited, especially at the 1-hour horizon (38–41 positives per split).
3. Clinical-note availability is sparse (~2% of hourly rows).
4. The text representation (character TF-IDF → SVD) is intentionally lightweight.
5. `composite_collapse` is a heterogeneous composite endpoint (three distinct deterioration mechanisms merged into one target).
6. No external or prospective clinical validation has been performed.
7. Occlusion-based explanations are model attributions, not causal clinical effects.

Recommended next steps: external validation → richer asynchronous modeling → calibration and clinical-utility analysis → prospective evaluation.

---

## 18. Citation and Data Provenance

The raw clinical data originate from the eICU Collaborative Research Database Demo. Reproducing this work requires following the dataset's own licensing, access, and citation requirements. Raw data are not redistributed through this repository.