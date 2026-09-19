# The ICU Monitor Dilemma

## Leakage-Controlled Multimodal Forecasting of ICU Physiological Deterioration

---

## 1. Problem

Intensive-care units continuously generate heterogeneous clinical data from bedside monitors, laboratory systems, and clinical documentation.

A conventional ICU monitor primarily reacts to threshold violations that have already occurred. This creates a fundamental operational problem: transient artifacts, sensor failures, physiological variability, and documentation delays can generate large numbers of alarms, while clinically meaningful deterioration may be difficult to distinguish from background noise.

The objective of this project is to construct a research prototype for **early forecasting of clinically meaningful physiological deterioration** from heterogeneous ICU data.

The system considers four information sources:

$$
\mathcal X(t)
=
\left[
X^{(p)}(t),
X^{(l)}(t),
X^{(n)}(t),
X^{(s)}
\right],
$$

where:

* \(X^{(p)}(t)\): physiological observations,
* \(X^{(l)}(t)\): laboratory observations,
* \(X^{(n)}(t)\): clinical notes,
* \(X^{(s)}\): static stay-level information.

The forecasting objective is:

$$
p_h(t)
=
P
\left(
Y_h(t)=1
\mid
\mathcal F(t)
\right),
$$

where:

* \(t\) is the prediction anchor,
* \(h\) is a future prediction horizon,
* \(Y_h(t)\) indicates whether deterioration occurs within that horizon,
* \(\mathcal F(t)\) denotes information actually available at prediction time.

The project deliberately separates:

$$
\boxed{
\text{data construction}
\rightarrow
\text{risk prediction}
\rightarrow
\text{alarm generation}
}
$$

rather than treating a high predicted risk and a clinical alarm as the same object.

---

## 2. Dataset

The experiments use the freely accessible **eICU Collaborative Research Database Demo**.

The project uses the following core source tables:

* `patient`
* `vitalPeriodic`
* `vitalAperiodic`
* `lab`
* `note`

Additional available tables were audited for modality coverage and schema information.

### Raw-data audit

| Table             |      Rows | Required | Readable |
| ----------------- | --------: | :------: | :------: |
| `patient`         |     2,520 |    Yes   |    Yes   |
| `vitalPeriodic`   | 1,634,960 |    Yes   |    Yes   |
| `vitalAperiodic`  |   274,088 |    Yes   |    Yes   |
| `lab`             |   434,660 |    Yes   |    Yes   |
| `note`            |    24,758 |    Yes   |    Yes   |
| `nurseCharting`   | 1,477,163 |    No    |    Yes   |
| `nurseAssessment` |    91,589 |    No    |    Yes   |
| `physicalExam`    |    84,058 |    No    |    Yes   |

The raw-data integrity stage verifies:

1. required files exist;
2. files are readable;
3. required columns exist;
4. expected schemas are present;
5. official SHA256 checksums match.

The raw data are treated as immutable source data. Intermediate and generated outputs are constructed separately under `data/interim/`, `data/processed/`, `results/`, `figures/`, and `models/`.

### Modality coverage

Among the 2,520 ICU stays:

| Available modalities  | Stays | Percentage |
| --------------------- | ----: | ---------: |
| None                  |    36 |      1.43% |
| Notes only            |     9 |      0.36% |
| Labs only             |    42 |      1.67% |
| Labs + notes          |    53 |      2.10% |
| Vitals only           |    13 |      0.52% |
| Vitals + notes        |    18 |      0.71% |
| Vitals + labs         |   172 |      6.83% |
| Vitals + labs + notes | 2,177 |     86.39% |

Thus, although the modalities are heterogeneous and incomplete, the majority of stays contain all three core dynamic modalities.

---

## 3. Research Question

The central research question is:

> **Can a leakage-controlled temporal model use heterogeneous ICU physiology, laboratory measurements, clinical notes, and static information to forecast clinically meaningful deterioration several hours before it occurs, while remaining robust to missing or delayed modalities and producing an operationally interpretable alarm process?**

The study decomposes this into several narrower questions.

### RQ1 — Predictive signal

Does the available ICU data contain useful information for forecasting future deterioration?

### RQ2 — Modality contribution

Which information source contributes the dominant predictive signal?

$$
\text{Physiology}
\quad\text{vs.}\quad
\text{Laboratory}
\quad\text{vs.}\quad
\text{Text}
$$

### RQ3 — Fusion

Does explicit multimodal representation provide incremental predictive value over:

* simple statistical baselines,
* physiology-only models,
* single-modality models,
* naive feature concatenation?

### RQ4 — Missingness

How does predictive performance change when one or more modalities are absent, delayed, or partially observed?

### RQ5 — Operationalization

How does continuous deterioration risk translate into an alarm stream, and what is the tradeoff between event sensitivity, warning time, and alarm burden?

### RQ6 — Robustness

How sensitive is the system to corruption of the physiological stream, including measurement noise and temporary sensor blackout?

---

## 4. Prediction Target

The endpoint was audited before downstream model development and then frozen.

The selected endpoint is `composite_collapse`.

It is defined as the first occurrence of any of:

1. sustained hypotension;
2. sustained hypoxemia;
3. ICU death.

### Hypotension

The first sustained episode satisfying

$$
MAP < 55\text{ mmHg}
$$

for at least 30 minutes.

### Hypoxemia

The first sustained episode satisfying

$$
SpO_2 < 88\%
$$

for at least 30 minutes.

### ICU death

Death occurring during the ICU stay.

The composite event time is therefore:

$$
T_i^{collapse}
=
\min
\left(
T_i^{hypotension},
T_i^{hypoxemia},
T_i^{death}
\right).
$$

### Horizon-specific labels

For an anchor \(t\) and horizon \(h\):

$$
Y_h(t)
=
\mathbf 1
\left[
t<T^{collapse}\leq t+h
\right].
$$

The implemented horizons are:

$$
h\in\{1,3,6\}\text{ hours}.
$$

An anchor inside an already active deterioration episode is excluded from prediction.

Label invariants are explicitly checked so that valid labels satisfy:

$$
T^{collapse}>t,
$$

$$
T^{collapse}\leq t+h,
$$

and

$$
t<T^{ICU\ discharge}.
$$

### Endpoint audit

| Candidate endpoint | 1h positives | 3h positives | 6h positives | Decision              |
| ------------------ | -----------: | -----------: | -----------: | --------------------- |
| Composite collapse |          292 |        1,259 |        2,457 | **Selected**          |
| ICU death          |           74 |          216 |          418 | Retained as candidate |
| Hospital death     |           51 |          167 |          371 | Rejected              |
| Severe hypotension |          116 |          545 |        1,093 | Retained as candidate |
| Severe hypoxemia   |          130 |          607 |        1,192 | Retained as candidate |

Hospital death was rejected as the primary endpoint because a substantial fraction of hospital-death events did not occur within the ICU observation window and therefore had less reliable temporal alignment with the ICU forecasting task.

---

## 5. Temporal Setup

The raw ICU data are asynchronous.

Physiological measurements can arrive at high frequency, laboratory measurements are sparse and irregular, and notes are event-driven.

The pipeline therefore constructs an **hourly prediction grid**.

For ICU stay \(i\), define anchor times:

$$
t_{i,k}
=
t_{i,0}+60k\text{ minutes}.
$$

At anchor \(t_{i,k}\), the feature constructor is allowed to use only information available at or before that time.

Define the information filtration:

$$
\mathcal F_{i,k}
=
\sigma
\left(
X_i(\tau):\tau\leq t_{i,k}
\right).
$$

The prediction problem is therefore:

$$
\hat p_{i,k,h}
=
f_\theta
\left(
\mathcal F_{i,k}
\right)_h.
$$

The implemented prediction horizons are:

$$
1\text{ h},\quad3\text{ h},\quad6\text{ h}.
$$

### Final cohort

The cohort construction begins with:

$$
2520
$$

ICU stays belonging to:

$$
1841
$$

patients.

After eligibility filtering, the final cohort contains:

$$
1565
$$

ICU stays belonging to:

$$
1294
$$

patients.

### Cohort funnel

| Stage                                |     Stays |  Patients |
| ------------------------------------ | --------: | --------: |
| All ICU stays                        |     2,520 |     1,841 |
| Valid ID + ICU discharge             |     2,520 |     1,841 |
| ICU LOS >24 h                        |     1,626 |     1,332 |
| Known ICU discharge                  |     1,626 |     1,332 |
| Plausible timeline                   |     1,622 |     1,328 |
| ≥12 vital observations in first 24 h |     1,566 |     1,294 |
| Plausible event timing               | **1,565** | **1,294** |

### Patient-level split

The partition is performed at the patient level.

| Split      | Stays | Patients | Event stays |
| ---------- | ----: | -------: | ----------: |
| Train      | 1,086 |      905 |         182 |
| Validation |   234 |      194 |          40 |
| Test       |   245 |      195 |          40 |

No patient is intentionally represented across multiple partitions.

This prevents information leakage caused by placing different ICU stays belonging to the same patient into different splits.

---

## 6. Multimodal Architecture

The system treats each clinical modality as a distinct information-generating process.

At temporal index \(k\):

$$
P_k\in\mathbb R^{35}
$$

represents physiological features,

$$
L_k\in\mathbb R^{80}
$$

represents laboratory features, and

$$
N_k\in\mathbb R^{66}
$$

represents text features.

Static information is:

$$
S\in\mathbb R^{15}.
$$

Thus the dynamic multimodal representation has:

$$
35+80+66=181
$$

features per hourly row.

The conceptual architecture is:

```text
                ┌─────────────────────┐
                │ Physiological data  │
                └──────────┬──────────┘
                           │
                           ▼
                  Physiology encoder
                           │
                           ▼
                       H^(p)
                           │
                           │
                ┌─────────────────────┐
                │ Laboratory data     │
                └──────────┬──────────┘
                           │
                           ▼
                     Lab encoder
                           │
                           ▼
                       H^(l)
                           │
                           │
                ┌─────────────────────┐
                │ Clinical notes      │
                └──────────┬──────────┘
                           │
                           ▼
                     Text encoder
                           │
                           ▼
                       H^(n)
                           │
                           │
                ┌─────────────────────┐
                │ Static information  │
                └──────────┬──────────┘
                           │
                           ▼
                     Static features
                           │
                           ▼
                 ┌────────────────────┐
                 │ Multimodal fusion  │
                 └─────────┬──────────┘
                           │
                           ▼
                    Risk prediction
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
             1h           3h           6h
```

Abstractly:

$$
H^{(p)}
=
f_p(P_{1:T}),
$$

$$
H^{(l)}
=
f_l(L_{1:T}),
$$

$$
H^{(n)}
=
f_n(N_{1:T}),
$$

followed by:

$$
Z
=
f_{fusion}
\left(
H^{(p)},H^{(l)},H^{(n)},S
\right).
$$

The final risk vector is:

$$
\hat{\mathbf p}
=
(\hat p_1,\hat p_3,\hat p_6).
$$

The implemented multimodal model contains approximately:

$$
172,070
$$

trainable parameters.

The exact PyTorch implementation is defined in:

```text
src/nets.py
```

The README intentionally describes the architectural contract mathematically without reproducing implementation-specific layer code.

---

## 7. Missingness Handling

Missingness is treated as part of the clinical data-generating process rather than as a nuisance that can simply be ignored.

The three core dynamic modalities have substantially different observation densities.

### Physiology

The physiological stream is comparatively dense.

The principal variables are:

* heart rate,
* respiratory rate,
* SpO2,
* temperature,
* systolic blood pressure,
* diastolic blood pressure,
* MAP.

The resulting physiological representation has:

$$
d_p=35.
$$

### Laboratory data

Laboratory measurements are irregular and sparse.

Sixteen laboratory variables are represented using five temporal summary statistics each:

$$
d_l=16\times5=80.
$$

### Clinical notes

Clinical notes are highly event-driven.

The text representation is:

$$
\text{character TF-IDF}
\rightarrow
\text{SVD}
\rightarrow
\mathbb R^{66}.
$$

Only approximately 2% of hourly rows contain a newly available note.

### Missing-modality experiments

The trained system is explicitly evaluated under modality perturbation.

Examples include:

* laboratory removal;
* text removal;
* physiology removal;
* combined modality removal;
* 20%, 40%, and 60% laboratory dropout;
* 60-minute note delay;
* 180-minute note delay.

The principal finding is that removing physiology causes the largest performance degradation, indicating that physiological observations contain the dominant predictive signal in the present cohort.

---

## 8. Leakage Prevention

Leakage prevention is a first-class component of the project.

### 8.1 Patient-level splitting

The train/validation/test split is performed by patient rather than by individual hourly observation.

### 8.2 Temporal feature truncation

At anchor \(t\), only observations satisfying:

$$
\tau\leq t
$$

may contribute to the features.

### 8.3 Clinical-note availability

A critical leakage source was identified during the note audit.

Approximately 98% of notes were documented after their clinical timestamp.

Median documentation delay was approximately:

$$
12\text{ minutes}.
$$

Therefore note availability is determined using:

$$
t^{available}=t^{entry},
$$

rather than retrospective clinical timing.

A note can contribute to an anchor only if:

$$
t^{entry}\leq t_{anchor}.
$$

### 8.4 Training-only representation fitting

The text TF-IDF/SVD transformation is fitted on training notes rather than the full dataset.

### 8.5 Endpoint freezing

Candidate endpoints are audited before downstream modeling.

The selected endpoint is then frozen and downstream stages read the locked task definition.

### 8.6 Label causality

Labels are checked to ensure that event onset is strictly after the prediction anchor and within the selected forecast horizon.

### 8.7 Truncation tests

The feature construction pipeline includes explicit tests in which the available history is truncated at randomly selected anchor times.

The resulting features are checked against the corresponding full pipeline representation.

This provides an implementation-level check of:

$$
\Phi(X_{\leq t})
\not\leftarrow
X_{>t}.
$$

---

## 9. Baselines

The experimental hierarchy includes progressively more complex models.

### Logistic regression

A flat baseline:

$$
\hat p
=
\sigma(w^\top x+b),
$$

where:

$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

### Physiology-only

Uses the physiological temporal representation.

### Laboratory-only

Uses laboratory features without physiological or text information.

### Text-only

Uses the clinical-note representation.

### Naive multimodal concatenation

The modality features are concatenated before temporal modeling.

### Proposed multimodal model

Each modality is represented separately before multimodal fusion.

This hierarchy is necessary because a complex multimodal architecture is only scientifically useful if it is compared against simpler explanations of the same predictive signal.

---

## 10. Multimodal Model

The multimodal model receives:

$$
\{P_k,L_k,N_k\}_{k=1}^{T}
$$

and static information:

$$
S.
$$

The modality-specific representations are abstractly:

$$
H^{(p)}=f_p(P_{1:T}),
$$

$$
H^{(l)}=f_l(L_{1:T}),
$$

$$
H^{(n)}=f_n(N_{1:T}).
$$

These representations are combined:

$$
Z
=
f_{fusion}
\left(
H^{(p)},
H^{(l)},
H^{(n)},
S
\right).
$$

The output head estimates:

$$
\hat{\mathbf p}
=
\begin{bmatrix}
\hat p_1\\
\hat p_3\\
\hat p_6
\end{bmatrix}.
$$

The model contains approximately:

$$
172,070
$$

trainable parameters.

The implementation is located in:

```text
src/nets.py
```

Training is implemented in:

```text
src/08_train_multimodal.py
```

The model is trained using the training partition and model-selection information from the validation partition.

The test set is reserved for final evaluation.

---

## 11. Evaluation

The primary discrimination metric is **area under the precision-recall curve (AUPRC)**.

For predictions converted to a binary decision at threshold \(\tau\):

$$
Precision(\tau)
=
\frac{TP(\tau)}
{TP(\tau)+FP(\tau)}
$$

and:

$$
Recall(\tau)
=
\frac{TP(\tau)}
{TP(\tau)+FN(\tau)}.
$$

AUPRC summarizes the precision-recall relationship across thresholds.

This is particularly appropriate for the present highly imbalanced deterioration task.

### Test prevalence

At the 6-hour horizon:

$$
\pi
=
\frac{322}{12485}
\approx
0.0258.
$$

Thus the event prevalence is approximately:

$$
2.58\%.
$$

### Test AUPRC

| Model               |         1h |         3h |         6h |
| ------------------- | ---------: | ---------: | ---------: |
| Logistic            |     0.0125 |     0.0483 |     0.0880 |
| Physiology-only     | **0.0918** | **0.1012** | **0.1266** |
| Laboratory-only     |     0.0060 |     0.0278 |     0.0527 |
| Text-only           |     0.0036 |     0.0136 |     0.0259 |
| Naive concatenation |     0.0480 |     0.0770 |     0.1034 |
| Multimodal          |     0.0215 |     0.0833 |     0.1175 |

### Interpretation

The principal empirical observation is:

> Physiology provides the dominant predictive signal in this cohort.

The multimodal architecture remains competitive, but its observed test-set performance does not establish an advantage over physiology-only prediction.

### Uncertainty quantification

Hourly anchors within an ICU stay are correlated.

Therefore confidence intervals are computed using **ICU-stay-level bootstrap resampling**, rather than treating every hourly anchor as independent.

For bootstrap replicate \(b\), ICU stays are sampled with replacement and all anchors belonging to each sampled stay are retained.

The final uncertainty analysis uses:

$$
B=2000
$$

bootstrap replicates.

### Six-hour confidence intervals

| Model               |  AUPRC |           95% CI |
| ------------------- | -----: | ---------------: |
| Logistic            | 0.0880 | [0.0453, 0.1509] |
| Physiology-only     | 0.1266 | [0.0557, 0.2473] |
| Naive concatenation | 0.1034 | [0.0553, 0.1802] |
| Multimodal          | 0.1175 | [0.0592, 0.1954] |

### Paired differences

| Comparison                       | Difference |             95% CI |
| -------------------------------- | ---------: | -----------------: |
| Multimodal − Physiology-only     |    −0.0091 | [−0.0980, +0.0575] |
| Multimodal − Naive concatenation |    +0.0141 | [−0.0496, +0.0885] |
| Multimodal − Logistic            |    +0.0294 | [−0.0145, +0.0862] |

All intervals contain zero.

Therefore the experiment does **not** establish statistically distinguishable performance between the multimodal model and these comparison models.

---

## 12. Alarm Policy

Prediction and alarm generation are deliberately separated.

The model produces a continuous risk estimate:

$$
p_t.
$$

An alarm controller converts this risk into an operational event.

A simple threshold rule is:

$$
A_t
=
\mathbf1[p_t\geq\tau].
$$

To avoid reacting to isolated predictions, the implemented controller additionally requires:

$$
p_t\geq\tau
$$

for:

$$
k=2
$$

consecutive predictions.

After an alarm is triggered, a cooldown period of:

$$
60\text{ minutes}
$$

is applied.

The evaluated operating point uses:

$$
\tau=0.05.
$$

### Test alarm results

| Quantity                |  Value |
| ----------------------- | -----: |
| Threshold               |   0.05 |
| Consecutive predictions |      2 |
| Cooldown                | 60 min |
| Patient-days            | 520.21 |
| Total alarms            |  5,950 |
| Alarms / patient-day    |  11.44 |
| Alarm precision         |  4.49% |
| Events                  |     73 |
| Events detected         |     62 |
| Event sensitivity       | 84.93% |
| Median warning time     | 5.18 h |
| Q1 warning time         | 2.89 h |
| Q3 warning time         | 5.63 h |

### Interpretation

The alarm operating point demonstrates a fundamental tradeoff:

$$
\text{earlier/more sensitive detection}
\quad\leftrightarrow\quad
\text{higher alarm burden}.
$$

The system should **not** be interpreted as clinically deployment-ready at this operating point.

The result instead demonstrates why the prediction problem and the alarm-policy problem must be evaluated separately.

---

## 13. Explainability

The explanation module estimates how the model's prediction changes when modalities are perturbed.

For a model \(f\), an abstract modality contribution can be written:

$$
\Delta_m
=
f(x)-f(x_{\setminus m}),
$$

where:

* \(x\) is the original multimodal input;
* \(x_{\setminus m}\) is the corresponding input with modality \(m\) removed or suppressed.

This is a **model-attribution quantity**.

It is not a causal effect.

Therefore:

$$
\Delta_m\neq
\text{clinical causal effect}.
$$

The current explanation analysis evaluates 50 alarm samples.

The observed modality attribution is dominated by physiological information in the current implementation.

This is consistent with the predictive ablation experiments in which removing physiology produces the largest degradation.

---

## 14. Reproducibility

The project is designed to be reproducible from a clean Python environment.

### Important principle

**Use a fresh virtual environment for reproducibility.**

Do not assume that an existing Python environment contains the correct versions of:

* Python packages,
* PyTorch,
* CUDA-dependent components,
* scientific Python libraries.

If an existing environment is already present, it is recommended to create a new environment specifically for this project.

The repository contains:

```text
requirements.txt
```

for the general Python dependencies and:

```text
install_torch.py
```

for PyTorch installation.

---

### 14.1 Prerequisites

The repository assumes:

* Python 3.x;
* access to the eICU Demo raw data;
* sufficient disk space for raw/intermediate/generated artifacts;
* sufficient RAM for preprocessing;
* optional NVIDIA GPU for accelerated model training.

CUDA availability is hardware/environment dependent.

The pipeline can be executed without CUDA, although model training can be substantially slower on CPU-only systems.

---

### 14.2 Windows PowerShell

Open PowerShell and move to the repository root:

```powershell
cd path\to\ICU_monitor_dilemma
```

Create a fresh environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade `pip`:

```powershell
python -m pip install --upgrade pip
```

Install the repository dependencies:

```powershell
pip install -r requirements.txt
```

Install PyTorch using the repository installer:

```powershell
python install_torch.py
```

Verify PyTorch:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

If PowerShell blocks virtual-environment activation, the execution-policy restriction may need to be handled according to the local machine's security policy.

Do not bypass organizational security policies merely to run the project.

---

### 14.3 Linux

Move to the repository:

```bash
cd /path/to/ICU_monitor_dilemma
```

Create a fresh environment:

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch:

```bash
python install_torch.py
```

Verify:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

---

### 14.4 macOS

Move to the repository:

```bash
cd /path/to/ICU_monitor_dilemma
```

Create a fresh environment:

```bash
python3 -m venv .venv
```

Activate:

```bash
source .venv/bin/activate
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the repository PyTorch installer:

```bash
python install_torch.py
```

Verify:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

macOS systems do not provide NVIDIA CUDA in the usual Linux/Windows configuration. Hardware acceleration availability therefore depends on the installed PyTorch/platform configuration.

---

### 14.5 Clean environment recreation

If the local environment becomes inconsistent, recreate it.

#### Windows

```powershell
deactivate

Remove-Item -Recurse -Force .\.venv

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
python install_torch.py
```

#### Linux/macOS

```bash
deactivate 2>/dev/null || true

rm -rf .venv

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
python install_torch.py
```

---

### 14.6 Raw data placement

The raw eICU Demo data should be placed under:

```text
data/raw/eicu_demo/
```

The raw directory should contain the required source tables and associated metadata/checksum files.

The pipeline does not intentionally delete raw data.

---

### 14.7 Full pipeline execution — Windows

Create the logs directory:

```powershell
New-Item -ItemType Directory -Force -Path .\logs | Out-Null
```

Run the complete pipeline while simultaneously saving a timestamped log:

```powershell
python run_all.py 2>&1 |
    Tee-Object -FilePath ".\logs\run_all_full_$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss').log"
```

This provides:

1. live console output;
2. a persistent execution log.

---

### 14.8 Full pipeline execution — Linux/macOS

Create the logs directory:

```bash
mkdir -p logs
```

Run:

```bash
python run_all.py 2>&1 | tee "logs/run_all_full_$(date +%Y-%m-%d_%H-%M-%S).log"
```

---

### 14.9 Full pipeline without log capture

Windows:

```powershell
python run_all.py
```

Linux/macOS:

```bash
python run_all.py
```

---

### 14.10 Dry run

To inspect the execution order without running any script:

```powershell
python run_all.py --dry-run
```

or:

```bash
python run_all.py --dry-run
```

---

### 14.11 Resume from a specific stage

For example, to begin at feature construction:

```powershell
python run_all.py --from 06
```

or:

```bash
python run_all.py --from 06
```

This is useful when upstream artifacts already exist and only downstream stages need to be rerun.

---

### 14.12 Skip selected stages

For example:

```powershell
python run_all.py --skip 08 09 10 11
```

or:

```bash
python run_all.py --skip 08 09 10 11
```

This can be useful for debugging data construction without retraining the models.

---

### 14.13 Run an individual stage

Every numbered script remains independently executable.

For example:

```powershell
python src/00_download_check.py
python src/01_schema_audit.py
python src/02_task_audit.py
python src/03_build_cohort.py
python src/04_build_timeline.py
python src/05_build_labels.py
python src/06_build_features.py
python src/07_train_baselines.py
python src/08_train_multimodal.py
python src/09_evaluate.py
python src/10_alarm_analysis.py
python src/11_explain.py
python src/12_bootstrap_ci.py
```

This is useful when diagnosing a specific stage.

---

## 15. Project Structure

```text
ICU_monitor_dilemma/
│
├── .venv/
│   └── ...                         # Local virtual environment
│
├── data/
│   │
│   ├── raw/
│   │   └── eicu_demo/
│   │       ├── *.csv.gz
│   │       ├── sqlite/
│   │       ├── LICENSE.txt
│   │       └── SHA256SUMS.txt
│   │
│   ├── interim/
│   │   ├── vital_clean.csv
│   │   ├── timeline.csv
│   │   ├── lab_clean.csv
│   │   ├── note_clean.csv
│   │   └── text_embeddings.npy
│   │
│   └── processed/
│       ├── train.npz
│       ├── validation.npz
│       ├── test.npz
│       └── metadata.json
│
├── models/
│   ├── baseline/
│   │   └── ...
│   │
│   └── multimodal/
│       └── multimodal.pt
│
├── results/
│   │
│   ├── audit/
│   │   ├── schema_summary.csv
│   │   ├── missingness.csv
│   │   ├── modality_coverage.csv
│   │   ├── modality_combinations.csv
│   │   └── lab_names.csv
│   │
│   ├── cohort/
│   │   └── ...
│   │
│   ├── features/
│   │   ├── vital_summary.csv
│   │   └── feature_summary.csv
│   │
│   ├── predictions/
│   │   ├── baseline_predictions.csv
│   │   ├── multimodal_predictions.csv
│   │   └── ablation_predictions.csv
│   │
│   ├── evaluation/
│   │   ├── metrics.csv
│   │   ├── calibration.csv
│   │   ├── ablations.csv
│   │   ├── lead_time_summary.csv
│   │   ├── calibrators.csv
│   │   ├── missing_modality.csv
│   │   ├── stress_tests.csv
│   │   ├── bootstrap_ci_6h.csv
│   │   └── bootstrap_differences_6h.csv
│   │
│   ├── alarms/
│   │   ├── alarm_metrics.csv
│   │   ├── alarm_events.csv
│   │   └── alarm_threshold_sweep.csv
│   │
│   └── explanations/
│       ├── explanation_examples.csv
│       ├── modality_contributions.csv
│       └── time_window_contributions.csv
│
├── figures/
│   ├── roc.png
│   ├── pr.png
│   ├── calibration.png
│   ├── ablation.png
│   ├── lead_time.png
│   ├── alarm_tradeoff.png
│   ├── missing_modality.png
│   ├── modality_importance.png
│   ├── example_explanation_01.png
│   └── example_explanation_02.png
│
├── report/
│   ├── whitepaper.md
│   └── tables.md
│
├── src/
│   ├── 00_download_check.py
│   ├── 01_schema_audit.py
│   ├── 02_task_audit.py
│   ├── 03_build_cohort.py
│   ├── 04_build_timeline.py
│   ├── 05_build_labels.py
│   ├── 06_build_features.py
│   ├── 07_train_baselines.py
│   ├── 08_train_multimodal.py
│   ├── 09_evaluate.py
│   ├── 10_alarm_analysis.py
│   ├── 11_explain.py
│   ├── 12_bootstrap_ci.py
│   ├── common.py
│   ├── features.py
│   └── nets.py
│
├── install_torch.py
├── run_all.py
├── requirements.txt
├── README.md
└── .gitignore
```

### Pipeline dependency graph

```text
00  RAW DATA INTEGRITY
 │
 ▼
01  SCHEMA / MODALITY AUDIT
 │
 ▼
02  TASK AUDIT + ENDPOINT FREEZE
 │
 ▼
03  COHORT + PATIENT SPLIT
 │
 ▼
04  PHYSIOLOGICAL TIMELINE
 │
 ▼
05  MULTI-HORIZON LABELS
 │
 ▼
06  FEATURE TENSORS
 │
 ├───────────────┐
 ▼               ▼
07 BASELINES    08 MULTIMODAL MODEL
 │               │
 └───────┬───────┘
         ▼
09 EVALUATION + ABLATION
         │
         ├───────────────┐
         ▼               ▼
10 ALARM + ROBUSTNESS   11 EXPLANATION + STRESS
         │               │
         └───────┬───────┘
                 ▼
        12 BOOTSTRAP UNCERTAINTY
```

### Responsibility of each stage

| Stage | Responsibility                                   |
| ----- | ------------------------------------------------ |
| `00`  | Raw data integrity                               |
| `01`  | Schema, missingness, modality coverage           |
| `02`  | Task audit and endpoint freeze                   |
| `03`  | Cohort construction and patient-level split      |
| `04`  | Physiological temporal representation            |
| `05`  | Multi-horizon labels                             |
| `06`  | Multimodal feature tensors                       |
| `07`  | Baseline models                                  |
| `08`  | Multimodal model                                 |
| `09`  | Predictive evaluation and ablation               |
| `10`  | Alarm controller and missing-modality robustness |
| `11`  | Explainability and stress testing                |
| `12`  | Clustered bootstrap uncertainty                  |

---

## 16. How to Run

### Recommended complete reproduction workflow

From a fresh checkout:

```text
1. Obtain the eICU Demo data.
2. Place raw data under data/raw/eicu_demo/.
3. Create a fresh Python virtual environment.
4. Activate the environment.
5. Upgrade pip.
6. Install requirements.txt.
7. Run install_torch.py.
8. Verify PyTorch/CUDA.
9. Create logs/.
10. Run run_all.py.
11. Inspect results/.
12. Inspect figures/.
13. Read report/tables.md and report/whitepaper.md.
```

### Windows — complete workflow

```powershell
cd path\to\ICU_monitor_dilemma

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

python install_torch.py

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

New-Item -ItemType Directory -Force -Path .\logs | Out-Null

python run_all.py 2>&1 |
    Tee-Object -FilePath ".\logs\run_all_full_$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss').log"
```

### Linux/macOS — complete workflow

```bash
cd /path/to/ICU_monitor_dilemma

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python install_torch.py

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

mkdir -p logs

python run_all.py 2>&1 | tee "logs/run_all_full_$(date +%Y-%m-%d_%H-%M-%S).log"
```

---

### Clean full rerun

If a complete rerun is required, generated artifacts can be removed while preserving raw data.

#### Windows

```powershell
Remove-Item -Recurse -Force .\data\interim\* -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\data\processed\* -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\results\* -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\figures\* -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .\models\* -ErrorAction SilentlyContinue

New-Item -ItemType Directory -Force -Path .\logs | Out-Null

python run_all.py 2>&1 |
    Tee-Object -FilePath ".\logs\run_all_full_$(Get-Date -Format 'yyyy-MM-dd_HH-mm-ss').log"
```

#### Linux/macOS

```bash
rm -rf data/interim/*
rm -rf data/processed/*
rm -rf results/*
rm -rf figures/*
rm -rf models/*

mkdir -p logs

python run_all.py 2>&1 | tee "logs/run_all_full_$(date +%Y-%m-%d_%H-%M-%S).log"
```

**Do not include `data/raw/` in the cleanup commands.**

The raw source data should remain untouched.

---

### Restart from feature construction

If stages `00`–`05` have already completed successfully:

```powershell
python run_all.py --from 06
```

Linux/macOS:

```bash
python run_all.py --from 06
```

---

### Run only data/model preparation

For example, to execute through the baseline stage without training the multimodal model or running downstream evaluation:

```powershell
python run_all.py --skip 08 09 10 11 12
```

---

### Skip expensive stages during debugging

For example:

```powershell
python run_all.py --skip 08 09 10 11 12
```

This allows the data construction and baseline portions to be inspected without running all downstream analysis.

---

### Inspect the pipeline without executing it

```powershell
python run_all.py --dry-run
```

or:

```bash
python run_all.py --dry-run
```

---

### Individual script execution

Every stage can be run directly:

```bash
python src/00_download_check.py
python src/01_schema_audit.py
python src/02_task_audit.py
python src/03_build_cohort.py
python src/04_build_timeline.py
python src/05_build_labels.py
python src/06_build_features.py
python src/07_train_baselines.py
python src/08_train_multimodal.py
python src/09_evaluate.py
python src/10_alarm_analysis.py
python src/11_explain.py
python src/12_bootstrap_ci.py
```

This is useful when debugging a specific stage or validating intermediate artifacts.

---

# 17. Deliverables and Expected Outputs

The repository is intended to provide both a reproducible implementation and a research record.

## 17.1 Research deliverables

### Whitepaper

```text
report/whitepaper.md
```

Contains:

* problem formulation;
* dataset;
* research questions;
* endpoint definition;
* temporal setup;
* multimodal architecture;
* missingness analysis;
* leakage prevention;
* baseline comparisons;
* evaluation;
* alarm analysis;
* explainability;
* robustness;
* statistical uncertainty;
* limitations;
* conclusions.

### Result tables

```text
report/tables.md
```

Contains publication-style tables for:

* data audit;
* modality coverage;
* endpoint audit;
* cohort construction;
* patient split;
* feature dimensions;
* label prevalence;
* model performance;
* ablations;
* bootstrap confidence intervals;
* alarm analysis;
* missing-modality experiments;
* stress testing;
* explanation analysis.

---

## 17.2 Primary generated results

After a successful full run, the principal result directories are:

```text
results/
├── audit/
├── cohort/
├── features/
├── predictions/
├── evaluation/
├── alarms/
└── explanations/
```

Important outputs include:

```text
results/evaluation/metrics.csv
results/evaluation/ablations.csv
results/evaluation/calibration.csv
results/evaluation/lead_time_summary.csv
results/evaluation/missing_modality.csv
results/evaluation/stress_tests.csv
results/evaluation/bootstrap_ci_6h.csv
results/evaluation/bootstrap_differences_6h.csv

results/alarms/alarm_metrics.csv
results/alarms/alarm_events.csv
results/alarms/alarm_threshold_sweep.csv

results/explanations/explanation_examples.csv
results/explanations/modality_contributions.csv
results/explanations/time_window_contributions.csv
```

---

## 17.3 Model artifacts

The trained multimodal model is written under:

```text
models/multimodal/
```

with the principal checkpoint:

```text
models/multimodal/multimodal.pt
```

Baseline artifacts are stored under:

```text
models/baseline/
```

---

## 17.4 Figures

The pipeline produces figures covering:

* ROC analysis;
* precision-recall analysis;
* calibration;
* model ablation;
* lead-time behavior;
* alarm tradeoffs;
* missing-modality robustness;
* explanation examples;
* modality importance;
* physiological noise;
* sensor blackout.

These are stored under:

```text
figures/
```

---

## 17.5 Reproducibility log

When the pipeline is executed using the recommended command, a timestamped log is produced under:

```text
logs/
```

For Windows:

```text
logs/run_all_full_YYYY-MM-DD_HH-mm-ss.log
```

For Linux/macOS:

```text
logs/run_all_full_YYYY-MM-DD_HH-mm-ss.log
```

The log provides a persistent record of:

* executed stages;
* stage outputs;
* errors;
* timing;
* final pipeline status.

---

# 18. Current Experimental Interpretation

The principal result of the current experiment is not that multimodal fusion automatically improves deterioration prediction.

Instead, the experiment finds that:

1. physiological observations contain the dominant predictive signal in the current cohort;
2. the multimodal model remains competitive but does not establish superiority over physiology-only modeling;
3. text and laboratory modalities provide limited incremental predictive signal under the present representation and cohort;
4. removal of physiological information causes the largest degradation;
5. the alarm operating point demonstrates a substantial sensitivity-versus-alarm-burden tradeoff;
6. stay-level bootstrap uncertainty is large enough that point-estimate differences should not be interpreted as established superiority.

The strongest scientifically defensible statement is therefore:

> **Under the present eICU Demo cohort, hourly temporal representation, endpoint definition, feature engineering, and model configuration, physiological observations provide the dominant predictive signal. The multimodal fusion model achieves competitive performance, but the available test cohort does not establish statistically distinguishable superiority over physiology-only or other evaluated baselines.**

---

# 19. Limitations

This project is a research prototype rather than a clinically validated deployment system.

Important limitations include:

1. the use of the eICU Demo rather than a large external clinical cohort;
2. a final test cohort of 245 ICU stays;
3. only 322 positive 6-hour test anchors;
4. sparse clinical-note availability;
5. lightweight TF-IDF/SVD text representation;
6. heterogeneous composite endpoint definition;
7. limited statistical power for fine-grained model comparisons;
8. absence of external validation;
9. absence of prospective clinical evaluation;
10. an experimental rather than clinically optimized alarm policy;
11. model attribution that should not be interpreted as causal inference.

The appropriate next research steps are therefore:

$$
\text{external validation}
\rightarrow
\text{richer asynchronous modeling}
\rightarrow
\text{calibration}
\rightarrow
\text{clinical utility analysis}
\rightarrow
\text{prospective evaluation}.
$$

---

# 20. Scientific Reproducibility Principles

The project follows several principles throughout the pipeline:

### Freeze the task before optimizing the model

$$
\boxed{
\text{task definition}
\rightarrow
\text{data construction}
\rightarrow
\text{model}
}
$$

rather than changing the endpoint after observing model results.

### Respect information availability

$$
\boxed{
\text{available information at }t
\neq
\text{complete retrospective record}
}
$$

### Split by independent clinical entity

Patient-level partitioning is used rather than randomly splitting hourly observations.

### Separate prediction from intervention

$$
\boxed{
\text{risk score}
\neq
\text{alarm}
}
$$

### Quantify uncertainty

Model comparisons are accompanied by clustered bootstrap confidence intervals.

### Report negative findings

If additional modalities fail to provide measurable incremental information, that result is retained rather than hidden.

---

# 21. Citation and Data Provenance

The raw clinical data originate from the eICU Collaborative Research Database Demo.

Users reproducing this work should follow the dataset's licensing, access, citation, and attribution requirements.

The raw data should not be redistributed through this repository unless explicitly permitted by the dataset's terms.

---

# 22. Final Repository Workflow

The intended workflow is:

```text
                     RAW eICU DATA
                           │
                           ▼
                  00. DATA INTEGRITY
                           │
                           ▼
                01. SCHEMA / COVERAGE
                           │
                           ▼
               02. ENDPOINT AUDIT/FREEZE
                           │
                           ▼
                 03. COHORT + SPLIT
                           │
                           ▼
                04. TEMPORAL TIMELINE
                           │
                           ▼
                  05. FUTURE LABELS
                           │
                           ▼
                   06. FEATURES
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
             07. BASELINES    08. MULTIMODAL
                  │                 │
                  └────────┬────────┘
                           ▼
                  09. EVALUATION
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
          10. ALARM/ROBUSTNESS   11. EXPLANATION
                 │                   │
                 └─────────┬─────────┘
                           ▼
                  12. BOOTSTRAP CI
                           │
                           ▼
              REPORT + TABLES + FIGURES
```

The repository is therefore not merely a trained model.

It is an end-to-end research pipeline:

$$
\boxed{
\text{raw clinical data}
\rightarrow
\text{causal temporal dataset}
\rightarrow
\text{multimodal forecasting}
\rightarrow
\text{uncertainty}
\rightarrow
\text{alarm analysis}
\rightarrow
\text{scientific report}
}
$$

The objective is to make every major scientific assumption explicit, every major transformation reproducible, and every reported result traceable to a defined stage of the pipeline.
