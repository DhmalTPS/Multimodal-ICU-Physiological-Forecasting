# The ICU Monitor Dilemma

## Leakage-Controlled Multimodal Forecasting of Physiological Deterioration from Asynchronous ICU Data

<!-- > Math in this document uses GitHub's native Markdown math syntax: `$...$` for inline expressions and `$$...$$` for display equations, each on its own line with no blank line inside the block. See `README.md` for the full reproducibility guide and `tables.md` for every result table referenced below. -->

---

## Abstract

Intensive-care monitoring produces a heterogeneous stream of physiological measurements, laboratory observations, and clinical documentation. These modalities differ substantially in sampling frequency, temporal regularity, availability mechanism, and information latency. Deterioration forecasting in the ICU is therefore not merely a supervised classification problem: it is a prediction problem over a partially observed clinical process, in which the information available at prediction time must be distinguished from the complete retrospective record of the same clinical state.

This work develops a leakage-controlled multimodal forecasting pipeline using the eICU Collaborative Research Database Demo. The prediction endpoint is the first onset of sustained hypotension, sustained hypoxemia, or ICU death. Predictions are generated at hourly anchors for 1-, 3-, and 6-hour horizons. The final cohort contains 1,565 ICU stays from 1,294 patients, split by patient into training/validation/test partitions.

Physiological, laboratory, textual, and static information are represented separately before temporal multimodal fusion: physiology into a 35-dimensional representation, laboratory into an 80-dimensional representation (16 variables × 5 temporal statistics), clinical notes into a 66-dimensional TF-IDF/SVD representation, and static information into a 15-dimensional stay-level representation. The multimodal network contains 172,070 trainable parameters.

The principal experimental result is that **physiological information dominates predictive performance** in the current cohort. At the 6-hour horizon, physiology-only prediction achieves an AUPRC of 0.1266, compared with 0.1175 for the proposed multimodal model and 0.1034 for naive multimodal concatenation. A 2,000-replicate ICU-stay bootstrap yields a 95% confidence interval of $[0.0592, 0.1954]$ for the multimodal AUPRC. Paired bootstrap differences relative to physiology-only, naive concatenation, and logistic regression all contain zero — the experiments do not establish statistically distinguishable superiority of multimodal fusion in this cohort.

The system also separates risk prediction from alarm generation. At one experimentally selected operating point, it achieves 84.9% event sensitivity and a median warning time of 5.18 hours, but produces 11.44 alarms per patient-day at only 4.49% alarm precision — the result illustrates, rather than resolves, the alarm-fatigue problem.

The principal contribution is therefore methodological: a complete forecasting pipeline in which endpoint definition, temporal availability, patient-level splitting, modality missingness, model comparison, alarm policy, robustness testing, explainability, and statistical uncertainty are each treated as separate, explicit scientific objects.

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Clinical Endpoint](#2-clinical-endpoint)
3. [Horizon-Specific Prediction](#3-horizon-specific-prediction)
4. [Cohort Construction](#4-cohort-construction)
5. [Patient-Level Partitioning](#5-patient-level-partitioning)
6. [Temporal Discretization](#6-temporal-discretization)
7. [Leakage Control for Clinical Notes](#7-leakage-control-for-clinical-notes)
8. [Physiological Modality](#8-physiological-modality)
9. [Laboratory Modality](#9-laboratory-modality)
10. [Text Modality](#10-text-modality)
11. [Static Modality](#11-static-modality)
12. [Multimodal Representation and Architecture](#12-multimodal-representation-and-architecture)
13. [Baseline Hierarchy](#13-baseline-hierarchy)
14. [Evaluation Metric](#14-evaluation-metric)
15. [Predictive Results](#15-predictive-results)
16. [Ablation Analysis](#16-ablation-analysis)
17. [Clustered Statistical Uncertainty](#17-clustered-statistical-uncertainty)
18. [Bootstrap Results](#18-bootstrap-results)
19. [Paired Bootstrap Comparison](#19-paired-bootstrap-comparison)
20. [Alarm Policy](#20-alarm-policy)
21. [Alarm Results](#21-alarm-results)
22. [Missing-Modality Analysis](#22-missing-modality-analysis)
23. [Stress Testing](#23-stress-testing)
24. [Explainability](#24-explainability)
25. [Scientific Interpretation](#25-scientific-interpretation)
26. [Limitations](#26-limitations)
27. [Reproducibility Architecture](#27-reproducibility-architecture)
28. [Conclusion](#28-conclusion)

---

## 1. Problem Formulation

Let an ICU stay be indexed by $i$. The underlying patient state evolves continuously in time, $S_i(t) \in \mathcal S$, but is never observed directly. Instead, three asynchronous measurement processes generate observations — physiology, laboratory results, and clinical text — together with a static, time-invariant vector:

<!-- $$
\mathcal X_i = \left\{ X_i^{(p)}(t),\ X_i^{(l)}(t),\ X_i^{(n)}(t),\ X_i^{(s)} \right\}.
$$ -->

$$
\mathcal{X}_i =
\left\lbrace
X_i^{(p)}(t),
X_i^{(l)}(t),
X_i^{(n)}(t),
X_i^{(s)}
\right\rbrace.
$$

The central forecasting quantity is the conditional onset probability

$$
p_{i,h}(t) = \Pr\big( Y_{i,h}(t) = 1 \mid \mathcal F_i(t) \big),
$$

where $\mathcal F_i(t)$ is the information available at time $t$. The distinction between $\mathcal F_i(t)$ and the complete retrospective record is essential: if a clinical note describes an event occurring at time $t$ but is entered into the database at time $t + \delta$, that note cannot legitimately belong to $\mathcal F_i(t)$ — it belongs to $\mathcal F_i(t+\delta)$. The data pipeline treats information availability as part of the prediction problem, not an afterthought.

---

## 2. Clinical Endpoint

Several deterioration definitions were audited before the prediction task was frozen (Table 3 in `tables.md`). The selected endpoint is

$$
T_i^{\text{collapse}} = \min\left( T_i^{\text{hypo}},\ T_i^{\text{hypox}},\ T_i^{\text{death}} \right),
$$

where the **hypotension** onset is the first time $\text{MAP}_i(t) < 55$ mmHg is sustained for at least 30 minutes, the **hypoxemia** onset is the first time $\text{SpO}_{2,i}(t) < 88\%$ is sustained for at least 30 minutes, and **ICU death** is the third event type. The endpoint therefore represents the first clinically significant deterioration process detectable by the available monitoring signals.

---

## 3. Horizon-Specific Prediction

At anchor time $t$, define

$$
Y_{i,h}(t) = \mathbf 1\big[\, t < T_i^{\text{collapse}} \le t+h \,\big].
$$

The model estimates $\hat p_{i,h}(t) = f_\theta\big(\mathcal F_i(t)\big)_h$ for horizons $h \in \{1, 3, 6\}$ hours, producing a vector

$$
\hat{\mathbf p}_i(t) = \begin{bmatrix} \hat p_{i,1}(t) \\ \hat p_{i,3}(t) \\ \hat p_{i,6}(t) \end{bmatrix}.
$$

The task is consequently multi-horizon forecasting rather than a single binary prediction.

---

## 4. Cohort Construction

The raw eICU Demo contains 2,520 ICU stays. The final cohort is obtained through sequential eligibility constraints, the principal one being ICU length of stay $L_i^{\text{ICU}} > 24$ hours. Additional constraints enforce valid identifiers, plausible temporal ordering, sufficient early physiological observations, and valid event timing. The resulting cohort is $N_{\text{stays}} = 1565$, $N_{\text{patients}} = 1294$ (full funnel in `tables.md`, Table 4). The distinction between stays and patients matters because one patient can contribute multiple ICU stays.

---

## 5. Patient-Level Partitioning

Let $\mathcal P = \{1, \dots, N_{\text{patients}}\}$. The patient set is partitioned as $\mathcal P = \mathcal P_{\text{train}} \cup \mathcal P_{\text{val}} \cup \mathcal P_{\text{test}}$, with $\mathcal P_a \cap \mathcal P_b = \emptyset$ for $a \neq b$, and **all** ICU stays belonging to a patient remain in the same partition. The resulting partition contains 1,086 training stays, 234 validation stays, and 245 test stays (Table 5). This prevents a model from having implicitly seen a patient's physiological signature during training while being evaluated on that same patient's held-out stay.

---

## 6. Temporal Discretization

The underlying observations are irregular. Define hourly anchor times $t_{i,k} = t_{i,0} + 60k$. At each anchor, the feature constructor receives only the observation history up to that time, $\mathcal X_i^{\le t_{i,k}}$, corresponding to the information filtration

$$
\mathcal F_{i,k} = \sigma\big( \mathcal X_i(\tau) : \tau \le t_{i,k} \big).
$$

A valid feature constructor $\Phi$ must satisfy $\Phi\big(\mathcal X_i^{\le t_{i,k}}\big) = \Phi_{i,k}$ and must **not** depend on $\mathcal X_i(\tau)$ for any $\tau > t_{i,k}$. The implementation performs explicit truncation tests to verify this property on randomly selected stay/hour pairs (20 pairs checked, all passed, tolerance $10^{-5}$).

---

## 7. Leakage Control for Clinical Notes

Clinical documentation is a particularly important temporal leakage source. Let $t_i^{\text{clinical}}$ be the clinical timestamp of a note and $t_i^{\text{entry}}$ be its database documentation timestamp. The audit found that approximately **98% of notes were entered after their clinical timestamp**, with a median delay of approximately 12 minutes. Therefore

$$
t_i^{\text{available}} = t_i^{\text{entry}},
$$

and a note is included at anchor $t$ only if $t_i^{\text{entry}} \le t$. Using $t_i^{\text{clinical}}$ instead would let the forecasting model access retrospective documentation before that information was actually available. This is a general principle:

$$
\text{prediction time} \ \neq\ \text{clinical-event time} \ \neq\ \text{documentation time},
$$

and the pipeline must respect the first quantity when constructing the information set.

---

## 8. Physiological Modality

The physiological stream contains $X^{(p)} = [\text{HR}, \text{RR}, \text{SpO}_2, \text{Temp}, \text{SBP}, \text{DBP}, \text{MAP}]$. The cleaned dataset contains approximately 1.72 million observations, and the hourly feature representation has dimension $d_p = 35$. Physiological measurements are relatively dense compared with the other modalities — HR, SpO₂, blood-pressure variables, and respiratory rate all have high stay-level coverage (Table 6), while temperature is considerably sparser (8.9% stay coverage). This density difference is important when interpreting the modality ablation results in §16.

---

## 9. Laboratory Modality

Sixteen laboratory variables are used. For variable $j$, let $\mathcal L_j(t) = \{ x_{j,r} : t_{j,r} \le t \}$ be its trailing observation set. The feature constructor computes a five-dimensional summary $\phi_j(t) = [\phi_{j,1}(t), \dots, \phi_{j,5}(t)]$ (latest value, freshness, 24-hour slope, log-count, ever-measured flag). Across 16 laboratory variables, $d_l = 16 \times 5 = 80$. The laboratory stream is therefore represented as a sparse, temporally summarized process rather than a dense, regularly sampled vector.

---

## 10. Text Modality

Clinical notes are represented using character-level TF-IDF followed by truncated SVD. The transform is fit exclusively on the training-note corpus $D_{\text{train}}$, producing an embedding map $\phi_{\text{text}} : \mathcal D \rightarrow \mathbb R^{64}$ (padded with freshness and log-count to $d_t = 66$). The model never consumes raw note strings directly.

A major empirical limitation is the low temporal density of text: only approximately **2% of hourly rows** contain a newly available note, and the encoder was fit on just 225 unique training notes. This directly explains why text-only prediction is weak in the current experiment — it does **not** establish that clinical text is intrinsically uninformative.

---

## 11. Static Modality

Static variables $X^{(s)} \in \mathbb R^{15}$ include demographic, admission, missingness-indicator, pre-ICU-duration, sex, and ICU-unit information. Unlike the other modalities, these are associated with the stay rather than a particular temporal observation, so $X_i^{(s)}$ is available throughout the temporal sequence for stay $i$.

---

## 12. Multimodal Representation and Architecture

At each anchor $t_k$, define $P_k \in \mathbb R^{35}$, $L_k \in \mathbb R^{80}$, $N_k \in \mathbb R^{66}$, giving a combined temporal feature dimension of $35 + 80 + 66 = 181$, plus the static representation $S \in \mathbb R^{15}$.

Each modality is transformed into a learned representation,

$$
H^{(p)} = f_p(P_{1:T}), \qquad H^{(l)} = f_l(L_{1:T}), \qquad H^{(n)} = f_n(N_{1:T}),
$$

and a fusion operator constructs $Z = f_{\text{fusion}}\big(H^{(p)}, H^{(l)}, H^{(n)}, S\big)$, from which the final risk head produces $\hat{\mathbf p} = g(Z) \in [0,1]^3$.

Concretely:

- **Physiology encoder** — a causal temporal convolutional network (TCN): three residual blocks of causally left-padded dilated 1-D convolutions (dilations 1, 2, 4), giving a receptive field that comfortably covers the full 24-hour lookback without ever looking ahead.
- **Laboratory encoder** — a two-layer MLP applied row-wise (an instantaneous encoder, since the lab features already encode 24-hour slope/freshness internally).
- **Text encoder** — a linear projection of the frozen 66-dimensional note embedding into the shared representation space (no fine-tuning).
- **Causal cross-modal attention** — physiology attends causally to text and labs; labs attend causally to text; each attention operation is jointly masked by *causality* (query position $t$ may only attend to source positions $\le t$) and by *modality availability* (a source row with a zero modality-mask entry is excluded from the softmax), with a learned null token keeping the softmax well-defined even when a modality is absent for an entire sample.
- **Missingness-aware reliability gate** — a per-modality scalar reliability score is computed, masked to $-\infty$ for absent modalities, and renormalized only over the modalities present at that row:

$$
Z_t = \sum_{k} \alpha_{t,k}\, \mathcal Z_{t,k}, \qquad \alpha_{t,k} = \frac{m_{t,k}\, e^{\tilde s_{t,k}}}{\sum_j m_{t,j}\, e^{\tilde s_{t,j}}}.
$$

  If only physiology is available at a row, $\alpha_t$ collapses to a one-hot vector on physiology, rather than implicitly treating a zero-imputed missing modality as weak evidence.
- **GRU longitudinal state** — a single-layer GRU consumes $[Z_t; m_t]$ across the 24-hour window, initialized from the static representation.
- **Output heads** — three sigmoid heads (1h / 3h / 6h) from the final GRU hidden state concatenated with the static embedding.

The implemented model contains **172,070 trainable parameters**. The exact PyTorch graph is defined in `src/nets.py`; this report intentionally distinguishes the mathematical abstraction of the architecture from its exact implementation.

---

## 13. Baseline Hierarchy

The experimental design deliberately establishes a hierarchy of increasingly complex models:

| Model | Definition |
|---|---|
| B0 — Logistic regression | $\hat p = \sigma(w^\top x + b)$, $\sigma(z) = \frac{1}{1+e^{-z}}$, on a flat temporally-collapsed feature vector |
| B1 — Physiology-only | $\hat p_h = f_{\theta_p}(P_{1:T}, S)$ |
| B2 — Laboratory-only | $\hat p_h = f_{\theta_l}(L_{1:T}, S)$ |
| B3 — Text-only | $\hat p_h = f_{\theta_n}(N_{1:T}, S)$ |
| B4 — Naive concatenation | $X_k = [P_k; L_k; N_k]$, followed by a shared temporal model |
| Proposed — Multimodal fusion | Separate modality representations, fused as in §12 |

This hierarchy answers three distinct questions: (1) Is there predictive signal at all? (2) Which modality contains that signal? (3) Does explicit multimodal representation provide incremental predictive value over simpler alternatives?

---

## 14. Evaluation Metric

For a rare-event forecasting problem, accuracy is uninformative; the principal metric is **AUPRC**. Given precision $\text{Precision} = TP / (TP+FP)$ and recall $\text{Recall} = TP / (TP+FN)$ traced across thresholds,

$$
\text{AUPRC} = \int_0^1 P(R)\, dR.
$$

The 6-hour test prevalence is $\pi = 322 / 12485 \approx 0.0258$ — a highly imbalanced event.

---

## 15. Predictive Results

| Model | 1h | 3h | 6h |
|---|---:|---:|---:|
| Logistic | 0.0125 | 0.0483 | 0.0880 |
| **Physiology-only** | **0.0918** | **0.1012** | **0.1266** |
| Laboratory-only | 0.0060 | 0.0278 | 0.0527 |
| Text-only | 0.0036 | 0.0136 | 0.0259 |
| Naive concatenation | 0.0480 | 0.0770 | 0.1034 |
| Multimodal | 0.0215 | 0.0833 | 0.1175 |

Physiology-only prediction is the strongest model among those evaluated at every horizon. The proposed multimodal model does **not** outperform the physiology-only model on the held-out test set — a scientifically informative result, since the system was explicitly designed to test whether heterogeneous modalities provide incremental predictive information.

---

## 16. Ablation Analysis

| Modalities | AUPRC (6h) |
|---|---:|
| Physiology + text | 0.1256 |
| Physiology | 0.1243 |
| Physiology + laboratory | 0.1180 |
| Physiology + laboratory + text | 0.1175 |
| Laboratory | 0.0470 |
| Laboratory + text | 0.0456 |
| Text | 0.0291 |

Physiological information is the dominant predictive component. This does **not** imply laboratories and notes have no clinical value — it implies only that, in this experiment,

$$
\Delta_{\text{observed}} = \text{AUPRC}_{\text{multimodal}} - \text{AUPRC}_{\text{physiology}}
$$

is not positive. Possible explanations include sparse modality availability, low temporal density of notes, a lightweight text representation, limited cohort size, overlap between laboratory and physiological information, limited model capacity, and the composite-endpoint construction itself. The correct scientific interpretation is dataset- and implementation-specific, not a general statement about clinical text or labs.

---

## 17. Clustered Statistical Uncertainty

Hourly observations within an ICU stay are not independent: for stay $i$, the observations $\{(y_{ik}, \hat p_{ik})\}_{k \in \mathcal A_i}$ across its anchor set $\mathcal A_i$ share patient- and stay-level dependencies. Treating all 12,485 test anchors as independent would underestimate uncertainty, so the **bootstrap unit is the ICU stay**.

For bootstrap replicate $b$, sample $I_1^{(b)}, \dots, I_N^{(b)} \sim \text{Uniform}(\{1, \dots, N\})$ with replacement, retaining all anchors belonging to each selected stay. The statistic becomes

$$
\hat\theta^{(b)} = \text{AUPRC}\left( \bigcup_{r=1}^{N} \mathcal A_{I_r^{(b)}} \right),
$$

and with $B = 2000$ replicates, the percentile confidence interval is $\big[ Q_{0.025}(\hat\theta^{(1:B)}),\ Q_{0.975}(\hat\theta^{(1:B)}) \big]$.

---

## 18. Bootstrap Results

| Model | AUPRC | 95% CI |
|---|---:|---:|
| Multimodal | 0.1175 | $[0.0592, 0.1954]$ |
| Physiology-only | 0.1266 | $[0.0557, 0.2473]$ |
| Naive concatenation | 0.1034 | $[0.0553, 0.1802]$ |
| Logistic | 0.0880 | $[0.0453, 0.1509]$ |

---

## 19. Paired Bootstrap Comparison

For two models $A$ and $B$, define $\Delta^{(b)} = \hat\theta_A^{(b)} - \hat\theta_B^{(b)}$ using the *same* bootstrap stay sample for both models — a paired comparison that controls for which stays happened to be resampled.

| Comparison | $\Delta$ | 95% CI |
|---|---:|---:|
| Multimodal vs. physiology-only | $-0.0091$ | $[-0.0980, +0.0575]$ |
| Multimodal vs. naive concatenation | $+0.0141$ | $[-0.0496, +0.0885]$ |
| Multimodal vs. logistic regression | $+0.0294$ | $[-0.0145, +0.0862]$ |

Every interval contains zero. The appropriate conclusion is:

> The observed test-set differences do not establish statistically distinguishable performance between the multimodal model and the evaluated comparison models.

This is preferable to assigning significance to point-estimate differences that are uncertain at the available sample size.

---

## 20. Alarm Policy

Risk prediction and alarm generation are distinct functions. Let $p_t$ be the model's risk estimate. A threshold-only controller, $A_t = \mathbf 1[p_t \ge \tau]$, would react to isolated fluctuations, so the implemented controller additionally requires $p_t \ge \tau$ for $k = 2$ consecutive temporal predictions, followed by a cooldown period $\Delta_c = 60$ minutes after each alarm. The evaluated threshold is $\tau = 0.05$, selected on validation and frozen before test evaluation.

---

## 21. Alarm Results

On the test cohort: $N_{\text{alarms}} = 5950$, an alarm rate of $11.44$ per patient-day, event sensitivity of $62/73 = 0.8493$, and a median warning time of $5.18$ hours. Alarm precision is only $0.0449$ — approximately **95.5% of generated alarms do not correspond to a detected endpoint event**. This is critical: the system demonstrates that forecasting performance and alarm usability are not interchangeable. A model can produce useful ranking information while still producing an operationally unacceptable alarm burden.

---

## 22. Missing-Modality Analysis

| Condition | AUPRC (6h) |
|---|---:|
| Full model | 0.1175 |
| Physiology removed | 0.0456 |
| Physiology + laboratory removed | 0.0291 |
| Physiology + text removed | 0.0470 |
| Text removed | 0.1180 |
| Laboratory removed | 0.1256 |

The largest degradation occurs when physiology is removed, providing direct experimental evidence that the model relies primarily on physiological observations. Conversely, the limited degradation from removing text is consistent with its sparse temporal availability.

---

## 23. Stress Testing

**Physiological noise.** With $X^{(p)}_{\text{perturbed}} = X^{(p)} + \epsilon$, $\epsilon \sim \mathcal N(0, \sigma^2)$, AUPRC decreases smoothly as $\sigma$ increases from $0 \to 0.1 \to 0.25 \to 0.5 \to 1 \to 2$: $0.1175 \to 0.1170 \to 0.1138 \to 0.1046 \to 0.0849 \to 0.0627$.

**Sensor blackout.** Removing a contiguous interval of recent physiological observations of duration $B \in \{0.5, 1, 2, 4\}$ hours decreases AUPRC from $0.1088$ (0.5–1h blackout) to $0.0992$ (4h blackout).

Both degradations are smooth and monotone — evidence against pathological reliance on a narrow, brittle slice of the physiology signal.

---

## 24. Explainability

For model $f$, define the modality perturbation contribution $\Delta_m = f(x) - f(x_{\setminus m})$, where $x_{\setminus m}$ is the input with modality $m$ removed or suppressed. This answers *"how much does the model's output change when this modality is removed?"* — it does **not** answer *"how much did this modality causally change the patient's physiological state?"* This distinction matters in clinical machine learning.

Across 50 explained alarm samples, 32 were classified as physiology-dominant and 18 as laboratory-dominant. Median contributions: $\text{median}(\Delta_{\text{phys}}) = 0.0308$, $\text{median}(\Delta_{\text{lab}}) = -0.0339$, $\text{median}(\Delta_{\text{text}}) = 0$. These are model explanations, not clinical causal statements.

---

## 25. Scientific Interpretation

The experiment supports five principal observations:

1. **Temporal leakage is a first-order concern.** Clinical availability and clinical timestamp are not equivalent; the note audit shows retrospective documentation would otherwise enter the feature set before it was operationally available.
2. **Physiological measurements dominate this cohort.** The physiology-only model produces the highest AUPRC across every evaluated horizon.
3. **Multimodal fusion is not automatically beneficial.** The multimodal model does not exceed physiology-only performance on the held-out test cohort.
4. **The uncertainty is substantial.** Paired bootstrap intervals contain zero for every evaluated multimodal comparison.
5. **Prediction quality and alarm quality are different objectives.** An operating point with 84.9% event sensitivity still produces 11.44 alarms per patient-day and only 4.49% precision.

These observations jointly motivate a system design in which data construction, predictive modeling, uncertainty quantification, and alarm policy are evaluated independently.

---

## 26. Limitations

- **Dataset size** — only 245 ICU stays in the held-out test cohort; the 6-hour endpoint contains 322 positive anchors, so uncertainty remains substantial.
- **Demo dataset** — the eICU *Demo* is used, not a large multi-center external validation cohort; results should not be read as estimates of general clinical performance.
- **Text sparsity** — only ~2% of hourly rows contain a newly available note; weak text-only performance may reflect this rather than an inherent lack of information in clinical text.
- **Representation choice** — the text representation (TF-IDF → SVD) is intentionally lightweight; more expressive clinical-language representations may extract information unavailable to this representation.
- **Endpoint definition** — the composite endpoint intentionally combines different physiological deterioration mechanisms, improving event availability at the cost of target homogeneity.
- **Alarm policy** — the evaluated operating point is experimental, not clinically validated.
- **External validation** — none has been performed; a deployable system would require temporal, institutional, and ideally prospective external validation.

---

## 27. Reproducibility Architecture

The pipeline is implemented as ordered stages, $00 \rightarrow 01 \rightarrow \dots \rightarrow 12$:

| Stage | Responsibility |
|---|---|
| 00 | Raw data integrity |
| 01 | Schema and modality audit |
| 02 | Endpoint selection and freeze |
| 03 | Cohort construction and patient split |
| 04 | Temporal physiological representation |
| 05 | Label construction |
| 06 | Feature tensors + causal truncation test |
| 07 | Baselines |
| 08 | Multimodal model |
| 09 | Evaluation |
| 10 | Alarm policy and missing-modality analysis |
| 11 | Explanation and stress testing |
| 12 | Statistical uncertainty (cluster bootstrap) |

The ordering is intentional: **endpoint → cohort → temporal data → features → models → evaluation**, rather than model → metric → retrofitted task. Full run instructions are in `README.md`.

---

## 28. Conclusion

The ICU Monitor Dilemma experiment demonstrates a complete pipeline for forecasting deterioration from heterogeneous ICU data while explicitly addressing temporal leakage, irregular sampling, missing modalities, patient-level dependence, alarm generation, and statistical uncertainty.

The strongest empirical signal in the present cohort originates from physiological observations. The multimodal architecture remains useful as an experimental framework — it provides a principled mechanism for incorporating heterogeneous information — but the current test cohort does not establish that multimodal fusion improves predictive discrimination over physiology-only modeling.

The central methodological conclusion is therefore not that one architecture has solved ICU alarm fatigue, but rather:

> **Reliable ICU forecasting requires causal temporal data construction, modality-aware modeling, uncertainty quantification, and operational alarm analysis — together, not separately.**

The resulting system is best interpreted as a rigorously evaluated research prototype. Its next scientific step is external validation, richer modeling of asynchronous modalities, calibration and utility analysis, and prospective evaluation of alarm burden.