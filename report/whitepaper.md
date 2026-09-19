# The ICU Monitor Dilemma

## Leakage-Controlled Multimodal Forecasting of Physiological Deterioration from Asynchronous ICU Data

---

# Abstract

Intensive-care monitoring produces a heterogeneous stream of physiological measurements, laboratory observations, and clinical documentation. These modalities differ substantially in sampling frequency, temporal regularity, availability mechanisms, and information latency. Consequently, deterioration forecasting in the ICU is not merely a supervised classification problem; it is a prediction problem over a partially observed clinical process in which the information available at prediction time must be distinguished from information subsequently documented about the same clinical state.

This work develops a leakage-controlled multimodal forecasting pipeline using the eICU Collaborative Research Database Demo. The prediction endpoint is defined as the first onset of sustained hypotension, sustained hypoxemia, or ICU death. Predictions are generated at hourly anchors for 1-, 3-, and 6-hour horizons. The final cohort contains 1,565 ICU stays from 1,294 patients and is divided using a patient-level train/validation/test partition.

Physiological, laboratory, textual, and static information are represented separately before temporal multimodal fusion. Physiological data are transformed into 35-dimensional temporal representations, laboratory data into 80-dimensional representations from 16 laboratory variables and five temporal statistics per variable, clinical notes into 66-dimensional TF-IDF/SVD representations, and static information into a 15-dimensional stay-level representation. The multimodal network contains 172,070 trainable parameters.

The principal experimental result is that physiological information dominates predictive performance in the current cohort. At the 6-hour horizon, physiology-only prediction achieves an AUPRC of 0.1266, compared with 0.1175 for the proposed multimodal model and 0.1034 for naive multimodal concatenation. A 2,000-replicate ICU-stay bootstrap yields a 95% confidence interval of [0.0592, 0.1954] for the multimodal AUPRC. Paired bootstrap differences relative to physiology-only, naive concatenation, and logistic regression all contain zero. Thus, the experiments do not establish statistically distinguishable superiority of multimodal fusion in this cohort.

The system also separates risk prediction from alarm generation. At one experimentally selected alarm operating point, the system achieves 84.9% event sensitivity and a median warning time of 5.18 hours, but produces 11.44 alarms per patient-day with only 4.49% alarm precision. The result therefore illustrates rather than resolves the alarm-fatigue problem.

The principal contribution is consequently methodological: a complete forecasting pipeline in which endpoint definition, temporal availability, patient-level splitting, modality missingness, model comparison, alarm policy, robustness testing, explainability, and statistical uncertainty are treated as separate scientific objects.

---

# 1. Problem Formulation

Let an ICU stay be indexed by \(i\).

The underlying patient state evolves continuously in time:

$$
S_i(t)\in\mathcal S.
$$

The state is not directly observed. Instead, multiple clinical measurement processes generate observations:

$$
X_i^{(p)}(t),
\qquad
X_i^{(l)}(t),
\qquad
X_i^{(n)}(t),
$$

corresponding respectively to physiology, laboratory measurements, and clinical text.

Static information is represented by

$$
X_i^{(s)}.
$$

The complete observation process can therefore be written abstractly as

$$
\mathcal X_i
=
\left\{
X_i^{(p)}(t),
X_i^{(l)}(t),
X_i^{(n)}(t),
X_i^{(s)}
\right\}.
$$

The central forecasting quantity is

$$
p_{i,h}(t)
=
P
\left(
Y_{i,h}(t)=1
\mid
\mathcal F_i(t)
\right),
$$

where \(\mathcal F_i(t)\) is the information available at time \(t\).

The distinction between \(\mathcal F_i(t)\) and the complete retrospective record is essential.

If a clinical note describes an event occurring at time \(t\) but was entered into the database at time \(t+\delta\), then the note cannot legitimately belong to

$$
\mathcal F_i(t).
$$

It belongs to

$$
\mathcal F_i(t+\delta).
$$

The data pipeline therefore treats information availability as part of the prediction problem.

---

# 2. Clinical Endpoint

Several possible deterioration definitions were audited before the prediction task was frozen.

The selected endpoint is

$$
T_i^{\mathrm{collapse}}
=
\min
\left(
T_i^{\mathrm{hypo}},
T_i^{\mathrm{hypox}},
T_i^{\mathrm{death}}
\right).
$$

### Hypotension

The hypotension onset is the first time at which

$$
\mathrm{MAP}_i(t)<55\text{ mmHg}
$$

is sustained for at least 30 minutes.

### Hypoxemia

The hypoxemia onset is the first time at which

$$
\mathrm{SpO}_{2,i}(t)<88\%
$$

is sustained for at least 30 minutes.

### ICU death

The third event type is ICU death.

The endpoint therefore represents the first clinically significant deterioration process detected by the available monitoring signals.

---

# 3. Horizon-Specific Prediction

At an anchor time \(t\), define

$$
Y_{i,h}(t)
=
\mathbf 1
\left[
t<T_i^{\mathrm{collapse}}\le t+h
\right].
$$

The model estimates

$$
\hat p_{i,h}(t)
=
f_\theta
\left(
\mathcal F_i(t)
\right)_h.
$$

The implemented horizons are

$$
h\in\{1,3,6\}\text{ hours}.
$$

Thus each temporal anchor produces a vector

$$
\hat{\mathbf p}_i(t)
=
\begin{bmatrix}
\hat p_{i,1}(t)\\
\hat p_{i,3}(t)\\
\hat p_{i,6}(t)
\end{bmatrix}.
$$

The task is consequently multi-horizon forecasting rather than a single binary prediction.

---

# 4. Cohort Construction

The raw eICU Demo contains 2,520 ICU stays.

The final cohort is obtained through sequential eligibility constraints.

The principal constraint is an ICU stay longer than 24 hours:

$$
L_i^{ICU}>24\text{ h}.
$$

Additional constraints enforce valid identifiers, plausible temporal ordering, sufficient early physiological observations, and valid event timing.

The resulting cohort is:

$$
N_{\mathrm{stays}}=1565,
$$

$$
N_{\mathrm{patients}}=1294.
$$

The distinction between stays and patients is important because one patient can contribute multiple ICU stays.

---

# 5. Patient-Level Partitioning

Let

$$
\mathcal P
=
\{1,\ldots,N_{\mathrm{patients}}\}.
$$

The patient set is partitioned:

$$
\mathcal P
=
\mathcal P_{\mathrm{train}}
\cup
\mathcal P_{\mathrm{val}}
\cup
\mathcal P_{\mathrm{test}},
$$

with

$$
\mathcal P_a\cap\mathcal P_b=\emptyset
$$

for \(a\neq b\).

All ICU stays belonging to a patient remain in the same partition.

The resulting partition contains:

$$
1086
$$

training stays,

$$
234
$$

validation stays, and

$$
245
$$

test stays.

This prevents a model from seeing the same patient's physiological signature during training and evaluation.

---

# 6. Temporal Discretization

The underlying observations are irregular.

Define hourly anchor times

$$
t_{i,k}
=
t_{i,0}+60k.
$$

At each anchor, the feature constructor receives the observation history

$$
\mathcal X_i^{\leq t_{i,k}}.
$$

The information filtration is therefore

$$
\mathcal F_{i,k}
=
\sigma
\left(
\mathcal X_i(\tau):
\tau\le t_{i,k}
\right).
$$

A valid feature constructor must satisfy:

$$
\Phi
\left(
\mathcal X_i^{\leq t_{i,k}}
\right)
=
\Phi_{i,k}.
$$

It must not depend on

$$
\mathcal X_i(\tau),
\qquad
\tau>t_{i,k}.
$$

The implementation explicitly performs truncation tests to verify this property on randomly selected stay/hour pairs.

---

# 7. Leakage Control for Clinical Notes

Clinical documentation is a particularly important temporal leakage source.

Let

$$
t_i^{clinical}
$$

be the clinical timestamp and

$$
t_i^{entry}
$$

be the database documentation timestamp.

The audit found that approximately 98% of notes were entered after their clinical timestamp, with a median delay of approximately 12 minutes.

Therefore:

$$
t_i^{available}=t_i^{entry}.
$$

A note is included at anchor \(t\) only if

$$
t_i^{entry}\le t.
$$

Using \(t_i^{clinical}\) instead would allow the forecasting model to access retrospective documentation before that information was actually available.

This is a general principle:

$$
\boxed{
\text{prediction time}
\neq
\text{clinical-event time}
\neq
\text{documentation time}
}
$$

and the prediction pipeline must respect the first quantity when constructing the information set.

---

# 8. Physiological Modality

The physiological stream contains:

$$
X^{(p)}
=
[
HR,
RR,
SpO_2,
Temp,
SBP,
DBP,
MAP
].
$$

The cleaned dataset contains approximately 1.72 million observations.

The hourly feature representation has dimension

$$
d_p=35.
$$

Physiological measurements are relatively dense compared with the other modalities. In particular, HR, SpO2, blood-pressure variables, and respiratory rate have high stay-level coverage.

Temperature is considerably sparser.

This difference in observation density becomes important when interpreting the modality ablation results.

---

# 9. Laboratory Modality

Sixteen laboratory variables are used.

For each laboratory variable \(j\), temporal statistics are constructed.

Abstractly, let

$$
\mathcal L_j(t)
=
\{
x_{j,r}:t_{j,r}\le t
\}.
$$

The feature constructor computes a five-dimensional summary:

$$
\phi_j(t)
=
\left[
\phi_{j,1}(t),
\ldots,
\phi_{j,5}(t)
\right].
$$

Across 16 laboratory variables:

$$
d_l=16\times5=80.
$$

The laboratory stream is therefore represented as a sparse, temporally summarized process rather than as a dense regularly sampled vector.

---

# 10. Text Modality

Clinical notes are represented using character-level TF-IDF followed by truncated SVD.

Let

$$
D_{\mathrm{train}}
$$

denote the training-note corpus.

The text transform is fitted exclusively on

$$
D_{\mathrm{train}},
$$

producing an embedding map

$$
\phi_{\mathrm{text}}:
\mathcal D
\rightarrow
\mathbb R^{66}.
$$

Thus

$$
d_t=66.
$$

The model does not directly consume raw note strings.

A major empirical limitation is the low temporal density of text: only approximately 2% of hourly rows contain a newly available note.

This provides a direct explanation for why text-only prediction is weak in the current experiment, although it does not establish that clinical text is intrinsically uninformative.

---

# 11. Static Modality

Static variables are represented by

$$
X^{(s)}\in\mathbb R^{15}.
$$

These include demographic, admission, missingness, pre-ICU duration, sex, and ICU-unit information.

Unlike the other modalities, these variables are associated with the stay rather than a particular temporal observation.

Consequently,

$$
X_i^{(s)}
$$

is available throughout the temporal sequence for stay \(i\).

---

# 12. Multimodal Representation

At each temporal anchor \(t_k\), define:

$$
P_k\in\mathbb R^{35},
$$

$$
L_k\in\mathbb R^{80},
$$

$$
N_k\in\mathbb R^{66}.
$$

The combined temporal feature space therefore has dimension

$$
35+80+66=181.
$$

The static representation is

$$
S\in\mathbb R^{15}.
$$

The raw conceptual multimodal input can therefore be represented as

$$
X_{1:T}
=
\{
(P_k,L_k,N_k)
\}_{k=1}^{T},
\qquad
S.
$$

The model transforms each modality into learned representations:

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
f_n(N_{1:T}).
$$

A fusion operator then constructs

$$
Z
=
f_{\mathrm{fusion}}
\left(
H^{(p)},
H^{(l)},
H^{(n)},
S
\right).
$$

The final risk head produces

$$
\hat{\mathbf p}
=
g(Z)
\in[0,1]^3.
$$

The implemented model contains 172,070 trainable parameters.

The exact neural-layer equations are implementation-specific and should be read directly from `src/nets.py`; this paper intentionally distinguishes the mathematical abstraction of the architecture from its exact PyTorch graph.

---

# 13. Baseline Hierarchy

The experimental design deliberately establishes a hierarchy of increasingly complex models.

### Baseline 0: Logistic regression

A flat representation is mapped to

$$
\hat p
=
\sigma(w^\top x+b),
$$

where

$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

### Physiology-only

$$
\hat p_h
=
f_{\theta_p}(P_{1:T},S).
$$

### Laboratory-only

$$
\hat p_h
=
f_{\theta_l}(L_{1:T},S).
$$

### Text-only

$$
\hat p_h
=
f_{\theta_n}(N_{1:T},S).
$$

### Naive concatenation

$$
X_k
=
[P_k;L_k;N_k],
$$

followed by a shared temporal model.

### Multimodal fusion

Separate modality representations are learned before fusion.

This hierarchy answers three distinct questions:

1. Is there predictive signal at all?
2. Which modality contains that signal?
3. Does explicit multimodal representation provide incremental predictive value?

---

# 14. Evaluation Metric

For a rare event forecasting problem, accuracy is not informative.

The principal metric is AUPRC.

Given a ranking induced by model scores, precision is

$$
\mathrm{Precision}
=
\frac{TP}{TP+FP},
$$

and recall is

$$
\mathrm{Recall}
=
\frac{TP}{TP+FN}.
$$

The precision-recall curve traces precision as the decision threshold varies.

The area under this curve is

$$
\mathrm{AUPRC}
=
\int_0^1
P(R)\,dR.
$$

The 6-hour test prevalence is

$$
\pi
=
\frac{322}{12485}
\approx
0.0258.
$$

Thus the event is highly imbalanced.

---

# 15. Predictive Results

The test-set AUPRC values are:

| Model               |     1h |     3h |     6h |
| ------------------- | -----: | -----: | -----: |
| Logistic            | 0.0125 | 0.0483 | 0.0880 |
| Physiology-only     | 0.0918 | 0.1012 | 0.1266 |
| Laboratory-only     | 0.0060 | 0.0278 | 0.0527 |
| Text-only           | 0.0036 | 0.0136 | 0.0259 |
| Naive concatenation | 0.0480 | 0.0770 | 0.1034 |
| Multimodal          | 0.0215 | 0.0833 | 0.1175 |

The principal observation is that physiology-only prediction is strongest among the evaluated models at the 1-, 3-, and 6-hour horizons.

The proposed multimodal model does not outperform the physiology-only model on the held-out test set.

This is scientifically informative because the system was explicitly designed to test whether heterogeneous modalities provide incremental predictive information.

---

# 16. Ablation Analysis

The 6-hour ablation results are:

| Modalities                     |    AUPRC |
| ------------------------------ | -------: |
| Physiology + text              | 0.125589 |
| Physiology                     | 0.124310 |
| Physiology + laboratory        | 0.118020 |
| Physiology + laboratory + text | 0.117497 |
| Laboratory                     | 0.046988 |
| Laboratory + text              | 0.045569 |
| Text                           | 0.029144 |

The result indicates that physiological information is the dominant predictive component.

Importantly, this does not imply that laboratories and notes have no clinical value.

It implies only:

$$
\Delta_{\mathrm{observed}}
=
\mathrm{AUPRC}_{\mathrm{multimodal}}
-
\mathrm{AUPRC}_{\mathrm{physiology}}
$$

is not positive in this experiment.

Possible explanations include:

* sparse modality availability,
* low temporal density of notes,
* lightweight text representation,
* limited cohort size,
* overlap between laboratory and physiological information,
* limited model capacity,
* endpoint construction.

The correct scientific interpretation is therefore dataset- and implementation-specific.

---

# 17. Clustered Statistical Uncertainty

Hourly observations within an ICU stay are not independent.

For a stay \(i\), let

$$
\mathcal A_i
$$

be its collection of temporal anchors.

Then the observations

$$
\{(y_{ik},\hat p_{ik})\}_{k\in\mathcal A_i}
$$

share patient and stay-level dependencies.

Treating all 12,485 test anchors as independent would therefore underestimate uncertainty.

The bootstrap unit is consequently the ICU stay.

For bootstrap replicate \(b\), sample

$$
I_1^{(b)},\ldots,I_N^{(b)}
\sim
\mathrm{Uniform}
(\{1,\ldots,N\}),
$$

with replacement.

All anchors belonging to a selected stay are retained together.

The statistic becomes

$$
\widehat{\theta}^{(b)}
=
\mathrm{AUPRC}
\left(
\bigcup_{r=1}^{N}
\mathcal A_{I_r^{(b)}}
\right).
$$

With

$$
B=2000
$$

replicates, the percentile confidence interval is

$$
[
Q_{0.025}(\widehat\theta^{(1:B)}),
Q_{0.975}(\widehat\theta^{(1:B)})
].
$$

---

# 18. Bootstrap Results

The multimodal model obtains:

$$
\widehat{\mathrm{AUPRC}}=0.1175,
$$

with

$$
95\%\ CI=[0.0592,0.1954].
$$

The physiology-only model obtains:

$$
0.1266,
$$

with

$$
95\%\ CI=[0.0557,0.2473].
$$

Naive concatenation obtains:

$$
0.1034,
$$

with

$$
95\%\ CI=[0.0553,0.1802].
$$

Logistic regression obtains:

$$
0.0880,
$$

with

$$
95\%\ CI=[0.0453,0.1509].
$$

---

# 19. Paired Bootstrap Comparison

For two models \(A\) and \(B\), define

$$
\Delta^{(b)}
=
\widehat{\mathrm{AUPRC}}_A^{(b)}
-
\widehat{\mathrm{AUPRC}}_B^{(b)}.
$$

The same bootstrap stay sample is used for both models.

This controls for variation in the sampled ICU stays and produces a paired comparison.

### Multimodal vs physiology-only

$$
\Delta=-0.0091
$$

with

$$
95\%\ CI
=
[-0.0980,+0.0575].
$$

### Multimodal vs naive concatenation

$$
\Delta=+0.0141
$$

with

$$
95\%\ CI
=
[-0.0496,+0.0885].
$$

### Multimodal vs logistic regression

$$
\Delta=+0.0294
$$

with

$$
95\%\ CI
=
[-0.0145,+0.0862].
$$

Every interval contains zero.

Therefore the appropriate conclusion is:

> The observed test-set differences do not establish statistically distinguishable performance between the multimodal model and the evaluated comparison models.

This is preferable to assigning significance to point-estimate differences that are uncertain at the available sample size.

---

# 20. Alarm Policy

Risk prediction and alarm generation are distinct functions.

Let

$$
p_t
$$

be the model risk.

A threshold-only controller would generate

$$
A_t
=
\mathbf1[p_t\ge\tau].
$$

However, threshold-only alarms can react to isolated fluctuations.

The implemented controller requires

$$
p_t\ge\tau
$$

for

$$
k=2
$$

consecutive temporal predictions.

After triggering, alarms are suppressed during a cooldown period

$$
\Delta_c=60\text{ min}.
$$

The evaluated threshold is

$$
\tau=0.05.
$$

This converts the continuous prediction process into an operational alarm process.

---

# 21. Alarm Results

On the test cohort:

$$
N_{\mathrm{alarms}}=5950.
$$

The alarm rate is

$$
11.44
$$

alarms per patient-day.

The event sensitivity is

$$
\frac{62}{73}
=
0.8493.
$$

Median warning time is

$$
5.18\text{ h}.
$$

However, alarm precision is only

$$
0.0449.
$$

Thus approximately 95.5% of generated alarms do not correspond to a detected endpoint event under the operational definition.

This result is critical.

The system demonstrates that forecasting performance and alarm usability are not interchangeable.

A forecasting model can produce useful ranking information while still producing an operationally unacceptable alarm burden.

---

# 22. Missing-Modality Analysis

The model is evaluated under controlled removal and degradation of modalities.

The 6-hour AUPRC is:

| Condition                       |  AUPRC |
| ------------------------------- | -----: |
| Full model                      | 0.1175 |
| Physiology removed              | 0.0456 |
| Physiology + laboratory removed | 0.0291 |
| Physiology + text removed       | 0.0470 |
| Text removed                    | 0.1180 |
| Laboratory removed              | 0.1256 |

The largest degradation occurs when physiology is removed.

This provides experimental evidence that the model relies primarily on physiological observations.

Conversely, the limited degradation from removing text is consistent with the sparse availability of notes in the temporal representation.

---

# 23. Stress Testing

Two controlled perturbation families are evaluated.

## 23.1 Physiological noise

Let

$$
X^{(p)}_{\mathrm{perturbed}}
=
X^{(p)}
+
\epsilon,
$$

with

$$
\epsilon
\sim
\mathcal N(0,\sigma^2).
$$

As \(\sigma\) increases:

$$
0
\rightarrow
0.1
\rightarrow
0.25
\rightarrow
0.5
\rightarrow
1
\rightarrow
2,
$$

AUPRC decreases:

$$
0.1175
\rightarrow
0.1170
\rightarrow
0.1138
\rightarrow
0.1046
\rightarrow
0.0849
\rightarrow
0.0627.
$$

This establishes controlled degradation under increasingly severe physiological corruption.

## 23.2 Sensor blackout

A contiguous interval of recent physiological observations is removed.

For blackout duration \(B\),

$$
B\in
\{0.5,1,2,4\}\text{ h}.
$$

AUPRC decreases from

$$
0.1088
$$

for 0.5/1-hour blackout to

$$
0.0992
$$

for a 4-hour blackout.

---

# 24. Explainability

For a model \(f\), define the modality perturbation contribution

$$
\Delta_m
=
f(x)-f(x_{\setminus m}),
$$

where \(x_{\setminus m}\) represents an input in which modality \(m\) has been removed or suppressed.

This quantity answers:

> How much does the model's output change when this modality is removed?

It does **not** answer:

> How much did this modality causally change the patient's physiological state?

This distinction is important in clinical machine learning.

Across 50 explained alarm samples, 32 were classified as physiology-dominant and 18 as laboratory-dominant under the implemented attribution rule.

The median contributions were:

$$
\operatorname{median}(\Delta_{\mathrm{phys}})
=
0.0308,
$$

$$
\operatorname{median}(\Delta_{\mathrm{lab}})
=
-0.0339,
$$

$$
\operatorname{median}(\Delta_{\mathrm{text}})
=
0.
$$

These values should therefore be interpreted as model explanations rather than clinical causal statements.

---

# 25. Scientific Interpretation

The experiment supports five principal observations.

### Observation 1: Temporal leakage is a first-order concern

Clinical availability and clinical timestamp are not equivalent.

The note audit demonstrates that retrospective documentation can otherwise enter the feature set before it was operationally available.

### Observation 2: Physiological measurements dominate this cohort

The physiology-only model produces the highest observed AUPRC across the evaluated horizons.

### Observation 3: Multimodal fusion is not automatically beneficial

The multimodal model does not exceed the physiology-only model on the held-out test cohort.

### Observation 4: The uncertainty is substantial

The paired bootstrap intervals contain zero for all evaluated multimodal comparisons.

### Observation 5: Prediction quality and alarm quality are different objectives

An operating point with 84.9% event sensitivity still produces 11.44 alarms per patient-day and only 4.49% precision.

These observations jointly motivate a system design in which data construction, predictive modeling, uncertainty quantification, and alarm policy are evaluated independently.

---

# 26. Limitations

## 26.1 Dataset size

Only 245 ICU stays are available in the held-out test cohort.

The 6-hour endpoint contains 322 positive anchor observations.

Therefore uncertainty remains substantial.

## 26.2 Demo dataset

The experiments use the eICU Demo rather than a large multi-center external validation cohort.

The results should therefore not be interpreted as estimates of clinical performance in a general ICU population.

## 26.3 Text sparsity

Although notes are available for many stays, only approximately 2% of hourly rows contain a newly available note.

The weak text-only performance may therefore reflect temporal sparsity and representation choice.

## 26.4 Representation choice

The text representation is intentionally lightweight:

$$
\mathrm{TFIDF}_{char}
\rightarrow
\mathrm{SVD}.
$$

More expressive clinical-language representations may extract information unavailable to the present representation.

## 26.5 Endpoint definition

The composite endpoint intentionally combines different physiological deterioration mechanisms.

This improves event availability but also creates a heterogeneous target.

## 26.6 Alarm policy

The evaluated alarm policy is an experimental operating point.

It is not clinically validated.

## 26.7 External validation

No external cohort is used.

A clinically deployable system would require temporal, institutional, and ideally prospective external validation.

---

# 27. Reproducibility Architecture

The complete pipeline is implemented as ordered stages:

$$
00\rightarrow01\rightarrow02\rightarrow03\rightarrow04\rightarrow05
\rightarrow06\rightarrow07\rightarrow08\rightarrow09\rightarrow10\rightarrow11\rightarrow12.
$$

The stages have distinct responsibilities.

| Stage | Responsibility                             |
| ----- | ------------------------------------------ |
| 00    | Raw data integrity                         |
| 01    | Schema and modality audit                  |
| 02    | Endpoint selection and freeze              |
| 03    | Cohort construction and patient split      |
| 04    | Temporal physiological representation      |
| 05    | Label construction                         |
| 06    | Feature tensors                            |
| 07    | Baselines                                  |
| 08    | Multimodal model                           |
| 09    | Evaluation                                 |
| 10    | Alarm policy and missing-modality analysis |
| 11    | Explanation and stress testing             |
| 12    | Statistical uncertainty                    |

The ordering is intentional.

In particular:

$$
\boxed{
\text{endpoint}
\rightarrow
\text{cohort}
\rightarrow
\text{temporal data}
\rightarrow
\text{features}
\rightarrow
\text{models}
\rightarrow
\text{evaluation}
}
$$

rather than

$$
\text{model}
\rightarrow
\text{metric}
\rightarrow
\text{retrofit task}.
$$

---

# 28. Conclusion

The ICU Monitor Dilemma experiment demonstrates a complete pipeline for forecasting deterioration from heterogeneous ICU data while explicitly addressing temporal leakage, irregular sampling, missing modalities, patient-level dependence, alarm generation, and statistical uncertainty.

The strongest empirical signal in the present cohort originates from physiological observations.

The multimodal architecture remains useful as an experimental framework because it provides a principled mechanism for incorporating heterogeneous information, but the current test cohort does not establish that multimodal fusion improves predictive discrimination over physiology-only modeling.

The central methodological conclusion is therefore not that one architecture has solved ICU alarm fatigue.

Rather:

$$
\boxed{
\text{Reliable ICU forecasting requires}
\;
\text{causal temporal data construction}
+
\text{modality-aware modeling}
+
\text{uncertainty quantification}
+
\text{operational alarm analysis}.
}
$$

The resulting system is best interpreted as a rigorously evaluated research prototype. Its next scientific step is external validation, richer modeling of asynchronous modalities, calibration and utility analysis, and prospective evaluation of alarm burden.
