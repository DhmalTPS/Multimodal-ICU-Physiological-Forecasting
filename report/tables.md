# Report Tables

<!-- > Math in this file uses GitHub's native `$...$` (inline) / `$$...$$` (display) syntax. Every table cell that needs a symbol uses `$...$`, never `\( ... \)`, since GitHub does not render the latter. -->

## Table 1 — Raw Data Audit

| Table | Rows | Required | Readable | Official SHA-256 |
|---|---:|:---:|:---:|:---:|
| `patient` | 2,520 | Yes | Yes | Yes |
| `vitalPeriodic` | 1,634,960 | Yes | Yes | Yes |
| `vitalAperiodic` | 274,088 | Yes | Yes | Yes |
| `lab` | 434,660 | Yes | Yes | Yes |
| `note` | 24,758 | Yes | Yes | Yes |
| `nurseCharting` | 1,477,163 | No | Yes | Yes |
| `nurseAssessment` | 91,589 | No | Yes | Yes |
| `physicalExam` | 84,058 | No | Yes | Yes |

All required tables were present, readable, schema-compatible, and matched the official checksums.

---

## Table 2 — Core Modality Coverage

| Modalities present | Stays | Percentage |
|---|---:|---:|
| None | 36 | 1.43% |
| Notes only | 9 | 0.36% |
| Labs only | 42 | 1.67% |
| Labs + notes | 53 | 2.10% |
| Vitals only | 13 | 0.52% |
| Vitals + notes | 18 | 0.71% |
| Vitals + labs | 172 | 6.83% |
| Vitals + labs + notes | 2,177 | 86.39% |

---

## Table 3 — Endpoint Audit

| Candidate endpoint | 1h positives | 3h positives | 6h positives | Decision |
|---|---:|---:|---:|---|
| Composite collapse | 292 | 1,259 | 2,457 | **Selected** |
| ICU death | 74 | 216 | 418 | Retained as candidate |
| Hospital death | 51 | 167 | 371 | Rejected |
| Severe hypotension | 116 | 545 | 1,093 | Retained as candidate |
| Severe hypoxemia | 130 | 607 | 1,192 | Retained as candidate |

**Locked endpoint:**

<!-- 
$$
T_{\text{collapse}} = \min\left\{ T_{\text{hypotension}},\ T_{\text{hypoxemia}},\ T_{\text{ICU death}} \right\}
$$

where $T_{\text{hypotension}}$ is the first sustained episode with $\text{MAP} < 55$ mmHg for $\ge 30$ minutes, and $T_{\text{hypoxemia}}$ is the first sustained episode with $\text{SpO}_2 < 88\%$ for $\ge 30$ minutes. -->

```math
T_{\text{collapse}}
=
\min\left(
T_{\text{hypotension}},
T_{\text{hypoxemia}},
T_{\text{ICU death}}
\right)
```

where $T_{\text{hypotension}}$ is the first sustained episode with $\mathrm{MAP} < 55,\mathrm{mmHg}$ lasting at least $30$ minutes, while $T_{\text{hypoxemia}}$ is the first sustained episode with $\mathrm{SpO}_2 < 88%$ lasting at least $30$ minutes.

---

## Table 4 — Cohort Funnel

| Eligibility stage | Stays | Patients |
|---|---:|---:|
| All ICU stays | 2,520 | 1,841 |
| Valid identifier + discharge | 2,520 | 1,841 |
| ICU LOS > 24 h | 1,626 | 1,332 |
| Known ICU discharge status | 1,626 | 1,332 |
| Plausible timeline | 1,622 | 1,328 |
| ≥ 12 vital observations in first 24 h | 1,566 | 1,294 |
| Plausible event timing | **1,565** | **1,294** |

---

## Table 5 — Patient-Level Data Split

| Split | Stays | Patients | Event stays | Candidate anchors | Median LOS (h) | ICU mortality | Event-stay rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 1,086 | 905 | 182 | 58,890 | 52.41 | 4.88% | 16.76% |
| Validation | 234 | 194 | 40 | 14,883 | 56.35 | 4.70% | 17.09% |
| Test | 245 | 195 | 40 | 12,562 | 51.08 | 4.90% | 16.33% |

---

## Table 6 — Physiological Coverage

| Variable | Observations | Stays with variable | Stay-hours observed | Mean | SD |
|---|---:|---:|---:|---:|---:|
| HR | 1,495,922 | 99.49% | 96.97% | 85.79 | 17.96 |
| RR | 1,256,951 | 92.33% | 83.82% | 20.36 | 6.30 |
| SpO₂ | 1,329,384 | 99.17% | 89.64% | 96.60 | 3.32 |
| Temperature | 106,249 | 8.88% | 7.26% | 37.12 | 0.88 |
| SBP | 432,660 | 99.42% | 87.78% | 122.84 | 25.64 |
| DBP | 432,609 | 99.42% | 87.78% | 62.35 | 15.61 |
| MAP | 435,401 | 99.42% | 87.91% | 81.01 | 17.86 |

---

## Table 7 — Feature Representation

| Modality | Representation | Dimension |
|---|---|---:|
| Physiology | Temporal physiological features (mean/min/max/observed-flag/time-since-observation × 7 channels) | $d_p = 35$ |
| Laboratory | 16 variables × 5 temporal statistics | $d_l = 80$ |
| Text | Character TF-IDF + truncated SVD | $d_t = 66$ |
| Static | Demographic / admission / ICU-unit features | $d_s = 15$ |

Total temporal multimodal input dimensionality:

$$
d_{\text{temporal}} = d_p + d_l + d_t = 35 + 80 + 66 = 181.
$$

---

## Table 8 — Label Prevalence

| Split | 1h positives | 1h prevalence | 3h positives | 3h prevalence | 6h positives | 6h prevalence |
|---|---:|---:|---:|---:|---:|---:|
| Train | 210 | 0.36% | 903 | 1.55% | 1,771 | 3.03% |
| Validation | 38 | 0.26% | 178 | 1.20% | 349 | 2.36% |
| Test | 41 | 0.33% | 169 | 1.35% | 322 | 2.58% |

---

## Table 9 — Test-Set AUPRC

| Model | 1h | 3h | 6h |
|---|---:|---:|---:|
| Logistic | 0.0125 | 0.0483 | 0.0880 |
| **Physiology-only** | **0.0918** | **0.1012** | **0.1266** |
| Laboratory-only | 0.0060 | 0.0278 | 0.0527 |
| Text-only | 0.0036 | 0.0136 | 0.0259 |
| Naive concatenation | 0.0480 | 0.0770 | 0.1034 |
| Multimodal | 0.0215 | 0.0833 | 0.1175 |

---

## Table 10 — Six-Hour Multimodal Ablation

| Available modalities | AUPRC |
|---|---:|
| Physiology + text | 0.1256 |
| Physiology only | 0.1243 |
| Physiology + laboratory | 0.1180 |
| Physiology + laboratory + text (full model) | 0.1175 |
| Laboratory only | 0.0470 |
| Laboratory + text | 0.0456 |
| Text only | 0.0291 |

The results indicate that physiology is the dominant predictive modality under the current representation and cohort.

---

## Table 11 — Six-Hour Bootstrap Confidence Intervals

Bootstrap unit: ICU stay. Number of replicates: $B = 2000$.

| Model | AUPRC | 95% percentile CI |
|---|---:|---:|
| Logistic | 0.0880 | [0.0453, 0.1509] |
| Physiology-only | 0.1266 | [0.0557, 0.2473] |
| Naive concatenation | 0.1034 | [0.0553, 0.1802] |
| Multimodal | 0.1175 | [0.0592, 0.1954] |

---

## Table 12 — Paired Six-Hour AUPRC Differences

| Comparison | Difference | 95% CI |
|---|---:|---:|
| Multimodal − Physiology-only | −0.0091 | [−0.0980, +0.0575] |
| Multimodal − Naive concatenation | +0.0141 | [−0.0496, +0.0885] |
| Multimodal − Logistic | +0.0294 | [−0.0145, +0.0862] |

All three confidence intervals contain zero. The test cohort therefore does not establish a statistically distinguishable difference between the multimodal model and these comparison models.

---

## Table 13 — Alarm Operating Point

| Quantity | Value |
|---|---:|
| Threshold $\tau$ | 0.05 |
| Consecutive positive predictions $k$ | 2 |
| Cooldown | 60 min |
| Patient-days | 520.21 |
| Total alarms | 5,950 |
| Alarms / patient-day | 11.44 |
| False alarms / patient-day | 10.92 |
| Alarm precision | 4.49% |
| Events | 73 |
| Events detected | 62 |
| Event sensitivity | 84.93% |
| Median warning time | 5.18 h |
| Q1 warning time | 2.89 h |
| Q3 warning time | 5.63 h |

---

## Table 14 — Missing-Modality Robustness

| Perturbation | 6h AUPRC |
|---|---:|
| Note delay 180 min | 0.1272 |
| Laboratory removed | 0.1256 |
| Laboratory + text removed | 0.1243 |
| Note delay 60 min | 0.1229 |
| 60% laboratory dropout | 0.1203 |
| 40% laboratory dropout | 0.1192 |
| Text removed | 0.1180 |
| Full model | 0.1175 |
| 20% laboratory dropout | 0.1160 |
| Physiology + text removed | 0.0470 |
| Physiology removed | 0.0456 |
| Physiology + laboratory removed | 0.0291 |

---

## Table 15 — Physiological Noise Stress Test

| Noise scale $\sigma$ | AUROC | AUPRC |
|---:|---:|---:|
| 0.00 | 0.8027 | 0.1175 |
| 0.10 | 0.8018 | 0.1170 |
| 0.25 | 0.7990 | 0.1138 |
| 0.50 | 0.7903 | 0.1046 |
| 1.00 | 0.7631 | 0.0849 |
| 2.00 | 0.7101 | 0.0627 |

---

## Table 16 — Sensor Blackout Stress Test

| Blackout duration | AUROC | AUPRC |
|---:|---:|---:|
| 0.5 h | 0.7972 | 0.1088 |
| 1 h | 0.7972 | 0.1088 |
| 2 h | 0.7859 | 0.1047 |
| 4 h | 0.7642 | 0.0992 |

---

## Table 17 — Explanation Analysis

| Quantity | Result |
|---|---:|
| Alarm samples explained | 50 |
| Physiology-dominant samples | 32 |
| Laboratory-dominant samples | 18 |
| Median physiological contribution | +0.0308 |
| Median laboratory contribution | −0.0339 |
| Median text contribution | 0.0000 |

These values represent model-attribution quantities (see `report/whitepaper.md`, §13) and should not be interpreted as causal effects.