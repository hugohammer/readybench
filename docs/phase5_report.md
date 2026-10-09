# Phase 5 report: Study B (controlled degradation)

*9 October 2026. Protocol: plan section 11 (git tag `prereg-v2`). Analysis: `scripts/phase5_analysis.py` on all outputs in `results/raw/studyB/` plus the local TabPFN run (`results/raw/studyB_tabpfn_nofp.csv`). Tables: `results/phase5/`.*

## Runs

- **Scale:** 24 base datasets × 5 replications × (1 clean + 11 types × 5 levels) = **6,720 units, all completed**.
- **Where:**
  - 5,712 units ran on the OsloMet runners;
  - 1,008 units ran locally (2 workers in the 15.7 GB WSL budget, no out-of-memory events) after the CI quota was exhausted.
- **Measure gaps:** the thesis baselines (empcv) are missing for 5 units (open_payments, `fewer_rows` levels 4–5: the thesis encoding drops every column). There are no other measure gaps.
- **TabPFN probe** (replications 0–2): 4,032 of 4,032 units, without the fingerprint step (deviation logged).

## Harm of each degradation (median ΔAUC at level 5, clean test data)

| Degradation | Median ΔAUC | | Degradation | Median ΔAUC |
|---|---|---|---|---|
| label noise (40%) | −0.119 | | MAR (50% of cells) | −0.018 |
| fewer rows (×1/32) | −0.094 | | MCAR (50% of cells) | −0.017 |
| minority removal (imbalance ×32) | −0.081 | | irrelevant features (+400%) | −0.008 |
| minority label noise (80%) | −0.039 | | outliers (20% of cells) | −0.005 |
| MNAR (50%) | −0.028 | | duplicates (+200%) | −0.001 |
| redundant features (+400%) | −0.023 | | | |

**Modern boosting models are robust to missingness, duplicates, outliers and extra features.** They are hurt by wrong labels, too little data and rare positives.

## Pre-registered hypotheses (B5)

| | Result |
|---|---|
| **H4a** sensitivity | 19 of 21 targeted pairs react (median level–Δm ρ ≤ −0.5; most −1.0). The two redundancy measures do not: the added features correlate ≈ 0.95 with a *combination* of two features, below r > 0.8 with any single one. → **supported** (90% ≥ 80%) |
| **H4b** non-proportionality | Only 4 of 21 pairs have median ρ(Δm, ΔAUC) < 0.5: duplicates 0.14, outliers 0.46 and the two redundancy measures. The other 17 range 0.70–0.92. → **not supported** |
| **H4c** false alarms on redundant/irrelevant | harmless types: duplicates, irrelevant, outliers; no classical measure falls by > 0.5 SD on `irrelevant` (and `redundant` is not harmless) → **not supported** |

**Limitation of H4b.** Within one degradation type, severity drives both the measure and the performance in the same direction, so a high within-type correlation is close to automatic. H4b therefore says little about whether a measure's *size* of reaction reflects harm. The analyses below address that.

## Exploratory: calibration across degradation types

- **Tracking:** within each dataset, all types and levels are pooled, and we ask whether a larger change in the measure means a larger performance drop.

  | Measure set | Median ρ |
  |---|---|
  | Classical measures (set Q) | **0.05** |
  | Target-aware probes (set T) | **0.36** |
  | TabPFN | 0.62 |
  | 1-NN landmarker | 0.56 |
  | Thesis manual score (best classical) | 0.28 |

- **Classical measures react most to the least harmful problems** (median standardized change at level 5; Study A SD units):

  | Measure / degradation | Measure change | ΔAUC |
  |---|---|---|
  | AIDRIN duplicity / duplicates | −3.7 | −0.001 |
  | AIDRIN outliers / outliers | −4.3 | −0.005 |
  | AIDRIN completeness / MCAR | −3.4 | −0.017 |
  | AIDRIN imbalance degree / minority removal | −1.0 | −0.081 |
  | Thesis C3 sample size / fewer rows | −1.6 | −0.094 |

  These are false alarms in spirit, outside H4c's pre-registered definition.
- **Blind spot: label noise.** It is the most harmful degradation, and no classical measure detects it. The thesis manual score even *rises* (+0.67 SD), because flipped labels make the classes more balanced. TabPFN falls most under label noise (−3.3 SD), in line with the actual harm.
- **Caution for CV-based probes:** duplicated rows inflate them (1-NN +1.6 SD, TabPFN +0.3 SD), because copies land in both training and validation folds. A duplicate check remains useful, as a guard for the probes rather than as a predictor of model quality.

## Summary for the paper

- **Classical quality measures do what they are designed for:** they detect their targeted problem, and within that problem they scale with severity.
- **They are badly calibrated to harm:** their alarms are loudest for problems that modern models shrug off (missingness, duplicates, outliers), and absent for the most damaging one (label noise).
- **Cheap target-aware probes track harm much better across problem types,** but need a duplicate check to avoid optimism.
