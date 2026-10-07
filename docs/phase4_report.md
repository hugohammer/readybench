# Phase 4 report: Study A results

*7 October 2026. Analysis: `scripts/phase4_analysis.py` (committed 2d60dee, before results) on `results/measures_v1.csv` and `results/gold_v1.csv`, following the plan frozen as git tag `prereg-v1`. Tables: `results/phase4/`. Primary pool: 80 datasets, 21 of them not feasible (gold AUC < 0.80).*

## Confirmatory hypotheses (all three supported)

| Hypothesis | Decision rule (plan 2a) | Result |
|---|---|---|
| **H1** classical quality measures agree weakly | composites ρ < 0.3 with CI upper < 0.5; ≥ 80% of Q with \|ρ\| < 0.3 | thesis manual score ρ = 0.06 [−0.18, 0.30]; ICIRCA DQI 0.02 [−0.20, 0.24]; 16/16 Q measures \|ρ\| < 0.3 → **supported** |
| **H2** best target-aware > best classical | paired bootstrap CI of ρ difference > 0 (selection 24 / test 56) | selected TabPFN vs AIDRIN outliers; difference 1.09 [0.75, 1.39] on 53 test datasets → **supported** |
| **H3** cheap empirical score ρ ≥ 0.85 | point estimate ≥ 0.85 | thesis 5 baselines (AUC) ρ = 0.88 [0.76, 0.95] → **supported** |

In H2, the classical measure that looked best on the selection part (AIDRIN outliers) had a negative ρ on the test part. That is consistent with the classical measures carrying no signal: the "best" one was chosen on noise.

## Agreement with the gold standard (oriented Spearman ρ, primary pool)

- **TabPFN v2 probe:** 0.93 [0.87, 0.97].
- **Thesis baselines** (shares the gold's test folds; see caveats): mean AUC 0.88, XGBoost 0.93, random forest 0.90.
- **Cheap landmarkers:** 1-NN 0.79, logistic regression 0.77, decision stump 0.64, naive Bayes 0.62.
- **Relevance signals, no model trained:** normalized max mutual information 0.76, max single-feature AUC 0.67.
- **Complexity measures:** neighbourhood measures N1–N3 about 0.62; linearity 0.43–0.49.
- **Published readiness tools:**
  - AIDRIN relevance metrics 0.42–0.47.
  - DREAMER Total Quality 0.20 [−0.05, 0.42] (n = 71); its class-overlap component (0.57) and built-in RF accuracy (0.54) carry the signal, which the weighted total dilutes.
  - ICIRCA DQI 0.02.
  - All AIDRIN quality metrics between −0.17 and 0.23.
- **Thesis C1–C4:** between −0.05 and 0.23.

## Feasibility detection (AUROC for gold AUC < 0.80; 21 not-feasible datasets)

| Measure | AUROC |
|---|---|
| TabPFN | 0.98 |
| Thesis baselines (AUC) | 0.97 |
| Logistic-regression landmarker | 0.93 |
| Max normalized MI | 0.92 |
| 1-NN | 0.91 |
| AIDRIN max relevance | 0.80 |
| DREAMER class overlap | 0.79 |
| AIDRIN completeness | 0.68 |
| DREAMER Total Quality | 0.58 |
| Thesis manual score | 0.55 |
| ICIRCA DQI | 0.53 |
| AIDRIN imbalance degree | 0.50 |

The ranking is the same at thresholds 0.75 and 0.85 (S7).

## Sensitivity analyses (S1–S9)

- **Conclusions are unchanged in every analysis.** TabPFN ρ = 0.93–0.95; thesis manual score 0.00–0.21; ICIRCA −0.04 to 0.19.
- **Adding synthetic datasets (S5) is the only notable change:** it weakens the simple signals (max single-feature AUC 0.67 → 0.51; logistic regression 0.77 → 0.62). Synthetic tasks such as GAMETES (epistasis) and parity hide signal in interactions, which single-feature and linear probes cannot see.
- **The thesis's original macro-F1 empirical score (S9):** ρ = 0.79.

## Cost (median seconds per dataset; hardware differs, see caveats)

| Measure group | Median s |
|---|---|
| ICIRCA | 0.0 |
| Thesis C1–C4 | 0.1 |
| Relevance | 1.0 |
| Landmarkers | 1.4 |
| AIDRIN | 5.4 |
| Thesis baselines | 6.8 |
| TabPFN (GPU) | 17.7 |
| PyMFE complexity | 18.7 |
| DREAMER (40 processes) | 129 |

- **Cheap and strong:** relevance and landmarkers run in about a second and reach ρ 0.6–0.8.
- **DREAMER coverage:** it could not process 11 of 131 datasets within 45 minutes (10 very wide, 1 very narrow).

## Caveats for the paper

1. **The gold metric is ROC-AUC,** which does not depend on class balance. Imbalance measures therefore cannot be expected to track it. Imbalance matters for threshold metrics (precision/recall at a cut-off), which TabRepo does not provide. This should be stated plainly. The macro-F1 version of the empirical score (S9) is a partial view.
2. **The thesis baselines are scored on the same test folds as the gold standard.** Part of their agreement (ρ 0.88) reflects shared test-set noise. TabPFN, the landmarkers and the relevance signals use their own internal CV, so their agreement is not inflated this way.
3. **Target-aware probes agreeing with model performance is expected** (landmarking). The main finding is the contrast: the published readiness and quality scores (AIDRIN quality metrics, ICIRCA DQI, DREAMER Total Quality, the thesis score) carry essentially no information about achievable performance on this benchmark, while probes costing about a second do.
4. **Runtimes come from different hardware:**
   - CPU groups: one core each on the EPYC runner;
   - DREAMER: 40 parallel processes;
   - TabPFN: an RTX 4060 laptop GPU.
   
   Compare orders of magnitude only.
5. **The benchmark is curated public data** (with the D1 extension for missingness). Classical quality problems vary less here than in raw client data. Study B (controlled degradation) addresses this directly.
