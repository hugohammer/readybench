# Amendment 1 report: caveats of Study A

*8 October 2026. Plan section 10 (git tag `prereg-v1.1`, registered after Phase 4 and before these outcomes were computed). Script: `scripts/phase4b_amendment.py`. Tables: `results/phase4b/`. Primary pool: 80 datasets.*

## Data checks

- **TabRepo stored predictions:** for the selected model per fold, AUC computed from TabRepo's stored test predictions equals TabRepo's own value to 3e-16 on 281 of 282 folds. The exception is mammography fold 1, whose stored predictions are all zeros; it is excluded, so that dataset uses 2 folds.
- **Datasets without stored predictions:** seven TabRepo datasets are not in TabRepo's public prediction release and were run through the ReadyBench recipe instead (deviation from A1, logged).
- **Reproducibility:** rerunning the ReadyBench recipe on the 30 extension datasets reproduced every Phase 2 gold AUC exactly (max difference 0.0000).

## Predictions

| | Prediction | Result |
|---|---|---|
| **P1** | with AP as gold, imbalance measures ρ ≥ 0.3 | `aidrin_imbalance_degree` 0.47 [0.28, 0.63]; `thesis_c2_balance` 0.46 [0.26, 0.63] → **supported** |
| **P2** | H1's rule holds with nAP and IG | IG: holds (thesis score 0.14, ICIRCA 0.10). nAP: fails narrowly (thesis score 0.31 with CI upper 0.52; ICIRCA 0.29 with CI upper 0.48). → **not supported** |
| **P3** | H2's procedure gives CI > 0 for AP, nAP, IG | AP 0.62 [0.36, 0.87]; nAP 0.64 [0.39, 0.90]; IG 1.01 [0.66, 1.31] → **supported** |
| **P4** | thesis baselines on independent folds 3–5 ρ ≥ 0.85 | 0.86 [0.73, 0.94] (shared folds: 0.88) → **supported** |

## Interpretation

- **Imbalance limits precision, and classical scores partly capture this.** When performance is measured by how well positives are found (AP), imbalance itself limits what is achievable. Classical scores that contain an imbalance component (thesis C2, AIDRIN imbalance degree, and through them the composites) then carry real information (ρ ≈ 0.4–0.47).
- **Without the prevalence floor, that information mostly disappears.** It weakens for lift over prevalence (nAP, ρ ≈ 0.3) and vanishes for information gain (ρ ≈ 0.1). So the Study A conclusion "classical scores carry no signal" is specific to ranking metrics (AUC) and calibration-type metrics (IG). For precision-type metrics it should be stated more carefully.
- **The target-aware probes remain clearly better under every metric.** TabPFN: AP 0.77, nAP 0.86, IG 0.93. The thesis baselines: 0.79–0.89.
- **H3 survives on independent test folds** (0.86). The shared-fold version is only slightly inflated (0.88).

## Incremental value (A3, exploratory)

- **Partial Spearman** of the classical composites with the gold AUC, given the probe: TabPFN −0.22 to 0.12; logistic-regression landmarker 0.02 to 0.18.
- **Leave-one-out prediction of the gold AUC:**
  - TabPFN alone: MAE 0.023; TabPFN + 4 classical measures: 0.025.
  - Logistic-regression landmarker alone: 0.049; with classical measures: 0.052.
- **Conclusion:** classical measures add nothing beyond a cheap probe for AUC.

## Common-hardware timing (A4, exploratory)

- **Setup:** one laptop core (taskset), 19 of 20 planned datasets. The largest dataset was not run: the in-memory measures needed > 14 GB on the 15 GB laptop.
- **Median seconds per dataset:**

  | Measure group | Median s |
  |---|---|
  | ICIRCA | 0.0 |
  | Thesis C1–C4 | 0.0 |
  | Relevance | 0.5 |
  | Landmarkers | 0.7 |
  | AIDRIN | 2.3 |
  | TabPFN (1 CPU core) | 3.1 |
  | Thesis baselines | 3.2 |
  | PyMFE complexity | 9.6 |
  | TabPFN (GPU, incl. ~10 s CUDA start-up) | 17.8 |
  | DREAMER (extrapolated to 1,000 subtables; 3 of 19 hit the 20-min limit for 100 subtables) | ~2,600 |

- **Conclusion:** every measure except DREAMER is affordable in a pre-sales setting.
