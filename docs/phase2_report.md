# Phase 2 report: gold standard

*26 September 2026. Runs: pipelines 9199 (≤ 200k rows), 9200 (> 200k rows), 9217 (leakage screen), 9218 (reruns after leakage fixes); 0 failed units. Result table: `results/gold_v1.csv`.*

## Recipe (analysis plan 4b)

- **Models:** tuned LightGBM and CatBoost, each a cross-validation bag on the OpenML training fold (5 inner folds, or an 80/20 holdout above 50k rows). Default settings + 16 random settings per family; default + 4 for the three datasets > 200k rows.
- **Selection:** by inner out-of-fold AUC only. The candidates are the best of each family and their average. The average was chosen in 66 of 132 folds, CatBoost in 42 and LightGBM in 24.
- **Evaluation:** OpenML folds 0–2, the same folds as TabRepo.

## Calibration on 12 TabRepo datasets

| Comparison | Spearman | Mean difference (ours − TabRepo) |
|---|---|---|
| TabRepo AutoGluon (primary gold) | 0.986 | −0.005 |
| TabRepo tuned boosting selected by validation | 1.000 | −0.005 |

- **The recipe reproduces TabRepo's own boosting models.** The small gap to AutoGluon is what one expects from its larger multi-model ensemble.
- **Largest single difference:** pc2 (−0.055). It has about 23 positives in total, so each test fold holds 2–3 positives and the fold-level SD is 0.19. The gap reflects noise in the dataset, not the recipe.
- **Tuning budget:** default + 4 settings versus default + 16 changes the gold AUC by +0.0002 on average (max +0.010; Spearman 0.997). This supports the reduced budget for the three largest datasets.

## Leakage screen (new step, `scripts/phase2_leakage_screen.py`)

Several extension datasets had gold AUC 1.000, which prompted the screen.

- **Rule:** flag a dataset if a single column, or its missingness indicator, reaches AUC ≥ 0.99 on its own. Categorical columns use out-of-fold target encoding, so unique IDs cannot score trivially.
- **Result:** 5 of 134 datasets flagged. No primary TabRepo dataset is among them.

| Dataset | Finding | Action |
|---|---|---|
| US_Stocks_Financial_Indicators | `2019_price_var` defines the class (AUC 1.000) | drop column → gold 0.772 |
| MembershipWoes | `END_DATE` present exactly for cancelled members (1.000) | drop column → gold 0.810 |
| irish | `Educational_level` determines the target (1.000) | drop column → gold 0.883 |
| Support2 (review, not flagged) | post-outcome variables for target *death* | drop `hospdead`, `d.time`, `sfdm2` → gold 0.850 |
| Skin_Cancer_PAD-UFES-20 (review) | metadata missingness predicts target (0.93); `diagnostic` determines it for 3 of 6 classes | excluded |
| houses (binarized) | `latitude` predicts binarized target (0.995) | excluded |
| colleges_aaup (binarized) | one count column predicts binarized target (0.992) | excluded |

- **Where the fixes live:** `configs/leakage_fixes.csv`, applied whenever a dataset is loaded, so every later phase sees the repaired data.
- **What stays in the pool:** eeg-eye-state and Run_or_walk (AUC ≥ 0.999) are time-ordered sensor data. Random CV folds flatter them; they are already flagged, and a sensitivity analysis excludes them. mushroom (0.9999) is genuinely easy (best single column 0.987).

## Final pool with gold standard

- **Primary:** 80 datasets, 55 TabRepo + 25 extension.
- **Sensitivity pools:** binarized 22, synthetic 29.
- **Primary datasets below the feasibility threshold:** 10 below 0.75, **21 below 0.80 (D2 primary)**, 27 below 0.85.
- **Mixed gold sources:** the primary gold (TabRepo AutoGluon or our recipe) and the uniform secondary gold (tuned boosting by validation for all) agree with Spearman 0.996 (mean difference +0.001), so mixing sources does not affect rankings.
