# ReadyBench analysis plan (pre-registration draft)

*Status: FROZEN 1 October 2026 as git tag `prereg-v1` (version 1.0), before any readiness measure was compared with the gold standard. Later changes are logged in section 9.*
*Authors: Jonas Lysfjord Kemi, Hugo Lewi Hammer (+ Fraktal contact, to be decided).*

Items marked **[DECISION]** need an answer before the plan is frozen.

## 1. Research questions

- **RQ1.** How well do published data-readiness and data-quality measures agree with the best achievable ML performance on a dataset?
- **RQ2.** Are cheap target-aware signals (feature relevance, simple landmarking models, foundation-model probes) better indicators of readiness than classical quality dimensions? At what compute cost?
- **RQ3.** Is a cheap empirical score (the mean of 5 untuned baseline models, as in the thesis) a good stand-in for the gold standard?
- **RQ4 (Study B).** When data quality is deliberately degraded, does each measure react in the same direction as the drop in ML performance, and in proportion to it? Does it raise false alarms for degradations that do not hurt performance?

## 2. Hypotheses (confirmatory)

- **H1.** Measures built only on classical quality dimensions have weak rank agreement with gold-standard AUC (|Spearman ρ| < 0.3). These are completeness, duplicates, outliers, class imbalance, sample size and feature redundancy, as used by the AIDRIN data-quality metrics, the thesis C1–C4 score and the ICIRCA 2026 DQI.
- **H2.** The best target-aware measure (R4–R6 in section 5) agrees better with the gold standard than the best classical quality measure, by a paired-bootstrap ρ difference with a 95% confidence interval that excludes 0.
- **H3.** The cheap empirical score has ρ ≥ 0.85 with the gold standard.
- **H4.** In Study B, each quality measure reacts to the degradation it targets. Its reaction is not proportional to the AUC drop (median within-dataset Spearman ρ between measure change and AUC change < 0.5 for at least half of the degradation types). At least one measure raises false alarms on redundant or irrelevant features.

Everything else is exploratory and will be reported as such.

### 2a. Operational definitions (added 2026-10-01, before freezing)

- **Oriented ρ:** Spearman ρ between a measure and the gold AUC, multiplied by the measure's pre-fixed direction (`measures.DIRECTION`), so positive means "agrees". Computed on the 80 primary datasets. A dataset is dropped for a measure only where that measure is undefined; the n is reported.
- **Classical quality set Q (H1, H2):** 16 measures.
  - AIDRIN: `aidrin_completeness`, `aidrin_duplicity`, `aidrin_outliers`, `aidrin_constant_frac`, `aidrin_max_pairwise_corr`, `aidrin_max_abs_skew`, `aidrin_max_abs_kurtosis`, `aidrin_row_completeness`, `aidrin_feature_coverage`, `aidrin_imbalance_degree`.
  - ICIRCA: `icirca_dqi` (as published).
  - Thesis: `thesis_manual_score` and `thesis_c1`–`c4`.
  - The two composite scores are `thesis_manual_score` and `icirca_dqi`.
- **Target-aware set T (H2):** 9 measures: `rel_max_abs_r`, `rel_max_single_auc`, `rel_mean_top5_auc`, `rel_max_norm_mi`, `lm_stump_auc`, `lm_nn1_auc`, `lm_naive_bayes_auc`, `lm_logreg_auc`, `tabpfn_auc`.
- **Measures in neither set** (reported, exploratory):
  - the target-aware AIDRIN relevance metrics;
  - DREAMER Total Quality and its components (a mixture of quality and target-aware parts);
  - the 22 PyMFE complexity measures;
  - `icirca_dqi_corrected`.
- **H1 is supported** if, for both composite scores, the oriented ρ has a point estimate < 0.3 and a 95% CI upper bound < 0.5, *and* at least 80% of the measures in Q have |ρ| < 0.3.
- **H2 procedure:**
  - The 80 primary datasets are split at random (seed 20260924) into a selection part (24) and a test part (56).
  - The measure with the highest oriented ρ in T, and in Q, is chosen on the selection part.
  - On the test part, the paired bootstrap (10,000 resamples) gives a CI for ρ_T − ρ_Q.
  - **H2 is supported** if the 95% CI lies above 0.
- **H3 is supported** if the oriented ρ of `emp_mean_auc` (the thesis baselines, AUC version) is ≥ 0.85. Its CI is reported.
- **Multiplicity:** H2 is the only significance test among H1–H3, so the Holm correction listed in draft 0.1 is not needed for H1–H3. It applies to the subgroup tests if any are reported with p-values.
- **Feasibility detection:** for each measure, the AUROC for separating "not feasible" (gold AUC < 0.80) from feasible datasets, with a bootstrap CI.
- **Sensitivity analyses:**

  | Code | Change |
  |---|---|
  | S1 | without the time-ordered datasets |
  | S2 | without the anonymized datasets |
  | S3 | without datasets flagged check / provenance / odd-target / near-ceiling / simulated-physics |
  | S4 | + binarized datasets |
  | S5 | + synthetic datasets |
  | S6 | secondary (uniform) gold standard |
  | S7 | feasibility thresholds 0.75 and 0.85 |
  | S8 | TabRepo datasets only (no own gold) |
  | S9 | the thesis's original empirical score (macro-F1 × 100) instead of AUC |

## 3. Datasets (Study A)

The pool is frozen in `configs/pool_v1.csv` (Phase 1, 25 September 2026). The report is in `docs/phase1_report.md`.

- **TabRepo** (release 2023_11_14): binary tasks with 500 ≤ rows ≤ 1,000,000 and ≤ 2,000 features, 105 datasets. They split into:
  - 55 **primary** (real data, natural binary target);
  - 18 **binarized** (a numeric or multi-class target split in two by OpenML);
  - 29 **synthetic** (generated by a formula or simulator);
  - 2 **image** (pixels, excluded);
  - 1 **duplicate** (cpu_small, a subset of cpu_act).
- **D1 extension (decided 2026-09-24: yes):** active binary OpenML datasets outside TabRepo with missing values in ≥ 5% of rows and a 10-fold CV task. Of 46 candidates:
  - 26 primary;
  - 6 binarized;
  - 11 duplicates of data already in the pool (e.g. credit-approval = Australian; German credit = credit-g; KDD Cup 2009 churn/upselling share the appetency features);
  - 3 text (excluded).
- **Primary analysis:** 80 datasets (after the Phase 2 leakage screen). **Sensitivity analyses:** + binarized (22) and + synthetic (29).
- **Leakage screen (added in Phase 2):** flag a dataset if a single column or its missingness indicator reaches AUC ≥ 0.99 alone. Repairs by dropping named columns are in `configs/leakage_fixes.csv`; unrepairable datasets are excluded as `leakage`. See `docs/phase2_report.md`.
- **Rules and review:** every category decision is recorded with a reason in `configs/pool_overrides.csv`. Datasets that stay in the pool but are borderline carry a flag (time-ordered, anonymized, odd target, provenance, near-ceiling, check). The anonymized and time-ordered flags define extra sensitivity analyses.
- **Duplicate check:** column fingerprints (Jaccard of per-column signatures ≥ 0.5) across all pairs. Only cpu_small/cpu_act were found beyond the manual review.
- **Review decisions (D3, 2026-10-01; made by the analysis team under delegation from H. L. Hammer).**
  - All 18 flagged datasets stay in the primary pool. Their flags define sensitivity analyses S1–S3 instead of exclusions.
  - `churn` (did 40701) stays flagged `check`: whether it is artificial cannot be settled from OpenML, so it falls under S3.
  - The leakage decisions in `configs/leakage_fixes.csv` and the `leakage` exclusions are confirmed.
  - No pool decision is revisited after the plan is frozen, except for errors, which would be logged in section 9.
- **Effect of the extension:** datasets with missing values rise from 18% to 44% of the primary pool, and the 90th percentile of missing cells from 4% to 18%.
- **Remaining limitation:** missingness coded as a value (e.g. -1 in porto-seguro) is not counted as missing. Completeness is therefore understated for some datasets. The readiness tools under study share this blind spot.
- **Feasibility cases (D2, AUC < 0.80):** 12 of the 55 primary TabRepo datasets; the extension adds more once its gold standard is computed.

## 4. Gold standard

- **Status:** computed in Phase 2 → `results/gold_v1.csv`.
- **(a) Primary:** the test ROC-AUC of AutoGluon (best quality, 4 h, 8 cores; `AutoGluon_bq_4h8c_2023_11_14`), averaged over TabRepo's 3 outer folds (OpenML task folds 0–2).
- **Secondary gold standard:** per fold, the best TabRepo configuration chosen by *validation* error (so the choice does not peek at the test data), with its test AUC averaged over folds.
- **Robustness check:** tuned boosted trees only (LightGBM, CatBoost, XGBoost), selected by validation error.
- **(b) Datasets outside TabRepo :** a fixed recipe on the same OpenML folds 0–2:
  - LightGBM and CatBoost with random search of N configurations and inner 5-fold CV, plus TabPFN on datasets with ≤ 10,000 rows;
  - the best by validation error is reported.
  - The recipe is calibrated on 10–15 TabRepo datasets. We report its agreement with the primary gold standard (ρ and mean absolute difference).
- **Secondary outcomes:** log-loss is not in the TabRepo tables. Balanced accuracy needs predictions. Both are computed only where we run models ourselves.
- **Feasibility label:** gold AUC < threshold means "not feasible". In the 117 binary datasets, 6 have AUC < 0.70 and 22 have AUC < 0.80.
  - **D2 – threshold (decided 2026-09-24):** primary 0.80, with sensitivity analyses at 0.75 and 0.85. The thesis plan said 0.70, but that leaves too few infeasible cases for a stable estimate.
- **Ceiling effect:** median gold AUC is about 0.95, so the distribution is skewed towards easy datasets. Rank-based statistics are used throughout for this reason.

## 5. Readiness measures

All measures are computed on the raw OpenML data with one documented minimal preprocessing step:
- target coded 0/1;
- identifier columns removed;
- no imputation, except where a measure requires complete data (documented per measure).

Target-aware measures that train models use only the training part of fold 0 when they are used for prediction. For rank-agreement analysis they use the full dataset, as a consultant would.

| Group | Measures | Implementation |
|---|---|---|
| R1 published tools | AIDRIN data-quality and "impact on AI" metrics (completeness, duplicity, outliers, class imbalance, correlations, feature relevance, …); DREAMER; the ICIRCA 2026 DQI | `aidrin==2026.9.1` headless API, telemetry off (the default); DREAMER from GitHub; DQI reimplemented from its repo |
| R2 thesis | Manual score C1–C4 and each sub-score | Code ported from the thesis repo |
| R3 complexity measures | Feature overlap (F1–F4), linearity (L1–L3), neighbourhood (N1–N4), imbalance (C1–C2) | PyMFE `complexity` group, subsample of at most 5,000 rows |
| R4 relevance | max \|point-biserial r\|, max mutual information, mean of the top 5 | Own code |
| R5 landmarkers | Decision stump, 1-NN, naive Bayes, logistic regression, each with 5-fold CV AUC on a subsample of at most 10,000 rows | Own code / PyMFE `landmarking` |
| R6 foundation-model probe | TabPFN cross-validated AUC on a subsample of at most 10,000 rows | `tabpfn` (added in Phase 3) |
| Cheap empirical score | The thesis's 5 untuned baselines, AUC on the same folds | Code ported from the thesis repo |

- **Implementation (Phase 3):** `src/readybench/measures.py` and `scripts/phase3_measures.py`, plus `scripts/phase3_dreamer.py`. Measure directions are fixed in `measures.DIRECTION` and exported to `outputs/phase3/directions.csv` before any comparison with the gold standard. All measures see the data as loaded, with leakage fixes applied.
  - **Row caps:** stratified subsamples, with the same seed for every measure. 50,000 rows by default; 5,000 for PyMFE complexity and DREAMER (both need quadratic memory or time); 10,000 for landmarkers.
  - **Cheap empirical score:** trains on at most 100,000 rows of each training fold (k-NN: 20,000, as in the thesis) and is scored on the full test fold of OpenML folds 0–2, the same folds as the gold standard. Its preprocessing is fitted on the training fold only.
  - **AIDRIN** (`aidrin==2026.9.1`, headless `run_metric`): run on a CSV with the features and the target, as a user would. Some AIDRIN metrics therefore include the target column; this is the tool's behaviour. AIDRIN has no overall score, so each metric enters separately. Undefined outputs are recorded as missing, e.g. the overall outlier score when a column is empty.
  - **DREAMER** (vendored at commit 96aabaf, unchanged): run in its own environment with its pinned dependencies (Python 3.10, pandas 1.5, numpy 1.23, scikit-learn 1.2). Settings: 1,000 random subtables (the lower end of the authors' range), their example free parameters (row/column exclusion 0.7/0.5, equal clustering/classification weights), categoricals as ordinal codes (DREAMER requires numeric input), and a 45-minute limit per dataset (very wide datasets are expected to exceed it; coverage is reported). The readiness measure is the master table's *Total Quality*; its components are also recorded.
  - **ICIRCA DQI:** reimplemented from the tool's source (commit 28bad02; the repository has no licence, so no code is copied). `icirca_dqi` reproduces the tool exactly. This includes its treatment of missing or duplicate percentages ≤ 1 as ratios, e.g. 0.8% missing is scored as 80%. `icirca_dqi_corrected` (exploratory) uses true ratios.
  - **Thesis score:** ported from the thesis repository. It reproduces the thesis values, e.g. diabetes 83.64 vs 83.62 and kc2 68.04 vs 68.05; the small differences come from rounding in the thesis tables.
  - **R6 TabPFN probe** (added 2026-10-01): TabPFN v2 (Hollmann et al., *Nature* 2025; `tabpfn==2.0.9`, ungated v2 weights; newer releases require a licence registration), 5-fold CV AUC on ≤ 10,000 rows, no tuning, categoricals flagged as such. Datasets with > 500 features use a seeded random subset of 500 (the v2 pre-training limit). It runs on a local GPU (RTX 4060, 8 GB; `scripts/phase3_tabpfn.py`, environment in `ci/requirements-gpu.txt`).
  - **DREAMER subtables:** DREAMER's `subtables` setting is per run, and the number of runs equals its `cores` setting; its README describes it as the total. We fix 1,000 subtables in total (40 runs × 25), on ≤ 5,000 rows. A first attempt with 56 × 1,000 = 56,000 subtables per dataset took about 1 h each and was cancelled (pipeline 9283). Its 35 finished scores are kept for a robustness check against the final run.
- **Recorded per measure:** wall-clock time on the reference machine, whether it needs labels, and whether it trains a model.
- **Direction:** each measure's sign is fixed *before* analysis from its definition (higher means more ready). No sign flipping after looking at the results.
- **Measures with several numbers:** evaluated (a) one metric at a time, and the tool's own aggregate where one exists; and (b) as an exploratory upper bound, a learned combination evaluated with leave-one-group-out CV across datasets (groups are dataset families, e.g. all GAMETES datasets).

## 6. Analyses (Study A)

1. **Primary:** Spearman ρ between each measure and the gold AUC, with a 95% percentile bootstrap CI (10,000 resamples of datasets).
2. **H2:** paired bootstrap of the ρ difference between the best R4–R6 measure and the best R1–R2 measure. "Best" is defined on a random 30% of datasets and tested on the other 70%.
3. **Feasibility detection:** AUROC and precision at 80% recall for flagging "not feasible" datasets.
4. **Cost against agreement:** plot of ρ against the median runtime per measure.
5. **Multiple testing:** Holm correction across the confirmatory tests H1–H3.
6. **Subgroups (exploratory):** by dataset size tercile, missing values present vs absent, imbalance ratio above vs below 3, and domain.
7. **Sensitivity:** include the synthetic datasets; use the secondary gold standard; use the other thresholds.

## 7. Study B (outline, detailed in draft 0.2)

- **Base datasets:** 20–30 real-world datasets with gold AUC ≥ 0.85, spread over sizes and domains.
- **Degradations (training data only, test data clean):**
  - missing values (completely at random, dependent on observed features, dependent on the missing value);
  - label noise (uniform and class-dependent);
  - minority-class removal;
  - fewer rows;
  - duplicate rows;
  - redundant (collinear) features;
  - irrelevant noise features;
  - outliers.
- **Design:** 6 severity levels and 5 seeds, following Mohammed et al. (2025) and Fault Lines (VLDB 2025).
- **Performance:** measured with the fixed recipe from section 4b.
- **Outputs:** sensitivity, proportionality and false-alarm rate per measure and degradation type.

## 8. Reproducibility

- Pinned environment (`requirements.lock`).
- The OpenML cache and the TabRepo tables are documented with SHA-256 hashes (`scripts/fetch_tabrepo_tables.py`).
- A fixed seed (20260924) set in the config files.
- All results written as tables under `outputs/` with the commit hash.

## 9. Deviations

Any change after the plan is frozen will be logged here with its date and reason. Changes to the draft before freezing are also listed, for transparency.

- **2026-10-01, after freezing, implementation detail:** directions for the DREAMER outputs were added to `measures.DIRECTION` (all +1, as DREAMER defines every master-table field so that higher is better). This applies the plan's rule (direction from the definition) and was done before any comparison with the gold standard.

- **2026-09-25, section 4b, no TabPFN in the recipe.** The GPU runner is down, and a CPU TabPFN run across the pool is not practical. The recipe is tuned LightGBM + CatBoost as CV bags (default + 16 random configs each), selected by inner out-of-fold AUC. Candidates are the best of each family and their average.
- **2026-09-26, section 3, leakage screen added** after gold AUC = 1.000 on several extension datasets. Four datasets repaired by dropping leaking columns and three excluded. Details in `docs/phase2_report.md`.
- **2026-09-26, section 4, secondary gold made uniform:** tuned boosting selected by validation for every dataset (TabRepo configs or our recipe), instead of TabRepo configs only.
- **2026-09-25, section 4b, smaller budget for the 3 datasets > 200,000 rows** (IEEE-CIS fraud, ATLAS Higgs, Homesite): default + 4 random configs per family, with a single 80/20 inner holdout. The full budget would dominate the phase's compute for little expected change in AUC. The effect is reported for the calibration datasets by comparing the 4-config and 16-config results.
