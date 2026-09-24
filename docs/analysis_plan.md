# ReadyBench analysis plan (pre-registration draft)

*Status: draft 0.1, 24 September 2026. To be frozen, with a commit hash and date, before any readiness measure is compared with the gold standard.*
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

## 3. Datasets (Study A)

- **Source:** TabRepo release 2023_11_14 (Salinas & Erickson, AutoML 2024). It covers 117 binary OpenML tasks with stored test results for 1,530 model configurations (11 model families) and for AutoML systems.
- **Filters:** 500 ≤ rows ≤ 1,000,000 and ≤ 2,000 features, leaving 105 datasets.
- **Synthetic datasets:** 23 datasets look synthetic or artificial by a name-based heuristic (GAMETES, fri_*, twonorm, madelon, …). The list will be checked by hand in Phase 1. They are excluded from the primary analysis, which leaves about 82 real-world datasets. They are kept for a sensitivity analysis.
- **Known weakness:** only 15 of the 105 have missing values. Classical quality dimensions vary little in curated benchmarks, so Study A has limited power to test them. Study B covers this. We will report the spread of every quality dimension across the pool.
- **D1 – extend the pool (decided 2026-09-24: yes).** Add about 30–50 real-world binary OpenML datasets with missing values and/or strong imbalance that are not in TabRepo. Their gold standard would come from our own run (section 4b). This makes the pool more representative, at the cost of compute and extra time. 
- **De-duplication:** check OpenML dataset IDs and names for the same data listed twice. If found, keep one copy.

## 4. Gold standard

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

Any change after the plan is frozen will be logged here with its date and reason.
