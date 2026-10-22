# ReadyBench

Validating data-readiness measures against gold-standard ML performance on tabular binary classification.
This is a follow-up to J. L. Kemi's master thesis (OsloMet, 2026; https://github.com/jonaskemi/MasterThesis).

- **Study A:** readiness measures vs. gold-standard performance on 131 OpenML datasets (80 primary).
- **Study B:** controlled degradation of 24 clean datasets (11 degradation types × 5 levels × 5 replications).

The pre-registered plan is in `docs/analysis_plan.md`:
- git tag `prereg-v1`: Study A;
- git tag `prereg-v1.1`: amendment 1;
- git tag `prereg-v2`: Study B protocol.

Phase reports are in `docs/phase*_report.md`.

## Paper

Kemi, J. L. and Hammer, H. L. *Evaluation of Readiness Measures to Predict Machine Learning Prediction Performance.* Manuscript in preparation for the Journal of Data-centric Machine Learning Research (DMLR), 2026.
How to cite this code: see `CITATION.cff` (GitHub's "Cite this repository" button) and the Zenodo DOI of the release.

## Reproducing the analyses from the committed results

All results that the paper reports are in `results/`. The analyses and figures can be rerun from these files alone, without downloading any datasets or retraining any models:

```bash
uv venv .venv --python 3.12 && uv pip sync requirements.lock   # or: python -m venv .venv && .venv/bin/pip install -r requirements.lock
export PYTHONPATH=src
.venv/bin/python scripts/phase4_analysis.py                       # Study A: H1-H3, feasibility, sensitivity -> outputs/phase4/
.venv/bin/python scripts/phase4b_amendment.py                     # amendment: AP/nAP/IG, folds 3-5, timing -> outputs/phase4b/
.venv/bin/python scripts/phase5_analysis.py results/raw/studyB/*  # Study B: H4 and tracking -> outputs/phase5_analysis/
.venv/bin/python scripts/make_figures.py                          # Figures 1-4 -> outputs/figures/
```

Rerunning the full experiments (gold standard, measures, Study B) downloads the datasets from OpenML and needs considerable compute; see the pipeline below.

## Pipeline

| Phase | Scripts | Main outputs |
|---|---|---|
| 1 Dataset pool | `phase1_candidates`, `phase1_build_pool`, `phase1_profile_pool`, `phase1_report` | `configs/pool_v1.csv`, `docs/phase1_report.md` |
| 2 Gold standard | `phase2_gold`, `phase2_leakage_screen`, `phase2_merge`; amendment: `phase2b_tabrepo_metrics`, `phase2d_merge_secondary` | `results/gold_v1.csv`, `results/gold_secondary_v1.csv` |
| 3 Readiness measures | `phase3_measures` (CPU groups), `phase3_dreamer` (own Python 3.10 env), `phase3_tabpfn` (local GPU), `phase3_merge` | `results/measures_v1.csv` |
| 4 Study A analysis | `phase4_analysis` (frozen rules), `phase4b_amendment`, `phase4c_timing` | `results/phase4/`, `docs/phase4_report.md` |
| 5 Study B | `phase5_studyB` (runner), `phase5_tabpfn` (local GPU), `phase5_analysis` | `outputs/phase5_analysis/` |

All scripts are in `scripts/`; shared code is in `src/readybench/` (`data`, `gold`, `measures`, `metrics`, `degrade`, `profile`).

## Environments

| Environment | Python | Defined in | Used for |
|---|---|---|---|
| main | 3.12 | `requirements.lock` | everything else |
| DREAMER | 3.10 | `ci/requirements-dreamer.txt` | DREAMER, with its original pins |
| GPU | 3.12 | `ci/requirements-gpu.txt` | TabPFN v2 (`tabpfn==2.0.9`) |

## Running on the OsloMet GitLab runners

Pipelines run only when started on purpose:

```bash
.venv/bin/python tools/gitlab.py run --task <script> --runner xl-cloud-runner --args "..."
.venv/bin/python tools/gitlab.py status [pipeline]
.venv/bin/python tools/gitlab.py fetch <pipeline>     # -> outputs/runs/<pipeline>/
.venv/bin/python tools/gitlab.py log <job> --tail 50
```

This needs `tools/gitlab.env` (gitignored) with an `api` token. Caches go to `data_cache/` (not tracked); set `READYBENCH_ROOT` to relocate them.

## Licence and third-party code

The code in this repository is released under the MIT licence (`LICENSE`).
`third_party/dreamer/` contains the DREAMER code (Ahangaran et al., 2024), vendored unchanged under its own MIT licence; see `third_party/dreamer/SOURCE.txt`.
The datasets are not included. They are downloaded from OpenML (https://www.openml.org) under their own licences, and the TabRepo tables from Hugging Face (`Tabrepo/tabrepo`).
