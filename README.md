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
