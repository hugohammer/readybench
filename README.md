# ReadyBench

Validating data-readiness measures against gold-standard ML performance on tabular binary classification.
This is a follow-up to J. L. Kemi's master thesis (OsloMet, 2026; https://github.com/jonaskemi/MasterThesis).

- Study A: readiness measures vs. gold-standard AUC on OpenML datasets (gold standard reused from TabRepo).
- Study B: controlled degradation of clean datasets.

The plan, hypotheses and open decisions are in `docs/analysis_plan.md`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/python scripts/fetch_tabrepo_tables.py
```

Caches go to `data_cache/` (not tracked by git). Set `READYBENCH_ROOT` to relocate them, e.g. on a server.

## Layout

```text
configs/   experiment configs (YAML)
docs/      analysis plan / pre-registration
scripts/   entry points, one per step
src/readybench/  library code
outputs/   generated tables and figures (not tracked)
```
