"""Phase 1, step 1: list candidate datasets for the Study A pool (metadata only, no downloads).

Two sources (analysis plan section 3):
  * TabRepo binary tasks passing the size filters (gold standard available);
  * the D1 extension: active binary OpenML datasets outside TabRepo with
    missing values in >= 5% of rows and a 10-fold CV task (own gold standard).

Writes outputs/phase1/candidates.csv. Inclusion decisions are made by hand in
configs/pool_decisions.csv, using the descriptions in this file.
"""

import re
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from readybench.paths import OUTPUTS, ROOT, TABREPO_CACHE, setup_openml_cache  # noqa: E402

# Names that are generated data, image/stream generators or scratch uploads.
# Applied to the extension only; TabRepo datasets are reviewed one by one.
EXCLUDE_NAME = re.compile(
    r"BNG|^fri_|GAMETES|autoUniv|_seed_|^SEA|Hyperplane|RandomRBF|LED|mnist|cifar|fashion|"
    r"Devnagari|^Agrawal|^Stagger|synthetic|^dataset_|^test|^tmp|Click_prediction|kdd_internet",
    re.IGNORECASE,
)
MIN_MISSING_ROW_FRACTION = 0.05


def load_config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "study_a.yaml").read_text())


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["missing_cell_frac"] = df.NumberOfMissingValues / (df.NumberOfInstances * df.NumberOfFeatures)
    df["missing_row_frac"] = df.NumberOfInstancesWithMissingValues / df.NumberOfInstances
    df["imbalance_ratio"] = df.MajorityClassSize / df.MinorityClassSize
    return df


def main() -> None:
    import openml

    setup_openml_cache()
    cfg = load_config()["pool_filters"]
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # openml's noisy XML warnings

    datasets = openml.datasets.list_datasets(output_format="dataframe", status="active")
    tasks = openml.tasks.list_tasks(task_type=openml.tasks.TaskType.SUPERVISED_CLASSIFICATION,
                                    output_format="dataframe")
    tabrepo = pd.read_csv(TABREPO_CACHE / "task_metadata.csv")
    configs = pd.read_parquet(TABREPO_CACHE / "configs.parquet", columns=["tid", "problem_type"])
    binary_tids = configs.loc[configs.problem_type == "binary", "tid"].unique()

    size_ok = lambda d: (d.NumberOfInstances.between(cfg["min_rows"], cfg["max_rows"])  # noqa: E731
                         & (d.NumberOfFeatures <= cfg["max_features"]))

    # --- TabRepo part ---
    tr = tabrepo[tabrepo.tid.isin(binary_tids)]
    tr = tr[size_ok(tr)].merge(datasets[["did", "version", "description_160"]], on="did", how="left")
    tr = add_derived(tr).assign(source="tabrepo")

    # --- Extension part (D1) ---
    ext = datasets[(datasets.NumberOfClasses == 2) & size_ok(datasets)]
    ext = ext[~ext.did.isin(tabrepo.did) & ~ext.name.str.lower().isin(tabrepo.name.str.lower())]
    ext = ext[~ext.name.str.contains(EXCLUDE_NAME)]
    cv10 = tasks[tasks.estimation_procedure == "10-fold Crossvalidation"].groupby("did").tid.min()
    ext = ext[ext.did.isin(cv10.index)].assign(tid=lambda d: d.did.map(cv10))
    ext = add_derived(ext)
    ext = ext[ext.missing_row_frac >= MIN_MISSING_ROW_FRACTION]
    ext = ext.sort_values(["name", "version"]).drop_duplicates("name").assign(source="extension")

    cols = ["source", "did", "tid", "name", "version", "NumberOfInstances", "NumberOfFeatures",
            "NumberOfNumericFeatures", "NumberOfSymbolicFeatures", "missing_cell_frac", "missing_row_frac",
            "imbalance_ratio", "description_160"]
    out = pd.concat([tr[cols], ext[cols]], ignore_index=True)
    target = OUTPUTS / "phase1" / "candidates.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    print(out.groupby("source").size().to_string())
    print(f"-> {target}")


if __name__ == "__main__":
    main()
