"""Readiness measures (analysis plan section 5), one function per group.

Every group takes the dataset as loaded (leakage fixes applied) and returns
{measure_name: value}. DIRECTION fixes, before any comparison with the gold
standard, whether a higher value means "more ready" (+1) or "less ready" (-1).

Row caps (stratified subsamples, documented in the plan): 50,000 rows for all
groups unless stated; PyMFE complexity 5,000; landmarkers 10,000. The cheap
empirical score trains on at most 100,000 rows of each OpenML training fold
(k-NN: 20,000, as in the thesis) and is scored on the full test fold.
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from . import data

DEFAULT_ROWS = 50_000
COMPLEXITY_ROWS = 5_000
LANDMARK_ROWS = 10_000
EMPIRICAL_TRAIN_ROWS = 100_000
EMPIRICAL_KNN_ROWS = 20_000

DIRECTION: dict[str, int] = {}


def _register(directions: dict[str, int]) -> None:
    DIRECTION.update(directions)


# ------------------------------------------------------------------ helpers
def stratified_subsample(X: pd.DataFrame, y: pd.Series, n: int, seed: int = 0):
    if len(y) <= n:
        return X.reset_index(drop=True), y.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y.to_numpy() == cls)
        k = max(2, int(round(n * len(idx) / len(y))))
        keep.append(rng.choice(idx, min(k, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    return X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)


def categorical_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) or pd.api.types.is_bool_dtype(X[c])]


def categoricals_as_object(X: pd.DataFrame) -> pd.DataFrame:
    """Categoricals as plain Python strings with np.nan for missing (what scikit-learn expects)."""
    X = X.copy()
    for c in categorical_columns(X):
        s = X[c]
        X[c] = s.astype("string").astype(object).where(s.notna(), np.nan)
    return X


def ordinal_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """Categoricals as integer codes (missing stays NaN), everything float."""
    X = X.copy()
    for c in categorical_columns(X):
        codes = X[c].astype("category").cat.codes.astype("float64")
        X[c] = codes.where(codes >= 0)
    return X.astype("float64")


def sym_auc(y, score) -> float:
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


# ------------------------------------------------------------------ R1a AIDRIN
_register({
    "aidrin_completeness": +1, "aidrin_duplicity": -1, "aidrin_outliers": -1, "aidrin_constant_frac": -1,
    "aidrin_max_pairwise_corr": -1, "aidrin_max_abs_skew": -1, "aidrin_max_abs_kurtosis": -1,
    "aidrin_row_completeness": +1, "aidrin_feature_coverage": +1, "aidrin_relevance_max": +1,
    "aidrin_relevance_mean_top5": +1, "aidrin_imbalance_degree": -1,
})


def aidrin(X: pd.DataFrame, y: pd.Series) -> dict:
    """AIDRIN (Hiniduma et al., SSDBM 2024), headless API, file with features + target."""
    from aidrin.headless.api import run_metric

    X, y = stratified_subsample(X, y, DEFAULT_ROWS)
    cat = categorical_columns(X)
    num = [c for c in X.columns if c not in cat]
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "data.csv")
        X.assign(target=y.to_numpy()).to_csv(path, index=False)
        run = lambda name, **kw: run_metric(name, path, save_images=False, strip_visualizations=True, **kw)  # noqa: E731
        out = {}
        comp = run("completeness")["Completeness scores"]
        out["aidrin_completeness"] = float(comp.get("Overall Completeness", np.mean(
            [v for k, v in comp.items() if k != "target"])))
        out["aidrin_duplicity"] = float(run("duplicity")["Duplicity scores"]["Overall duplicity of the dataset"])
        out["aidrin_outliers"] = float(run("outliers")["Outlier scores"]["Overall outlier score"])
        const = run("constant_feature_count")
        out["aidrin_constant_frac"] = const["Constant feature count"] / max(const["Total features"], 1)
        out["aidrin_max_pairwise_corr"] = float(run("max_pairwise_correlation").get("Max Pairwise Correlation", np.nan))
        out["aidrin_max_abs_skew"] = float(run("skewness").get("Max Absolute Skewness", np.nan))
        out["aidrin_max_abs_kurtosis"] = float(run("kurtosis").get("Max Absolute Excess Kurtosis", np.nan))
        out["aidrin_row_completeness"] = float(run("row_level_completeness", required_columns=list(X.columns))
                                               ["Row-Level Completeness (%)"])
        out["aidrin_feature_coverage"] = float(run("feature_coverage_ratio", threshold=0.9)
                                               ["Feature Coverage Ratio (%)"])
        rel = run("feature_relevance", cat_columns=cat, num_columns=num, target_column="target")
        corr = np.abs(np.array([v for v in rel.get("Pearson Correlation to Target", {}).values()
                                if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float))
        out["aidrin_relevance_max"] = float(corr.max()) if corr.size else np.nan
        out["aidrin_relevance_mean_top5"] = float(np.sort(corr)[-5:].mean()) if corr.size else np.nan
        imb = run("class_imbalance", target_column="target").get("Imbalance degree", {})
        out["aidrin_imbalance_degree"] = float(imb.get("Imbalance Degree score", np.nan))
    return out


# ------------------------------------------------------------------ R1c ICIRCA DQI
_register({"icirca_dqi": +1, "icirca_dqi_corrected": +1})


def icirca(X: pd.DataFrame, y: pd.Series) -> dict:
    """Dataset Quality Index of Babu et al. (ICIRCA 2026), reimplemented from
    github.com/Meeravali-shaik/An-Automated-Framework-for-Dataset-Quality-Assessment
    at commit 28bad02 (dqi/dqi_calculator.py, dataset_profiler/profiler.py).
    The profile is computed on the frame with the target as last column, as in the tool.
    'icirca_dqi' reproduces the tool exactly, including its treatment of missing and
    duplicate percentages <= 1 as ratios; 'icirca_dqi_corrected' uses true ratios."""
    X, y = stratified_subsample(X, y, DEFAULT_ROWS)
    df = X.assign(target=y.to_numpy())
    cells = df.shape[0] * df.shape[1]
    missing_pct = round(df.isna().sum().sum() / cells * 100, 2)
    duplicate_pct = round(df.duplicated().sum() / len(df) * 100, 2)
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] >= 2:
        cm = numeric.corr().abs().to_numpy(dtype=float)
        vals = cm[~np.eye(cm.shape[0], dtype=bool)]
        mean_corr = float(np.nanmean(vals)) if np.isfinite(vals).any() else 0.0
    else:
        mean_corr = 0.0
    skew = numeric.skew(skipna=True).abs().dropna() if not numeric.empty else pd.Series(dtype=float)
    mean_skew = float(skew.mean()) if len(skew) else 0.0
    counts = df["target"].value_counts()
    imbalance = counts.max() / counts.min()

    def penalties(missing_ratio, duplicate_ratio):
        p = 0
        p += 0 if missing_ratio < 0.05 else 5 if missing_ratio < 0.15 else 10
        p += 0 if imbalance < 1.5 else 5 if imbalance < 3 else 10
        p += 0 if mean_corr < 0.3 else 5 if mean_corr < 0.6 else 10
        p += 0 if mean_skew < 1 else 5 if mean_skew < 2 else 10
        p += 0 if duplicate_ratio < 0.02 else 5 if duplicate_ratio < 0.05 else 10
        return max(0, min(100, 100 - p))

    as_ratio = lambda v: v / 100 if v > 1 else v  # noqa: E731  (the tool's _as_ratio)
    return {"icirca_dqi": penalties(as_ratio(missing_pct), as_ratio(duplicate_pct)),
            "icirca_dqi_corrected": penalties(missing_pct / 100, duplicate_pct / 100)}


# ------------------------------------------------------------------ R2 thesis score
_register({"thesis_c1_completeness": +1, "thesis_c2_balance": +1, "thesis_c3_sample_size": +1,
           "thesis_c4_feature_quality": +1, "thesis_manual_score": +1})


def thesis_manual(X: pd.DataFrame, y: pd.Series) -> dict:
    """Manual readiness score C1-C4 of Kemi (2026), ported from
    github.com/jonaskemi/MasterThesis (cleaning.py, encoding.py, meta_collector.py,
    readiness_scorer.py): dedupe, drop columns > 50% missing, impute, encode
    (binary -> 0/1, <= 30 levels -> one-hot drop-first, > 30 -> dropped)."""
    df = X.assign(target=y.to_numpy())
    missing_pct = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    df = df.drop_duplicates()
    df = df.loc[:, df.isna().mean() <= 0.5]
    feats = [c for c in df.columns if c != "target"]
    for c in feats:
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
        else:
            mode = df[c].mode(dropna=True)
            df[c] = df[c].astype("object").fillna(mode.iloc[0] if len(mode) else "missing")
    enc_parts = [df[["target"]]]
    for c in feats:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            enc_parts.append(s.to_frame())
        elif s.nunique() == 2:
            enc_parts.append((s == sorted(s.astype(str).unique())[0]).astype(int).to_frame(c))
        elif s.nunique() <= 30:
            enc_parts.append(pd.get_dummies(s.astype(str), prefix=c, drop_first=True).astype(int))
    enc = pd.concat(enc_parts, axis=1)

    counts = enc["target"].value_counts()
    imbalance = counts.max() / counts.min()
    numeric = enc.drop(columns="target").select_dtypes(include=[np.number])
    if numeric.shape[1] > 1:
        corr = numeric.sample(min(len(numeric), DEFAULT_ROWS), random_state=0).corr().to_numpy()
        upper = np.triu(corr, k=1)
        vals = upper[(upper != 0) & ~np.isnan(upper)]
        high_pairs = int((np.abs(vals) > 0.8).sum())
    else:
        high_pairs = 0
    low_var = int((numeric.var() < 0.01).sum())
    n_cols = max(enc.shape[1], 1)  # the thesis counted the target column too
    pairs = n_cols * (n_cols - 1) / 2 if n_cols > 1 else 1

    c1 = 1.0 - min(missing_pct / 100.0, 1.0)
    c2 = max(1.0 - math.log2(max(imbalance, 1.0)) / math.log2(100), 0.0)
    c3 = min(math.log10(max(len(enc), 1)) / 6.0, 1.0)
    c4 = 1.0 - min(high_pairs / pairs + low_var / n_cols, 1.0)
    return {"thesis_c1_completeness": c1, "thesis_c2_balance": c2, "thesis_c3_sample_size": c3,
            "thesis_c4_feature_quality": c4, "thesis_manual_score": (c1 + c2 + c3 + c4) / 4 * 100}


# ------------------------------------------------------------------ R3 complexity (PyMFE)
COMPLEXITY_FEATURES = ["f1", "f1v", "f2", "f3", "f4", "l1", "l2", "l3", "n1", "n2", "n3", "n4", "t1",
                       "lsc", "density", "cls_coef", "hubs", "t2", "t3", "t4", "c1", "c2"]
# Lorena et al. (2019): higher = more complex (less ready), except c1 (class entropy: higher = balanced)
_register({f"cx_{f}": (+1 if f == "c1" else -1) for f in COMPLEXITY_FEATURES})


def complexity(X: pd.DataFrame, y: pd.Series) -> dict:
    """PyMFE 'complexity' group (Lorena et al. 2019) on <= 5,000 rows; categoricals as
    ordinal codes, missing values imputed with the column median (PyMFE needs complete data)."""
    from pymfe.mfe import MFE

    X, y = stratified_subsample(X, y, COMPLEXITY_ROWS)
    Xn = ordinal_numeric(X)
    Xn = Xn.fillna(Xn.median()).fillna(0.0)
    Xn = Xn.loc[:, Xn.nunique() > 1]
    mfe = MFE(groups=["complexity"], summary=["mean"], random_state=0)
    mfe.fit(Xn.to_numpy(), y.to_numpy(), suppress_warnings=True)
    names, values = mfe.extract(suppress_warnings=True)
    got = {n.split(".")[0]: v for n, v in zip(names, values)}
    return {f"cx_{f}": (float(got[f]) if f in got and got[f] is not None else np.nan) for f in COMPLEXITY_FEATURES}


# ------------------------------------------------------------------ R4 relevance
_register({"rel_max_abs_r": +1, "rel_max_single_auc": +1, "rel_mean_top5_auc": +1, "rel_max_norm_mi": +1})


def relevance(X: pd.DataFrame, y: pd.Series) -> dict:
    """Cheap target-aware signals, no model training. Categoricals use 5-fold
    out-of-fold target encoding (as in the leakage screen)."""
    from sklearn.feature_selection import mutual_info_classif

    from .leakage import oof_target_encoding  # shared with the screen

    X, y = stratified_subsample(X, y, DEFAULT_ROWS)
    yv = y.to_numpy()
    cats = set(categorical_columns(X))
    aucs, rs = [], []
    encoded = {}
    for c in X.columns:
        s = X[c]
        if c in cats:
            v = oof_target_encoding(s.astype("string").fillna("__nan__").to_numpy(), yv)
            ok = np.ones(len(v), bool)
        else:
            v = s.to_numpy(dtype="float64")
            ok = ~np.isnan(v)
        encoded[c] = np.where(ok, v, np.nan)
        if ok.sum() < 20 or len(np.unique(v[ok])) < 2 or len(np.unique(yv[ok])) < 2:
            continue
        aucs.append(sym_auc(yv[ok], v[ok]))
        rs.append(abs(np.corrcoef(v[ok], yv[ok])[0, 1]))
    enc = pd.DataFrame(encoded)
    enc = enc.fillna(enc.median()).fillna(0.0)
    mi = mutual_info_classif(enc.to_numpy(), yv, discrete_features=[c in cats for c in enc.columns], random_state=0)
    p = yv.mean()
    h_y = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    aucs = np.sort(np.array(aucs))
    return {"rel_max_abs_r": float(np.nanmax(rs)) if rs else np.nan,
            "rel_max_single_auc": float(aucs[-1]) if aucs.size else np.nan,
            "rel_mean_top5_auc": float(aucs[-5:].mean()) if aucs.size else np.nan,
            "rel_max_norm_mi": float(mi.max() / h_y) if len(mi) else np.nan}


# ------------------------------------------------------------------ R5 landmarkers
LANDMARKERS = ["stump", "nn1", "naive_bayes", "logreg"]
_register({f"lm_{m}_auc": +1 for m in LANDMARKERS})


def _landmark_preprocessor(X: pd.DataFrame, scale: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, TargetEncoder

    cats = categorical_columns(X)
    nums = [c for c in X.columns if c not in cats]
    num_steps = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else [])
    cat_steps = [SimpleImputer(strategy="constant", fill_value="__nan__"), TargetEncoder(random_state=0)] + \
                ([StandardScaler()] if scale else [])
    return ColumnTransformer([("num", make_pipeline(*num_steps), nums), ("cat", make_pipeline(*cat_steps), cats)],
                             sparse_threshold=0)


def landmarkers(X: pd.DataFrame, y: pd.Series) -> dict:
    """Simple learners, 5-fold stratified CV AUC on <= 10,000 rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.tree import DecisionTreeClassifier

    X, y = stratified_subsample(X, y, LANDMARK_ROWS)
    X = categoricals_as_object(X)
    models = {
        "stump": (DecisionTreeClassifier(max_depth=1, random_state=0), False),
        "nn1": (KNeighborsClassifier(n_neighbors=1), True),
        "naive_bayes": (GaussianNB(), False),
        "logreg": (LogisticRegression(max_iter=2000), True),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    out = {}
    for name, (model, scale) in models.items():
        pipe = make_pipeline(_landmark_preprocessor(X, scale), model)
        out[f"lm_{name}_auc"] = float(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean())
    return out


# ------------------------------------------------------------------ cheap empirical score (thesis baselines)
BASELINES = ["logreg", "tree", "forest", "knn", "xgboost"]
_register({f"emp_{m}_auc": +1 for m in BASELINES} | {"emp_mean_auc": +1, "emp_mean_macro_f1": +1})


def empirical(X: pd.DataFrame, y: pd.Series, tid: int) -> dict:
    """The thesis's 5 untuned baselines on OpenML folds 0-2 (the gold-standard folds).
    Preprocessing fitted on the training fold: median/mode imputation, one-hot for
    <= 30 levels, drop > 30 levels (the thesis encoding)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    X = categoricals_as_object(X)
    cats = categorical_columns(X)
    low = [c for c in cats if X[c].nunique() <= 30]
    nums = [c for c in X.columns if c not in cats]

    def prep(scale):
        num = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else [])
        return ColumnTransformer([
            ("num", make_pipeline(*num), nums),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                  OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)), low),
        ], sparse_threshold=0)

    models = {
        "logreg": lambda: make_pipeline(prep(True), LogisticRegression(max_iter=5000, random_state=42)),
        "tree": lambda: make_pipeline(prep(False), DecisionTreeClassifier(random_state=42)),
        "forest": lambda: make_pipeline(prep(False), RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
        "knn": lambda: make_pipeline(prep(True), KNeighborsClassifier(n_neighbors=5, n_jobs=1)),
        "xgboost": lambda: make_pipeline(prep(False), XGBClassifier(n_estimators=100, random_state=42, n_jobs=1,
                                                                    eval_metric="logloss", verbosity=0)),
    }
    splits = pd.read_parquet(data.split_path(tid))
    aucs = {m: [] for m in models}
    f1s = {m: [] for m in models}
    for fold in (0, 1, 2):
        s = splits[splits.fold == fold]
        tr, te = s.row[~s.is_test].to_numpy(), s.row[s.is_test].to_numpy()
        for name, make in models.items():
            cap = EMPIRICAL_KNN_ROWS if name == "knn" else EMPIRICAL_TRAIN_ROWS
            Xtr, ytr = stratified_subsample(X.iloc[tr], y.iloc[tr], cap, seed=fold)
            model = make().fit(Xtr, ytr)
            proba = model.predict_proba(X.iloc[te])[:, 1]
            aucs[name].append(roc_auc_score(y.iloc[te], proba))
            f1s[name].append(f1_score(y.iloc[te], (proba >= 0.5).astype(int), average="macro", zero_division=0))
    out = {f"emp_{m}_auc": float(np.mean(v)) for m, v in aucs.items()}
    out["emp_mean_auc"] = float(np.mean(list(out.values())))
    out["emp_mean_macro_f1"] = float(np.mean([np.mean(v) for v in f1s.values()]) * 100)  # thesis empirical score
    return out


GROUPS = {"aidrin": aidrin, "icirca": icirca, "thesis": thesis_manual, "complexity": complexity,
          "relevance": relevance, "landmarkers": landmarkers, "empirical": empirical}


# ------------------------------------------------------------------ R6 TabPFN probe (GPU, run locally)
TABPFN_ROWS = 10_000
TABPFN_MAX_FEATURES = 500  # TabPFN v2 pre-training limit
_register({"tabpfn_auc": +1})


def tabpfn_probe(X: pd.DataFrame, y: pd.Series, device: str = "cuda") -> dict:
    """TabPFN v2 (Hollmann et al., Nature 2025; tabpfn==2.0.9, ungated v2 weights)
    as a cheap readiness probe: 5-fold stratified CV AUC on <= 10,000 rows, no tuning.
    Categoricals as ordinal codes, flagged to TabPFN as categorical; missing values
    passed through (TabPFN handles NaN). Datasets with > 500 features use a random
    (seeded) subset of 500 features."""
    from sklearn.model_selection import StratifiedKFold
    from tabpfn import TabPFNClassifier

    X, y = stratified_subsample(X, y, TABPFN_ROWS)
    cats = set(categorical_columns(X))
    if X.shape[1] > TABPFN_MAX_FEATURES:
        keep = np.sort(np.random.default_rng(0).choice(X.shape[1], TABPFN_MAX_FEATURES, replace=False))
        X = X.iloc[:, keep]
    cat_idx = [i for i, c in enumerate(X.columns) if c in cats]
    Xn = ordinal_numeric(X).to_numpy(dtype="float32")
    yv = y.to_numpy()
    aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xn, yv):
        clf = TabPFNClassifier(device=device, random_state=0, categorical_features_indices=cat_idx or None,
                               memory_saving_mode="auto")
        clf.fit(Xn[tr], yv[tr])
        aucs.append(roc_auc_score(yv[te], clf.predict_proba(Xn[te])[:, 1]))
    return {"tabpfn_auc": float(np.mean(aucs))}
