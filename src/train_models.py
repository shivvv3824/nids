"""
Phase 2 — train baseline and tree-based detectors on engineered NSL-KDD features.

SOC deployments care about **false positive rate** (benign traffic wrongly
alerted) as much as raw accuracy. We therefore:

- Fit a fast **logistic regression** baseline with ``class_weight='balanced'``
  to mimic a linear alert threshold policy analysts can reason about.
- Tune **RandomForest** and **XGBoost** with ``RandomizedSearchCV`` using
  ``roc_auc`` (ranking quality) while **refitting the best estimator on the
  full training matrix** so operating-point metrics (accuracy / FPR) reflect
  all available normal traffic structure.

Hyperparameter search intentionally uses a **stratified subsample** of the
training split for speed on laptops; final models are always refit on the
complete ``X_train``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

from .feature_engine import FeatureMatrixBundle


@dataclass
class TrainedModelPack:
    """Paths and metadata written by ``train_all``."""

    baseline_path: Path
    random_forest_path: Path
    xgboost_path: Path
    metadata_path: Path


def _resolve_parallel_jobs() -> int:
    """
    Return a safe ``n_jobs`` value for this runtime.

    Some restricted environments deny access to semaphore-related sysconf keys,
    which causes joblib/loky multiprocessing to fail early. In that case we
    force single-process execution so training still completes.
    """
    # Safe default for broad portability (sandboxes, CI, constrained macOS envs).
    # Set NIDS_N_JOBS to an integer (for example, -1 or 8) to opt into parallelism.
    configured = os.getenv("NIDS_N_JOBS")
    if configured is None:
        return 1
    try:
        return int(configured)
    except ValueError:
        return 1


def _stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[0] <= max_samples:
        return X, y
    X_s, _, y_s, _ = train_test_split(
        X,
        y,
        train_size=max_samples,
        stratify=y,
        random_state=random_state,
        shuffle=True,
    )
    return X_s, y_s


def _scale_pos_weight(y: np.ndarray) -> float:
    neg = int(np.sum(y == 0))
    pos = int(np.sum(y == 1))
    return float(neg / max(pos, 1))


def train_baseline(X_train: np.ndarray, y_train: np.ndarray, random_state: int) -> LogisticRegression:
    """
    Linear baseline: cheap to train, good for FPR benchmarking.

    ``class_weight='balanced'`` reweights the loss toward the minority class,
    which often reduces missed attacks at the cost of more benign alerts—
    exactly the trade analysts tune with thresholds in SIEM rules.
    """
    model = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=_resolve_parallel_jobs(),
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest_tuned(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int,
    search_subsample: int,
    n_iter: int,
    cv_folds: int,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    n_jobs = _resolve_parallel_jobs()
    X_s, y_s = _stratified_subsample(X_train, y_train, search_subsample, random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    param_dist: Dict[str, Any] = {
        "n_estimators": [200, 400, 600, 800],
        "max_depth": [12, 16, 24, 32, None],
        "min_samples_leaf": [1, 2, 4, 8],
        "min_samples_split": [2, 4, 8],
        "max_features": ["sqrt", 0.3, 0.5, 0.7],
        "class_weight": ["balanced_subsample", "balanced"],
    }
    base = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
        refit=True,
    )
    search.fit(X_s, y_s)
    best_params = dict(search.best_params_)
    final = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs, **best_params)
    final.fit(X_train, y_train)
    return final, {"best_params": best_params, "best_cv_roc_auc": float(search.best_score_)}


def _train_hist_gradient_fallback(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int,
    search_subsample: int,
    n_iter: int,
    cv_folds: int,
) -> Tuple[HistGradientBoostingClassifier, Dict[str, Any]]:
    """
    Fallback when ``xgboost`` cannot load (common macOS issue: missing ``libomp``).

    Uses ``HistGradientBoostingClassifier`` with a comparable boosted-tree search.
    """
    n_jobs = _resolve_parallel_jobs()
    X_s, y_s = _stratified_subsample(X_train, y_train, search_subsample, random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    param_dist: Dict[str, Any] = {
        "learning_rate": [0.03, 0.05, 0.08, 0.12],
        "max_depth": [4, 6, 8, 10, None],
        "max_leaf_nodes": [15, 31, 63, 127],
        "min_samples_leaf": [10, 20, 40],
        "l2_regularization": [0.0, 0.1, 1.0, 5.0],
        "max_iter": [200, 400, 600],
    }
    base = HistGradientBoostingClassifier(
        loss="log_loss",
        random_state=random_state,
        class_weight="balanced",
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
        refit=True,
    )
    search.fit(X_s, y_s)
    best_params = dict(search.best_params_)
    final = HistGradientBoostingClassifier(
        loss="log_loss",
        random_state=random_state,
        class_weight="balanced",
        **best_params,
    )
    final.fit(X_train, y_train)
    meta = {
        "best_params": best_params,
        "best_cv_roc_auc": float(search.best_score_),
        "backend": "sklearn_hist_gradient_boosting",
    }
    return final, meta


def train_xgboost_tuned(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int,
    search_subsample: int,
    n_iter: int,
    cv_folds: int,
) -> Tuple[Any, Dict[str, Any]]:
    n_jobs = _resolve_parallel_jobs()
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover - platform-specific binary deps
        print(
            "[Phase 2] XGBoost import failed; falling back to "
            f"HistGradientBoostingClassifier. Original error: {exc}"
        )
        model, meta = _train_hist_gradient_fallback(
            X_train,
            y_train,
            random_state=random_state,
            search_subsample=search_subsample,
            n_iter=n_iter,
            cv_folds=cv_folds,
        )
        return model, meta

    X_s, y_s = _stratified_subsample(X_train, y_train, search_subsample, random_state)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    spw = _scale_pos_weight(y_train)
    param_dist: Dict[str, Any] = {
        "n_estimators": [300, 500, 700, 900],
        "max_depth": [4, 6, 8, 10, 12],
        "learning_rate": [0.02, 0.05, 0.08, 0.12],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.85, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0, 10.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.1, 0.3],
    }
    base = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
        scale_pos_weight=spw,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
        refit=True,
    )
    search.fit(X_s, y_s)
    best_params = dict(search.best_params_)
    final = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
        scale_pos_weight=spw,
        **best_params,
    )
    final.fit(X_train, y_train)
    meta = {"best_params": best_params, "best_cv_roc_auc": float(search.best_score_), "backend": "xgboost"}
    return final, meta


def train_all(
    bundle: FeatureMatrixBundle,
    models_dir: Path,
    *,
    random_state: int = 42,
    search_subsample: int = 50_000,
    rf_search_iter: int = 18,
    xgb_search_iter: int = 18,
    cv_folds: int = 3,
    fast: bool = False,
) -> TrainedModelPack:
    """
    Fit baseline + tuned RF + tuned XGBoost and persist with joblib.

    Parameters
    ----------
    fast
        If ``True``, shrink search iterations and subsample budget for quick CI
        / smoke runs (not representative of full-study quality).
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    if fast:
        rf_search_iter = min(rf_search_iter, 6)
        xgb_search_iter = min(xgb_search_iter, 6)
        search_subsample = min(search_subsample, 12_000)
        cv_folds = 2

    X_train = bundle.X_train
    y_train = bundle.y_train

    print("[Phase 2] Training baseline logistic regression...")
    baseline = train_baseline(X_train, y_train, random_state)

    print("[Phase 2] Randomized search: RandomForest (roc_auc)...")
    rf_model, rf_meta = train_random_forest_tuned(
        X_train,
        y_train,
        random_state=random_state,
        search_subsample=search_subsample,
        n_iter=rf_search_iter,
        cv_folds=cv_folds,
    )

    print("[Phase 2] Randomized search: XGBoost (roc_auc)...")
    xgb_model, xgb_meta = train_xgboost_tuned(
        X_train,
        y_train,
        random_state=random_state,
        search_subsample=search_subsample,
        n_iter=xgb_search_iter,
        cv_folds=cv_folds,
    )

    baseline_path = models_dir / "baseline_logreg.joblib"
    rf_path = models_dir / "random_forest.joblib"
    xgb_path = models_dir / "xgboost.joblib"
    meta_path = models_dir / "training_metadata.joblib"

    joblib.dump(baseline, baseline_path)
    joblib.dump(rf_model, rf_path)
    joblib.dump(xgb_model, xgb_path)
    joblib.dump(
        {
            "random_forest": rf_meta,
            "xgboost": xgb_meta,
            "search_subsample": search_subsample,
            "fast_mode": fast,
        },
        meta_path,
    )

    print(f"[Phase 2] Saved baseline → {baseline_path}")
    print(f"[Phase 2] Saved RandomForest → {rf_path}")
    print(f"[Phase 2] Saved XGBoost → {xgb_path}")
    return TrainedModelPack(
        baseline_path=baseline_path,
        random_forest_path=rf_path,
        xgboost_path=xgb_path,
        metadata_path=meta_path,
    )


def load_trained_models(models_dir: Path) -> Dict[str, Any]:
    """Load models written by ``train_all``."""
    return {
        "baseline_logreg": joblib.load(models_dir / "baseline_logreg.joblib"),
        "random_forest": joblib.load(models_dir / "random_forest.joblib"),
        "xgboost": joblib.load(models_dir / "xgboost.joblib"),
        "metadata": joblib.load(models_dir / "training_metadata.joblib"),
    }
