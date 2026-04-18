"""
Phase 3 — evaluation metrics aligned with SOC operations.

We emphasize **false positive rate** (FPR): benign sessions incorrectly
flagged as attacks directly translate to analyst fatigue. ``roc_auc_score``
captures ranking quality of ``predict_proba`` outputs, while confusion
matrices make TN/FP/FN/TP explicit for post-incident review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from .feature_engine import FeatureMatrixBundle


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    FPR among negatives (label ``0`` = benign): FP / (FP + TN).

    Matches the intrusion-detection convention where class ``1`` is attack.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denom = fp + tn
    return float(fp / denom) if denom > 0 else 0.0


def detection_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Overall accuracy (not attack-only detection rate)."""
    return float(accuracy_score(y_true, y_pred))


@dataclass
class ModelEvaluation:
    name: str
    accuracy: float
    fpr: float
    roc_auc: Optional[float]
    confusion: List[List[int]]
    classification_report: str


def evaluate_estimator(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    name: str,
    threshold: float = 0.5,
) -> ModelEvaluation:
    """Score a sklearn-compatible classifier with optional probability AUC."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        y_hat = (proba >= threshold).astype(np.int32)
        try:
            auc = float(roc_auc_score(y_test, proba))
        except ValueError:
            auc = None
    else:
        y_hat = model.predict(X_test).astype(np.int32)
        auc = None

    cm = confusion_matrix(y_test, y_hat, labels=[0, 1])
    report = classification_report(y_test, y_hat, digits=4, zero_division=0)
    return ModelEvaluation(
        name=name,
        accuracy=detection_accuracy(y_test, y_hat),
        fpr=false_positive_rate(y_test, y_hat),
        roc_auc=auc,
        confusion=cm.tolist(),
        classification_report=report,
    )


def feature_importance_tables(
    models: Mapping[str, Any],
    feature_names: List[str],
    top_k: int = 20,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract ``feature_importances_`` from tree models when available.

    Logistic regression coefficients are on a different scale; we skip them
    here and surface only RF/XGB importances for IoC-style ranking.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for key in ("random_forest", "xgboost"):
        model = models.get(key)
        if model is None or not hasattr(model, "feature_importances_"):
            continue
        scores = np.asarray(model.feature_importances_, dtype=float)
        order = np.argsort(-scores)[:top_k]
        out[key] = [
            {"feature": feature_names[i], "importance": float(scores[i])} for i in order
        ]
    if out:
        rf_rows = {row["feature"]: row["importance"] for row in out.get("random_forest", [])}
        xgb_rows = {row["feature"]: row["importance"] for row in out.get("xgboost", [])}
        union = sorted(set(rf_rows) | set(xgb_rows), key=lambda f: -(rf_rows.get(f, 0.0) + xgb_rows.get(f, 0.0)))
        blended: List[Dict[str, Any]] = []
        for feat in union[:top_k]:
            blended.append(
                {
                    "feature": feat,
                    "rf": float(rf_rows.get(feat, 0.0)),
                    "xgb": float(xgb_rows.get(feat, 0.0)),
                    "mean": float((rf_rows.get(feat, 0.0) + xgb_rows.get(feat, 0.0)) / 2.0),
                }
            )
        out["blended_top"] = blended
    return out


def summarize_fpr_delta(baseline_fpr: float, model_fpr: float) -> Dict[str, float]:
    """Relative FPR reduction vs baseline (higher is better for SOC)."""
    if baseline_fpr <= 0:
        rel = 0.0 if model_fpr <= 0 else float("nan")
    else:
        rel = float((baseline_fpr - model_fpr) / baseline_fpr)
    return {"baseline_fpr": baseline_fpr, "model_fpr": model_fpr, "relative_fpr_drop": rel}


def run_full_evaluation(
    bundle: FeatureMatrixBundle,
    models: Mapping[str, Any],
    *,
    output_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Evaluate all models on the held-out test matrix and optionally persist JSON.
    """
    rows: List[Dict[str, Any]] = []
    baseline_eval: Optional[ModelEvaluation] = None
    for name, key in (
        ("Baseline (LogisticRegression)", "baseline_logreg"),
        ("RandomForest (tuned)", "random_forest"),
        ("XGBoost (tuned)", "xgboost"),
    ):
        model = models.get(key)
        if model is None:
            continue
        ev = evaluate_estimator(model, bundle.X_test, bundle.y_test, name=name)
        if key == "baseline_logreg":
            baseline_eval = ev
        rows.append(
            {
                "name": ev.name,
                "accuracy": ev.accuracy,
                "fpr": ev.fpr,
                "roc_auc": ev.roc_auc,
                "confusion_matrix": ev.confusion,
            }
        )
        print(f"\n=== {ev.name} ===")
        print(f"Accuracy: {ev.accuracy:.4f} | FPR (benign→attack): {ev.fpr:.4f} | ROC-AUC: {ev.roc_auc}")
        print(ev.classification_report)

    fpr_deltas: Dict[str, Dict[str, float]] = {}
    if baseline_eval is not None:
        base_fpr = baseline_eval.fpr
        print("\n=== FPR vs baseline (benign false alarms) ===")
        for ev in rows[1:]:
            delta = summarize_fpr_delta(base_fpr, ev["fpr"])
            fpr_deltas[ev["name"]] = delta
            pct = 100.0 * delta["relative_fpr_drop"]
            print(f"{ev['name']}: FPR {ev['fpr']:.4f} vs baseline {base_fpr:.4f} → relative drop {pct:.1f}%")

    importance = feature_importance_tables(models, bundle.feature_names, top_k=25)
    payload: Dict[str, Any] = {
        "evaluations": rows,
        "fpr_vs_baseline": fpr_deltas,
        "feature_importance": importance,
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n[Phase 3] Wrote evaluation JSON → {output_json}")
    return payload


def load_bundle(path: Path) -> FeatureMatrixBundle:
    return joblib.load(path)
