#!/usr/bin/env python3
"""
NIDS pipeline orchestrator (NSL-KDD).

This entrypoint wires modular phases so security engineers can reproduce
experiments from raw PCAP-derived connection summaries through model training,
evaluation, and SOC-style visualization. Each phase mirrors common MLSec
workflow gates: understand data (Phase~1), fit detectors (Phase~2), measure
operational impact like FPR (Phase~3), and communicate findings (Phase~4).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

# Allow ``python main.py`` from the ``nids_project`` directory without install.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_loader import describe_splits, load_nsl_kdd
from src.feature_engine import fit_preprocessing, summarize_engineered_columns


def _default_data_dir() -> Path:
    return _PROJECT_ROOT / "data"


def _artifacts_dir() -> Path:
    path = _PROJECT_ROOT / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _models_dir() -> Path:
    path = _PROJECT_ROOT / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_phase_1(
    data_dir: Path,
    train_file: str,
    test_file: str,
    artifacts_root: Path,
) -> Path:
    """
    Phase 1 — ingest NSL-KDD, clean, engineer features, fit preprocessing.

    Persists a joblib bundle with dense matrices and the sklearn preprocessor
    so later phases can train without re-parsing raw text.
    """
    artifacts_root.mkdir(parents=True, exist_ok=True)
    print("[Phase 1] Loading NSL-KDD...")
    train_df, test_df = load_nsl_kdd(
        data_dir,
        train_filename=train_file,
        test_filename=test_file,
    )
    print(describe_splits(train_df, test_df))

    engineered_preview = summarize_engineered_columns(train_df)
    print(
        f"[Phase 1] Engineered {len(engineered_preview)} derived columns "
        f"(preview: {', '.join(engineered_preview[:8])}...)"
    )

    print("[Phase 1] Fitting preprocessing on training split only...")
    bundle = fit_preprocessing(train_df, test_df)
    print(
        f"[Phase 1] Matrix shapes — "
        f"X_train: {bundle.X_train.shape}, X_test: {bundle.X_test.shape}"
    )

    artifact_path = artifacts_root / "phase1_feature_bundle.joblib"
    joblib.dump(bundle, artifact_path)
    print(f"[Phase 1] Saved preprocessing bundle → {artifact_path}")
    return artifact_path


def run_phase_2(bundle_path: Path, models_root: Path, *, fast: bool) -> None:
    """Phase 2 — baseline + tuned RandomForest + tuned XGBoost."""
    from src.train_models import train_all

    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Missing feature bundle: {bundle_path}. Run ``python main.py --phase 1`` first."
        )
    print(f"[Phase 2] Loading feature bundle from {bundle_path}")
    bundle = joblib.load(bundle_path)
    train_all(bundle, models_root, fast=fast)


def run_phase_3(bundle_path: Path, models_root: Path, artifacts_root: Path) -> None:
    """Phase 3 — metrics, ROC-AUC, FPR vs baseline, feature importances."""
    from src.evaluate import run_full_evaluation
    from src.train_models import load_trained_models

    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing feature bundle: {bundle_path}")
    required = [
        models_root / "baseline_logreg.joblib",
        models_root / "random_forest.joblib",
        models_root / "xgboost.joblib",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing trained model artifacts: "
            + ", ".join(str(p) for p in missing)
            + ". Run ``python main.py --phase 2`` first."
        )
    bundle = joblib.load(bundle_path)
    models = load_trained_models(models_root)
    run_full_evaluation(bundle, models, output_json=artifacts_root / "evaluation_report.json")


def run_phase_4(bundle_path: Path, models_root: Path, artifacts_root: Path) -> None:
    """Phase 4 — render SOC-style Matplotlib/Seaborn dashboard."""
    if not bundle_path.exists():
        raise FileNotFoundError(f"Missing feature bundle: {bundle_path}")
    out = artifacts_root / "soc_dashboard.png"
    print(f"[Phase 4] Rendering dashboard → {out}")
    from dashboard.threat_visualizer import render_dashboard_from_disk

    render_dashboard_from_disk(bundle_path, models_root, out, embedding_method="pca")
    print(f"[Phase 4] Dashboard saved → {out}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSL-KDD NIDS pipeline orchestrator")
    parser.add_argument(
        "--phase",
        choices=("1", "2", "3", "4", "all"),
        default="1",
        help="Pipeline phase to execute (``all`` runs 1→4 sequentially).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir(),
        help="Directory containing NSL-KDD train/test files.",
    )
    parser.add_argument(
        "--train-file",
        default="KDDTrain+.txt",
        help="Training file name inside data-dir (NSL Train+).",
    )
    parser.add_argument(
        "--test-file",
        default="KDDTest+.txt",
        help="Test file name inside data-dir (NSL Test+).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=_artifacts_dir(),
        help="Directory for phase outputs (feature bundle, reports, dashboard).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=_models_dir(),
        help="Directory for joblib-serialized estimators.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Smaller hyperparameter search for smoke tests / laptops.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir: Path = args.data_dir.expanduser().resolve()
    artifacts_root: Path = args.artifacts_dir.expanduser().resolve()
    models_root: Path = args.models_dir.expanduser().resolve()
    artifacts_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    bundle_path = artifacts_root / "phase1_feature_bundle.joblib"

    if args.phase in {"1", "all"}:
        run_phase_1(data_dir, args.train_file, args.test_file, artifacts_root)
    if args.phase in {"2", "all"}:
        run_phase_2(bundle_path, models_root, fast=args.fast)
    if args.phase in {"3", "all"}:
        run_phase_3(bundle_path, models_root, artifacts_root)
    if args.phase in {"4", "all"}:
        run_phase_4(bundle_path, models_root, artifacts_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
