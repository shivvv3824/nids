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


def run_phase_1(data_dir: Path, train_file: str, test_file: str) -> Path:
    """
    Phase 1 — ingest NSL-KDD, clean, engineer features, fit preprocessing.

    Persists a joblib bundle with dense matrices and the sklearn preprocessor
    so later phases can train without re-parsing raw text.
    """
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

    artifact_path = _artifacts_dir() / "phase1_feature_bundle.joblib"
    joblib.dump(bundle, artifact_path)
    print(f"[Phase 1] Saved preprocessing bundle → {artifact_path}")
    return artifact_path


def run_phase_2_stub() -> None:
    print(
        "[Phase 2] Pending: train RandomForest / XGBoost with "
        "`src/train_models.py` (hyperparameter search + joblib export)."
    )


def run_phase_3_stub() -> None:
    print(
        "[Phase 3] Pending: metrics + FPR comparison in `src/evaluate.py` "
        "(confusion matrix, ROC-AUC, feature importance extraction)."
    )


def run_phase_4_stub() -> None:
    print(
        "[Phase 4] Pending: SOC triage dashboard in "
        "`dashboard/threat_visualizer.py`."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSL-KDD NIDS pipeline orchestrator")
    parser.add_argument(
        "--phase",
        choices=("1", "2", "3", "4", "all"),
        default="1",
        help="Pipeline phase to execute (default runs Phase 1 preprocessing).",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir: Path = args.data_dir.expanduser().resolve()

    if args.phase in {"1", "all"}:
        run_phase_1(data_dir, args.train_file, args.test_file)
    if args.phase in {"2", "all"}:
        run_phase_2_stub()
    if args.phase in {"3", "all"}:
        run_phase_3_stub()
    if args.phase in {"4", "all"}:
        run_phase_4_stub()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
