"""
NSL-KDD ingestion and cleaning for network intrusion detection.

This module loads the NSL-KDD benchmark, which is derived from DARPA/KDD
connection records. Each row describes a TCP/IP session summarized by
statistical features (byte counts, error rates, service affinity, etc.).
From a SOC perspective, the goal is to separate benign ``normal`` flows
from attack-labeled flows while controlling false positives that would
flood analysts with benign alerts.

The raw files often ship **without a header row** and may include a
``difficulty`` column (Train+) that is not a predictive feature; we strip
it so models are not accidentally fit on evaluation metadata.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# Canonical NSL-KDD / KDD Cup 99 feature names (41 attributes + label).
# Train+ appends ``difficulty`` as the 43rd column; Test+ is typically 42 cols.
_KDD_FEATURE_NAMES: list[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
]

_TRAIN_PLUS_DIFFICULTY = "difficulty"


def _resolve_paths(
    data_dir: Path,
    train_name: str,
    test_name: str,
) -> Tuple[Path, Path]:
    train_path = data_dir / train_name
    test_path = data_dir / test_name
    return train_path, test_path


def _read_table(path: Path) -> pd.DataFrame:
    """
    Read a NSL-KDD table from disk.

    Supports comma-separated ``.csv``/``.txt`` exports. NSL artifacts are
    often distributed as ``KDDTrain+.txt`` without headers; we inject names
    explicitly to avoid column drift across environments.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. "
            "Download NSL-KDD (Train+ / Test+) and place them under data/."
        )
    return pd.read_csv(path, header=None, low_memory=False)


def _assign_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Map positional columns to semantic names and drop the NSL ``difficulty``
    helper if present (analysts should not train on evaluation hints).
    """
    ncols = raw.shape[1]
    if ncols == len(_KDD_FEATURE_NAMES):
        raw.columns = list(_KDD_FEATURE_NAMES)
    elif ncols == len(_KDD_FEATURE_NAMES) + 1:
        raw.columns = list(_KDD_FEATURE_NAMES) + [_TRAIN_PLUS_DIFFICULTY]
        raw = raw.drop(columns=[_TRAIN_PLUS_DIFFICULTY])
    else:
        raise ValueError(
            f"Unexpected column count {ncols} in NSL-KDD file; "
            f"expected {len(_KDD_FEATURE_NAMES)} or {len(_KDD_FEATURE_NAMES) + 1}."
        )
    return raw


def _strip_object_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _clean_categoricals(df: pd.DataFrame, cat_cols: Iterable[str]) -> pd.DataFrame:
    """
    Normalize categorical tokens (trim whitespace) and replace NSL sentinels.

    Missing values in NSL are often encoded as ``?``; treating those as NaN
    lets downstream encoders impute or ignore unknown levels consistently.
    """
    out = df.copy()
    for col in cat_cols:
        if col not in out.columns:
            continue
        out[col] = _strip_object_series(out[col])
        out[col] = out[col].replace({"?": np.nan})
    return out


def _clean_numeric(df: pd.DataFrame, num_cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in num_cols:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _impute(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    """
    Simple imputation tuned for connection metadata:

    - Numeric gaps become ``0.0`` (conservative "no signal" for rates/counts).
    - Categorical gaps become ``"unknown"`` so OneHotEncoder can bucket them.

    Analyst note: production NIDS should learn imputation statistics only from
    training traffic to avoid leakage; callers should fit imputers on train
    and apply to test. Phase~1 keeps the logic explicit in ``clean_nsl_kdd``
    with a ``reference`` frame for test-time imputation.
    """
    out = df.copy()
    for c in num_cols:
        out[c] = out[c].fillna(0.0)
    for c in cat_cols:
        out[c] = out[c].fillna("unknown")
    return out


def _build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create SOC-friendly targets:

    - ``label`` retains the original NSL attack family string for reporting.
    - ``binary_label`` collapses everything except ``normal`` to ``1`` (attack).

    Binary targets align with alert triage: analysts first decide benign vs
    malicious, then pivot to attack taxonomy for playbooks.
    """
    out = df.copy()
    out["label"] = _strip_object_series(out["label"]).str.lower()
    out["binary_label"] = (out["label"] != "normal").astype(np.int8)
    return out


def clean_nsl_kdd(
    df: pd.DataFrame,
    *,
    reference: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Clean a NSL-KDD frame: typing, sentinel handling, imputation, targets.

    Parameters
    ----------
    df
        Raw frame after ``_assign_columns``.
    reference
        If provided (typically the training split), categorical ``mode`` used
        for any remaining NaNs before the ``unknown`` bucket—reduces test-time
        distortion when a category was unseen during cleaning.
    """
    cat_cols = ["protocol_type", "service", "flag"]
    numeric_candidates = [c for c in df.columns if c not in cat_cols + ["label"]]

    cleaned = _clean_categoricals(df, cat_cols)
    cleaned = _clean_numeric(cleaned, numeric_candidates)
    cleaned = _build_targets(cleaned)

    ref = reference if reference is not None else cleaned
    for c in cat_cols:
        if cleaned[c].isna().any():
            mode = ref[c].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else "unknown"
            cleaned[c] = cleaned[c].fillna(fill)

    cleaned = _impute(cleaned, numeric_candidates, cat_cols)
    return cleaned


def load_nsl_kdd(
    data_dir: str | os.PathLike[str],
    *,
    train_filename: str = "KDDTrain+.txt",
    test_filename: str = "KDDTest+.txt",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load NSL-KDD train/test splits from ``data_dir``.

    Expected layout
    ---------------
    ``{data_dir}/{train_filename}`` and ``{data_dir}/{test_filename}`` in the
    canonical NSL comma-separated format without headers.

    Returns
    -------
    train_df, test_df
        Cleaned tables with ``label`` (multi-class string) and ``binary_label``.
    """
    root = Path(data_dir)
    train_path, test_path = _resolve_paths(root, train_filename, test_filename)

    train_raw = _assign_columns(_read_table(train_path))
    test_raw = _assign_columns(_read_table(test_path))

    train_df = clean_nsl_kdd(train_raw)
    test_df = clean_nsl_kdd(test_raw, reference=train_df)
    return train_df, test_df


def describe_splits(train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    """Human-readable summary for logging in ``main``."""
    lines = [
        "NSL-KDD split summary",
        f"  Train rows: {len(train_df):,} | Test rows: {len(test_df):,}",
        f"  Train attack rate: {train_df['binary_label'].mean():.3f}",
        f"  Test attack rate:  {test_df['binary_label'].mean():.3f}",
    ]
    return "\n".join(lines)
