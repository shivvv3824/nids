"""
Feature engineering and scaling for NSL-KDD intrusion detection.

Connection records mix low-level protocol indicators (e.g., TCP flags
encoded as ``flag``), volume metrics (``src_bytes``/``dst_bytes``), and
aggregated behavioral rates (``*error_rate`` columns). Attackers often
distort these relationships: DoS floods inflate ``count`` with abnormal
error rates, probes generate ``REJ``/``S0``-heavy sequences, and R2L/U2R
flows may show post-authentication compromise signals.

This module engineers **interpretable** derived signals (protocol/flag
structure, byte asymmetry, error pressure) and applies **separate**
scaling for numeric fields versus **One-Hot Encoding** for categorical
metadata—mirroring how SOC analysts mentally bucket categorical indicators
while scrutinizing continuous anomalies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Categorical connection metadata used by both analysts and encoders.
_CATEGORICAL_COLS = ("protocol_type", "service", "flag")


@dataclass
class FeatureMatrixBundle:
    """Container for matrices, labels, and the fitted sklearn preprocessor."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    preprocessor: ColumnTransformer
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 20+ engineered columns on top of raw NSL-KDD attributes.

    The features intentionally blend:

    - **Protocol structure** — explicit ``is_tcp`` / ``is_udp`` / ``is_icmp``
      indicators to capture transport-level prevalence anomalies.
    - **TCP / handshake stress** — interactions between transport and
      ``*serror_rate`` (SYN/FIN irregularities) and ``*rerror_rate``
      (reject-like behaviors) commonly associated with scans and floods.
    - **Byte / rate asymmetry** — volumetric imbalance and intensity metrics
      approximate byte ratios and packet/second proxies using ``count`` and
      ``duration`` (analysts watch asymmetric exfiltration and floods).

    Notes
    -----
    All operations are row-local (safe before train/test split leakage controls
    in ``fit_preprocessing`` which fits scalers **only** on training rows).
    """
    out = df.copy()
    p = out["protocol_type"].astype(str).str.lower()
    f = out["flag"].astype(str)

    # --- Protocol distribution (transport-level one-hot style signals) ---
    out["proto_is_tcp"] = (p == "tcp").astype(np.float32)
    out["proto_is_udp"] = (p == "udp").astype(np.float32)
    out["proto_is_icmp"] = (p == "icmp").astype(np.float32)

    # --- Flag-oriented indicators (TCP control-plane abuse heuristics) ---
    # REJ/S0 patterns often dominate probe/scan noise in KDD-derived sets.
    out["flag_contains_rej"] = f.str.contains("rej", case=False, na=False).astype(np.float32)
    out["flag_is_s0"] = (f == "S0").astype(np.float32)
    out["flag_is_rst"] = f.str.contains("rst", case=False, na=False).astype(np.float32)
    out["flag_is_sf"] = (f == "SF").astype(np.float32)

    # --- Byte ratios & volumetric stress ---
    out["byte_ratio_src_to_dst"] = out["src_bytes"] / (out["dst_bytes"] + 1.0)
    out["byte_ratio_dst_to_src"] = out["dst_bytes"] / (out["src_bytes"] + 1.0)
    out["log_total_bytes"] = np.log1p(out["src_bytes"] + out["dst_bytes"])
    out["bytes_per_event"] = (out["src_bytes"] + out["dst_bytes"]) / (out["count"] + 1.0)

    # ``count`` approximates packets/second when duration is small (flood indicator).
    out["conn_events_per_sec"] = out["count"] / (out["duration"] + 1.0)
    out["srv_share"] = out["srv_count"] / (out["count"] + 1.0)

    # --- Error / rejection pressure (SYN + REJ style SOC views) ---
    out["mean_srv_error_rate"] = (
        out["serror_rate"] + out["rerror_rate"] + out["srv_serror_rate"] + out["srv_rerror_rate"]
    ) / 4.0
    out["mean_host_error_rate"] = (
        out["dst_host_serror_rate"]
        + out["dst_host_srv_serror_rate"]
        + out["dst_host_rerror_rate"]
        + out["dst_host_srv_rerror_rate"]
    ) / 4.0
    out["rej_pressure"] = out["rerror_rate"] + out["srv_rerror_rate"]
    out["syn_error_pressure"] = out["serror_rate"] + out["srv_serror_rate"]
    out["host_rej_pressure"] = out["dst_host_rerror_rate"] + out["dst_host_srv_rerror_rate"]

    # Transport-conditioned error terms highlight TCP-specific abuse paths.
    out["tcp_x_syn_error"] = out["proto_is_tcp"] * out["syn_error_pressure"]
    out["tcp_x_rej_error"] = out["proto_is_tcp"] * out["rej_pressure"]

    # --- Service / host relationship anomalies ---
    out["mixed_service_anomaly"] = out["diff_srv_rate"] * out["dst_host_diff_srv_rate"]
    out["service_host_mismatch"] = out["same_srv_rate"] / (out["srv_diff_host_rate"] + 1e-3)
    out["dst_concentration"] = (out["dst_host_same_srv_rate"] * out["dst_host_srv_count"]) / (
        out["dst_host_count"] + 1.0
    )

    # --- Privilege / post-auth compromise proxies ---
    out["sf_risk_score"] = (
        out["hot"] + out["urgent"] + out["wrong_fragment"] + out["num_failed_logins"]
    )
    out["post_auth_compromise"] = out["logged_in"] * (
        out["num_compromised"] + out["num_shells"] + out["root_shell"]
    )
    out["root_activity"] = out["num_root"] + out["root_shell"] + out["su_attempted"]
    out["file_creation_velocity"] = out["num_file_creations"] / (out["duration"] + 1.0)
    out["access_files_intensity"] = out["num_access_files"] / (out["count"] + 1.0)

    return out


def _feature_columns_after_engineering(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split engineered frame into numeric vs categorical model inputs."""
    drop_from_x = {"label", "binary_label"}
    cols = [c for c in df.columns if c not in drop_from_x]
    numeric = [
        c
        for c in cols
        if c not in _CATEGORICAL_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical = [c for c in cols if c in _CATEGORICAL_COLS]
    if len(categorical) != len(_CATEGORICAL_COLS):
        raise ValueError("Expected protocol/service/flag columns after engineering.")
    return numeric, categorical


def build_preprocessor(numeric_cols: Sequence[str], categorical_cols: Sequence[str]) -> ColumnTransformer:
    """
    Construct a ColumnTransformer with StandardScaler + OneHotEncoder.

    Scaling continuous engineered statistics prevents gradient-based models
    from being dominated by wide-dynamic-range fields (byte counts), while
    OHE preserves sparse, high-cardinality ``service`` tokens that tree models
    can split cleanly.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(numeric_cols)),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical_cols),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_preprocessing(train_df: pd.DataFrame, test_df: pd.DataFrame) -> FeatureMatrixBundle:
    """
    Fit preprocessing on **training** data only, then transform both splits.

    Returns dense numpy matrices suitable for sklearn / XGBoost trainers and
    a ``feature_names`` list aligned with columns after transformation.
    """
    train_eng = engineer_features(train_df)
    test_eng = engineer_features(test_df)

    y_train = train_eng["binary_label"].to_numpy(dtype=np.int32)
    y_test = test_eng["binary_label"].to_numpy(dtype=np.int32)

    numeric_cols, categorical_cols = _feature_columns_after_engineering(train_eng)
    X_train_df = train_eng[numeric_cols + list(categorical_cols)]
    X_test_df = test_eng[numeric_cols + list(categorical_cols)]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    try:
        feature_names: List[str] = list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - sklearn version fallback
        # Manual naming: numbered OHE columns if get_feature_names_out fails.
        feature_names = list(numeric_cols)
        feature_names += [f"cat_{i}" for i in range(X_train.shape[1] - len(numeric_cols))]

    return FeatureMatrixBundle(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        preprocessor=preprocessor,
        train_frame=train_eng,
        test_frame=test_eng,
    )


def summarize_engineered_columns(df: pd.DataFrame) -> List[str]:
    """Return engineered column names present in ``engineer_features`` output."""
    eng = engineer_features(df)
    raw_cols = set(df.columns)
    return [c for c in eng.columns if c not in raw_cols]
