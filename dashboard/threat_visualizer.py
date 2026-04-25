"""
Phase 4 — SOC-style triage dashboard (static Matplotlib/Seaborn figure).

Panels are designed for **alert review**: separation of benign vs malicious
structure (embedding), ranked IoCs, confusion heatmap with FP/TP language,
and byte-ratio distributions across NSL attack families.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.data_loader import coarse_attack_family
from src.feature_engine import FeatureMatrixBundle


def _sample_matrix(
    X: np.ndarray,
    y: np.ndarray,
    max_points: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if X.shape[0] <= max_points:
        return X, y
    rng = np.random.default_rng(random_state)
    idx = rng.choice(X.shape[0], size=max_points, replace=False)
    return X[idx], y[idx]


def _embedding_2d(X: np.ndarray, method: str, random_state: int) -> np.ndarray:
    """2-D projection for triage scatter; robust to tiny smoke-test matrices."""
    if X.shape[0] < 10 or X.shape[1] < 2:
        rng = np.random.default_rng(random_state)
        return rng.normal(size=(X.shape[0], 2), scale=0.05)
    if method == "tsne" and X.shape[0] > 4000:
        # t-SNE is expensive at NSL scale; fall back to PCA for the dashboard build.
        method = "pca"
    if method == "tsne":
        perplexity = min(30, max(5, X.shape[0] // 4))
        ts = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=random_state,
        )
        return np.asarray(ts.fit_transform(X))
    n_comp = min(2, X.shape[1], max(1, X.shape[0] - 1))
    pca = PCA(n_components=n_comp, random_state=random_state)
    z = np.asarray(pca.fit_transform(X))
    if z.shape[1] == 1:
        z = np.column_stack([z, np.zeros(z.shape[0])])
    return z[:, :2]


def build_soc_dashboard(
    bundle: FeatureMatrixBundle,
    models: Mapping[str, Any],
    *,
    output_path: Path,
    embedding_method: str = "pca",
    sample_points: int = 8000,
    random_state: int = 42,
    primary_model_key: str = "xgboost",
) -> Path:
    """
    Render a multi-panel SOC dashboard to ``output_path`` (PNG recommended).

    Parameters
    ----------
    primary_model_key
        Which estimator's confusion matrix to highlight (default: XGBoost).
    """
    sns.set_theme(style="whitegrid", context="talk")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    X_vis, y_vis = _sample_matrix(bundle.X_test, bundle.y_test, sample_points, random_state)
    emb = _embedding_2d(X_vis, embedding_method, random_state)

    fig = plt.figure(figsize=(22, 16), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # --- Panel A: embedding ---
    ax_a = fig.add_subplot(gs[0, 0])
    palette = {0: "#2ecc71", 1: "#e74c3c"}
    for cls in (0, 1):
        mask = y_vis == cls
        label = "Benign (normal)" if cls == 0 else "Malicious (attack)"
        ax_a.scatter(
            emb[mask, 0],
            emb[mask, 1],
            s=10,
            alpha=0.35,
            label=label,
            color=palette[cls],
            linewidths=0,
        )
    ax_a.set_title(f"Attack pattern view ({embedding_method.upper()} on engineered features)")
    ax_a.set_xlabel("Component 1")
    ax_a.set_ylabel("Component 2")
    ax_a.legend(frameon=True)

    # --- Panel B: blended feature importance (top 10 IoCs) ---
    ax_b = fig.add_subplot(gs[0, 1])
    rf = models.get("random_forest")
    xgb_model = models.get("xgboost")
    names = np.array(bundle.feature_names)
    if rf is not None and hasattr(rf, "feature_importances_") and xgb_model is not None and hasattr(
        xgb_model, "feature_importances_"
    ):
        blend = (rf.feature_importances_ + xgb_model.feature_importances_) / 2.0
        top_idx = np.argsort(-blend)[:10]
        sns.barplot(
            x=blend[top_idx],
            y=names[top_idx],
            ax=ax_b,
            color="#3498db",
        )
        ax_b.set_title("Top 10 IoC-style features (mean RF + XGB importance)")
        ax_b.set_xlabel("Importance")
    else:
        ax_b.text(0.5, 0.5, "Tree importances unavailable", ha="center", va="center")
        ax_b.axis("off")

    # --- Panel C: confusion matrix heatmap ---
    ax_c = fig.add_subplot(gs[1, 0])
    primary = models.get(primary_model_key) or models.get("random_forest")
    if primary is None:
        ax_c.text(0.5, 0.5, "Primary model missing", ha="center", va="center")
        ax_c.axis("off")
    else:
        y_hat = primary.predict(bundle.X_test).astype(int)
        cm = confusion_matrix(bundle.y_test, y_hat, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Attack"])
        disp.plot(ax=ax_c, cmap="Blues", colorbar=False, values_format="d")
        ax_c.set_title(f"Confusion matrix — {primary_model_key} (test set)")
        tn, fp, fn, tp = cm.ravel()
        ax_c.text(
            0.5,
            -0.12,
            f"TN={tn} (benign OK) | FP={fp} (false alarms) | FN={fn} (missed) | TP={tp} (caught)",
            transform=ax_c.transAxes,
            ha="center",
            fontsize=12,
        )

    # --- Panel D: byte ratio distribution by attack family ---
    ax_d = fig.add_subplot(gs[1, 1])
    frame = bundle.test_frame.copy()
    frame["family"] = frame["label"].map(coarse_attack_family)
    plot_df = frame[frame["family"].isin(["Normal", "DoS", "Probe", "R2L", "U2R"])].copy()
    if plot_df.empty:
        ax_d.text(0.5, 0.5, "Insufficient labeled rows for family plot", ha="center", va="center")
        ax_d.axis("off")
    else:
        # Violin KDE can crash on some constrained SciPy/macOS runtimes.
        # Boxplot keeps the same analyst signal while avoiding that native path.
        sns.boxplot(
            data=plot_df,
            x="family",
            y="byte_ratio_src_to_dst",
            hue="family",
            ax=ax_d,
            order=["Normal", "DoS", "Probe", "R2L", "U2R"],
            palette="muted",
            dodge=False,
        )
        if ax_d.get_legend() is not None:
            ax_d.get_legend().remove()
        ax_d.set_title("Byte ratio (src→dst) by NSL attack family")
        ax_d.set_xlabel("Attack family")
        ax_d.set_ylabel("src_bytes / (dst_bytes + 1)")

    fig.suptitle("SOC Alert Triage — NSL-KDD NIDS (engineered connection metadata)", fontsize=20)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def render_dashboard_from_disk(
    bundle_path: Path,
    models_dir: Path,
    output_path: Path,
    *,
    embedding_method: str = "pca",
) -> Path:
    """Convenience loader for ``main.py`` orchestration."""
    bundle: FeatureMatrixBundle = joblib.load(bundle_path)
    models = {
        "baseline_logreg": joblib.load(models_dir / "baseline_logreg.joblib"),
        "random_forest": joblib.load(models_dir / "random_forest.joblib"),
        "xgboost": joblib.load(models_dir / "xgboost.joblib"),
    }
    return build_soc_dashboard(
        bundle,
        models,
        output_path=output_path,
        embedding_method=embedding_method,
    )
