"""
explainability.py — SHAP-based model interpretability.

Why explainability matters in fraud detection:
  1. Regulatory: UK FCA and EU AI Act require explainability for automated
     financial decisions. A model you can't explain cannot be deployed.
  2. Operational: Analysts need to understand WHY a transaction was flagged
     to investigate effectively and build institutional knowledge.
  3. Model debugging: SHAP reveals if the model is learning spurious signals
     (e.g., large V-feature magnitudes from data preprocessing artefacts).

SHAP (SHapley Additive exPlanations) is theoretically grounded in cooperative
game theory. For tree models (LightGBM, XGBoost), TreeExplainer runs in O(TLD)
time — efficient enough for production use on millions of transactions.

Outputs produced:
  - Global: feature importance bar chart + beeswarm summary plot
  - Local:  waterfall plot for top-N flagged transactions (analyst-facing)
  - Data:   SHAP values CSV for downstream analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script use


def compute_shap_values(
    model,
    X: pd.DataFrame,
    model_type: str = "tree",   # "tree" | "linear"
    max_samples: int = 5000,    # cap for speed; TreeExplainer is O(n)
    background_samples: int = 100,
) -> tuple[shap.Explanation, pd.DataFrame]:
    """
    Compute SHAP values for a given model and dataset.

    For tree models: uses TreeExplainer (exact, fast).
    For linear models: uses LinearExplainer with a background dataset.

    Returns:
        shap_explanation: SHAP Explanation object (contains values + base_values + data)
        X_sample: the sampled X used for explanation (aligned with shap values)
    """
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42).reset_index(drop=True)
    else:
        X_sample = X.reset_index(drop=True)

    if model_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)

    elif model_type == "linear":
        # LinearExplainer needs a background distribution
        background = shap.sample(X_sample, background_samples)
        explainer = shap.LinearExplainer(model, background)
        raw_vals = explainer.shap_values(X_sample)
        # Wrap in Explanation object for consistent API
        shap_values = shap.Explanation(
            values=raw_vals,
            base_values=np.full(len(X_sample), explainer.expected_value),
            data=X_sample.values,
            feature_names=X_sample.columns.tolist(),
        )
    else:
        raise ValueError(f"model_type must be 'tree' or 'linear', got '{model_type}'")

    return shap_values, X_sample


def plot_global_importance(
    shap_values: shap.Explanation,
    out_path: str | Path,
    top_n: int = 20,
    title: str = "Global Feature Importance (SHAP)",
) -> None:
    """
    Bar chart of mean absolute SHAP values — global feature importance.
    More reliable than tree's built-in feature_importances_ because it accounts
    for direction and interaction effects.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=top_n, show=False, ax=ax)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_beeswarm(
    shap_values: shap.Explanation,
    out_path: str | Path,
    top_n: int = 20,
    title: str = "SHAP Beeswarm — Feature Impact Distribution",
) -> None:
    """
    Beeswarm plot: shows distribution of SHAP values per feature across all samples.
    Reveals not just importance but direction and non-linearity.
    Red = high feature value, Blue = low feature value.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=top_n, show=False)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_waterfall_top_flagged(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    y_score: np.ndarray,
    out_dir: str | Path,
    top_n: int = 5,
) -> None:
    """
    Waterfall plots for the top-N highest-risk transactions.
    These are the analyst-facing explanations: 'why was this flagged?'

    Each waterfall shows how each feature pushed the prediction up or down
    from the model's base rate.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get indices of top-N scored transactions (within the sample)
    top_indices = np.argsort(-y_score[:len(X_sample)])[:top_n]

    for rank, idx in enumerate(top_indices, start=1):
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[idx], show=False, max_display=15)
        plt.title(
            f"Rank #{rank} Flagged Transaction — SHAP Explanation\n"
            f"Fraud Probability: {y_score[idx]:.4f}",
            fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"waterfall_rank_{rank:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_shap_values_csv(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    out_path: str | Path,
) -> pd.DataFrame:
    """
    Export SHAP values as CSV. Useful for:
    - Downstream statistical analysis of feature contributions
    - Building analyst-facing dashboards
    - Monitoring feature drift over time (SHAP value distribution shift)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vals = shap_values.values
    if vals.ndim == 3:
        # For multi-output models, take the fraud class (index 1)
        vals = vals[:, :, 1]

    shap_df = pd.DataFrame(vals, columns=[f"shap_{c}" for c in X_sample.columns])
    shap_df["base_value"] = shap_values.base_values if shap_values.base_values.ndim == 1 else shap_values.base_values[:, 1]

    out_path.write_bytes(b"")  # ensure parent exists
    shap_df.to_csv(out_path, index=False)
    return shap_df