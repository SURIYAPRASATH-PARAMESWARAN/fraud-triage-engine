"""
viz.py — Visualisation suite for the fraud triage engine.

Plots produced:
  1. fraud_capture_curve     — Recall@K vs review capacity (core project plot)
  2. precision_recall_curve  — Full PR curve comparison across models
  3. calibration_curve       — Reliability diagram (model vs perfect calibration)
  4. model_comparison_bar    — Side-by-side ROC-AUC / PR-AUC bar chart
  5. drift_curve             — PR-AUC degradation over time windows
  6. cost_curve              — Expected business cost vs review capacity K

All plots use a consistent, publication-quality style.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import precision_recall_curve

# ── Style constants ────────────────────────────────────────────────────────────
PALETTE = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED"]
STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8FAFC",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "grid.linestyle":   "--",
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right": False,
}
plt.rcParams.update(STYLE)


def _save(fig, path: Path, dpi: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── 1. Fraud Capture Curve ────────────────────────────────────────────────────

def plot_fraud_capture_curve(
    score_sets: Dict[str, Tuple[pd.Series, np.ndarray]],
    out_path: str | Path,
    ks: Tuple[int, ...] = (50, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 5000),
    policy_k: Optional[int] = 500,
    title: str = "Fraud Capture Curve — Recall@K vs Review Capacity",
) -> pd.DataFrame:
    """
    Plot recall@K for multiple models on the same axes.
    Also marks the operational policy line (K=500 by default).

    score_sets: dict of {model_name: (y_true, y_score)}
    Returns a DataFrame of the curve data for all models.
    """
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(10, 6))

    all_rows = []

    for i, (name, (y_true, y_score)) in enumerate(score_sets.items()):
        y_arr = y_true.to_numpy()
        order = np.argsort(-y_score)
        y_sorted = y_arr[order]
        total_frauds = y_sorted.sum()

        recalls = []
        for k in ks:
            k_clip = min(int(k), len(y_sorted))
            recall = float(y_sorted[:k_clip].sum() / total_frauds)
            recalls.append(recall)
            all_rows.append({"model": name, "k": k_clip, "recall": recall})

        ax.plot(list(ks), recalls, marker="o", markersize=4, linewidth=2,
                label=name, color=PALETTE[i % len(PALETTE)])

    # Policy line
    if policy_k is not None:
        ax.axvline(x=policy_k, color="black", linestyle="--", alpha=0.6, linewidth=1.2)
        ax.text(policy_k + 30, 0.05, f"Policy K={policy_k}", fontsize=9, color="black")

    # Random baseline
    if all_rows:
        any_k = [r["k"] for r in all_rows if r["model"] == list(score_sets.keys())[0]]
        total = len(list(score_sets.values())[0][0])
        fraud_total = int(list(score_sets.values())[0][0].sum())
        random_recalls = [min(k / total * fraud_total / fraud_total, 1.0) * (k / total) for k in any_k]
        ax.plot(any_k, [k / total for k in any_k], linestyle=":", color="grey",
                label="Random baseline", linewidth=1.2)

    ax.set_xlabel("Review Capacity (Top-K transactions reviewed)", fontsize=11)
    ax.set_ylabel("Recall (Fraction of Fraud Captured)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    _save(fig, out_path)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    return df


# ── 2. PR Curve Comparison ────────────────────────────────────────────────────

def plot_pr_curves(
    score_sets: Dict[str, Tuple[pd.Series, np.ndarray]],
    out_path: str | Path,
    title: str = "Precision-Recall Curves — Model Comparison",
) -> None:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(8, 6))

    from sklearn.metrics import average_precision_score

    for i, (name, (y_true, y_score)) in enumerate(score_sets.items()):
        y_arr = y_true.to_numpy()
        prec, rec, _ = precision_recall_curve(y_arr, y_score)
        auprc = average_precision_score(y_arr, y_score)
        ax.plot(rec, prec, linewidth=2, label=f"{name} (PR-AUC={auprc:.4f})",
                color=PALETTE[i % len(PALETTE)])

    base_rate = float(y_arr.mean())
    ax.axhline(y=base_rate, linestyle=":", color="grey", label=f"Random baseline ({base_rate:.4f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    _save(fig, out_path)


# ── 3. Calibration (Reliability) Diagram ─────────────────────────────────────

def plot_calibration_curves(
    calibration_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_path: str | Path,
    title: str = "Calibration Curves (Reliability Diagram)",
) -> None:
    """
    Reliability diagram: predicted probability vs observed fraud rate per bin.
    Perfect calibration = diagonal line. Deviation shows over/under-confidence.

    calibration_data: {model_name: (fraction_positive, mean_predicted)}
    """
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 7))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect calibration", linewidth=1.5)

    for i, (name, (frac_pos, mean_pred)) in enumerate(calibration_data.items()):
        ax.plot(mean_pred, frac_pos, marker="s", linewidth=2, markersize=5,
                label=name, color=PALETTE[i % len(PALETTE)])

    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Observed Fraud Rate (Fraction Positive)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    _save(fig, out_path)


# ── 4. Model Comparison Bar Chart ─────────────────────────────────────────────

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    out_path: str | Path,
    metrics: Tuple[str, ...] = ("roc_auc", "pr_auc", "recall@500", "precision@500"),
    title: str = "Model Comparison — Key Metrics",
) -> None:
    out_path = Path(out_path)

    # Filter to metrics that exist in the df
    metrics = [m for m in metrics if m in comparison_df.columns]
    n_metrics = len(metrics)
    n_models  = len(comparison_df)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        vals = comparison_df[metric].values
        bars = ax.bar(comparison_df["model"].values, vals,
                      color=PALETTE[:n_models], edgecolor="white", linewidth=0.8)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(metric.upper().replace("_", " "), fontsize=10, fontweight="bold")
        ax.set_ylim(0, min(1.1, max(vals) * 1.2))
        ax.tick_params(axis="x", rotation=20, labelsize=8)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, out_path)


# ── 5. Drift Curve ────────────────────────────────────────────────────────────

def plot_drift_curve(
    drift_df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Model Performance Over Time (Concept Drift Simulation)",
) -> None:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(10, 5))

    x = range(len(drift_df))
    ax.plot(x, drift_df["pr_auc"], marker="o", linewidth=2,
            color=PALETTE[0], label="PR-AUC")
    ax.plot(x, drift_df["roc_auc"], marker="s", linewidth=2,
            color=PALETTE[2], label="ROC-AUC", linestyle="--")

    # Highlight alert windows
    alerts = drift_df[drift_df["retrain_alert"] == True]
    if len(alerts) > 0:
        alert_indices = [drift_df.index.get_loc(i) for i in alerts.index]
        ax.scatter(alert_indices, alerts["pr_auc"],
                   color="red", zorder=5, s=100, label="⚠ Retrain Alert", marker="^")

    ax.set_xlabel("Time Window (Chronological)", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"W{i}" for i in x])

    _save(fig, out_path)


# ── 6. Cost Curve ─────────────────────────────────────────────────────────────

def plot_cost_curve(
    score_sets: Dict[str, Tuple[pd.Series, np.ndarray]],
    out_path: str | Path,
    fn_cost: float = 500.0,
    fp_cost: float = 10.0,
    ks: Tuple[int, ...] = (50, 100, 250, 500, 750, 1000, 1500, 2000, 3000),
    policy_k: Optional[int] = 500,
    title: str = "Expected Business Cost vs Review Capacity",
) -> None:
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, (y_true, y_score)) in enumerate(score_sets.items()):
        y_arr = y_true.to_numpy()
        order = np.argsort(-y_score)
        y_sorted = y_arr[order]
        total_frauds = int(y_sorted.sum())

        costs = []
        for k in ks:
            k_clip = min(int(k), len(y_sorted))
            caught = int(y_sorted[:k_clip].sum())
            missed = total_frauds - caught
            false_alerts = k_clip - caught
            cost = (missed * fn_cost) + (false_alerts * fp_cost)
            costs.append(cost)

        ax.plot(list(ks), costs, marker="o", markersize=4, linewidth=2,
                label=name, color=PALETTE[i % len(PALETTE)])

    if policy_k is not None:
        ax.axvline(x=policy_k, color="black", linestyle="--", alpha=0.6, linewidth=1.2)
        ax.text(policy_k + 30, ax.get_ylim()[0] + 1000, f"Policy K={policy_k}",
                fontsize=9, color="black")

    ax.set_xlabel("Review Capacity (Top-K)", fontsize=11)
    ax.set_ylabel(f"Expected Cost (£)\n[FN cost=£{fn_cost:.0f}, FP cost=£{fp_cost:.0f}]", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.legend(fontsize=9)

    _save(fig, out_path)