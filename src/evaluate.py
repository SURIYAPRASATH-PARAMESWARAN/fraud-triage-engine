"""
evaluate.py — Evaluation suite for fraud triage systems.

Metrics implemented:
  - ROC-AUC            : overall discrimination ability
  - PR-AUC             : precision-recall tradeoff (correct metric for imbalanced data)
  - Precision@K        : among top-K flagged, how many are real fraud?
  - Recall@K           : of all fraud, how much does top-K capture?
  - F1@K               : harmonic mean of P@K and R@K
  - Expected Cost@K    : business cost of the triage policy (FN + FP costs)
  - Calibration Error  : Expected Calibration Error (ECE) and reliability curve data
  - Lift@K             : how much better than random flagging?

The central thesis: fraud detection should be evaluated as a RANKING problem
under capacity constraints, not as a binary classifier at a fixed threshold.
All K-based metrics operationalise this directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


@dataclass
class TriageMetrics:
    """K-based metrics for a single review capacity K."""
    k: int
    precision: float
    recall: float
    f1: float
    lift: float                 # precision@k / base_rate
    expected_cost: float        # cost in £ given fn_cost and fp_cost
    frauds_caught: int
    frauds_missed: int
    false_alerts: int


@dataclass
class EvalReport:
    model_name: str
    roc_auc: float
    pr_auc: float
    brier_score: float          # lower is better; measures calibration quality
    ece: float                  # Expected Calibration Error
    triage: Dict[int, TriageMetrics]
    calibration_data: Tuple[np.ndarray, np.ndarray]   # (fraction_positive, mean_predicted)
    base_rate: float


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error: weighted average gap between predicted
    probability and empirical accuracy across confidence bins.

    A perfectly calibrated model has ECE=0. High ECE means the model's
    stated probabilities don't match reality — dangerous for threshold setting.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


def triage_metrics_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int,
    fn_cost: float = 500.0,   # £ cost of missing one fraud (chargeback + ops)
    fp_cost: float = 10.0,    # £ cost of one false alert (analyst review time)
) -> TriageMetrics:
    """
    Compute precision, recall, F1, lift, and expected business cost at review capacity K.

    fn_cost / fp_cost defaults reflect approximate real-world fraud operations:
    - Missing a fraud: ~£500 (average credit card fraud loss + chargeback fees)
    - False positive: ~£10 (analyst time: ~6 min at £100/hr fully loaded)

    These defaults can be overridden to match any organisation's cost structure.
    """
    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    k = min(k, len(y_sorted))
    topk = y_sorted[:k]
    rest = y_sorted[k:]

    total_frauds = int(y_true.sum())
    base_rate = total_frauds / len(y_true)

    caught = int(topk.sum())
    missed = int(rest.sum())
    false_alerts = int(k - caught)

    precision = caught / k if k > 0 else 0.0
    recall = caught / total_frauds if total_frauds > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    lift = precision / base_rate if base_rate > 0 else 0.0
    expected_cost = (missed * fn_cost) + (false_alerts * fp_cost)

    return TriageMetrics(
        k=k,
        precision=precision,
        recall=recall,
        f1=f1,
        lift=lift,
        expected_cost=expected_cost,
        frauds_caught=caught,
        frauds_missed=missed,
        false_alerts=false_alerts,
    )


def evaluate(
    model_name: str,
    y_true: pd.Series,
    y_score: np.ndarray,
    ks: Tuple[int, ...] = (100, 250, 500, 750, 1000, 2000),
    fn_cost: float = 500.0,
    fp_cost: float = 10.0,
    n_cal_bins: int = 10,
) -> EvalReport:
    """
    Full evaluation of a model's risk scores.
    Returns a structured EvalReport for comparison across models.
    """
    y_arr = y_true.to_numpy().astype(int)

    roc = float(roc_auc_score(y_arr, y_score))
    pr  = float(average_precision_score(y_arr, y_score))
    brier = float(brier_score_loss(y_arr, y_score))
    ece = compute_ece(y_arr, y_score, n_bins=n_cal_bins)

    # Calibration curve data for plotting
    fraction_pos, mean_pred = calibration_curve(y_arr, y_score, n_bins=n_cal_bins, strategy="quantile")

    triage: Dict[int, TriageMetrics] = {}
    for k in ks:
        triage[k] = triage_metrics_at_k(y_arr, y_score, k=k, fn_cost=fn_cost, fp_cost=fp_cost)

    return EvalReport(
        model_name=model_name,
        roc_auc=roc,
        pr_auc=pr,
        brier_score=brier,
        ece=ece,
        triage=triage,
        calibration_data=(fraction_pos, mean_pred),
        base_rate=float(y_arr.mean()),
    )


def optimal_k_by_cost(report: EvalReport) -> Tuple[int, TriageMetrics]:
    """
    Find the K that minimises expected business cost (FN cost + FP cost).
    This is the operationally optimal review capacity given the cost structure.
    """
    best_k = min(report.triage.items(), key=lambda x: x[1].expected_cost)
    return best_k[0], best_k[1]


def compare_models(reports: list[EvalReport]) -> pd.DataFrame:
    """
    Summary comparison table across all evaluated models.
    Includes both ranking metrics and calibration quality.
    """
    rows = []
    for r in reports:
        best_k, best_metrics = optimal_k_by_cost(r)
        rows.append({
            "model":            r.model_name,
            "roc_auc":          round(r.roc_auc, 4),
            "pr_auc":           round(r.pr_auc, 4),
            "brier_score":      round(r.brier_score, 6),
            "ece":              round(r.ece, 6),
            "optimal_k":        best_k,
            "cost_at_optimal_k": round(best_metrics.expected_cost, 0),
            "recall@500":       round(r.triage.get(500, TriageMetrics(500,0,0,0,0,0,0,0,0)).recall, 4),
            "precision@500":    round(r.triage.get(500, TriageMetrics(500,0,0,0,0,0,0,0,0)).precision, 4),
            "lift@500":         round(r.triage.get(500, TriageMetrics(500,0,0,0,0,0,0,0,0)).lift, 2),
        })
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False)


def save_triage_queue(
    X: pd.DataFrame,
    y_true: pd.Series,
    y_score: np.ndarray,
    k: int,
    out_path: str | Path,
    model_name: str = "model",
) -> pd.DataFrame:
    """
    Produce the analyst review queue: top-K transactions ranked by fraud probability.
    Includes risk tier labels (HIGH / MEDIUM / LOW) and a SHAP-ready index column.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = X.copy()
    df["y_true"]     = y_true.to_numpy()
    df["fraud_prob"] = y_score
    df["rank"]       = df["fraud_prob"].rank(ascending=False, method="first").astype(int)
    df["flagged"]    = (df["rank"] <= k).astype(int)

    # Risk tier: allows analysts to prioritise within the queue
    df["risk_tier"] = pd.cut(
        df["fraud_prob"],
        bins=[0.0, 0.3, 0.6, 0.85, 1.0],
        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        include_lowest=True,
    )

    df = df.sort_values("rank").reset_index(drop=True)
    top_k = df[df["flagged"] == 1].copy()
    top_k.to_csv(out_path, index=False)

    return top_k