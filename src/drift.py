"""
drift.py — Temporal model performance monitoring and concept drift simulation.

What is concept drift in fraud?
  Fraudsters adapt. A model trained on January data sees different patterns
  by March. Without drift monitoring, a deployed model silently degrades
  while the team assumes it's still working.

This module:
  1. Splits the test set into temporal windows (simulating weekly/monthly batches)
  2. Evaluates model performance on each window independently
  3. Produces a drift curve: PR-AUC over time
  4. Implements a simple retraining trigger: alert when PR-AUC drops by >threshold

This is a simulation of what a real MLOps monitoring pipeline would do.
In production, this logic runs continuously on live scoring outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass
class DriftWindow:
    window_id: int
    time_start: float
    time_end: float
    n_transactions: int
    n_fraud: int
    fraud_rate: float
    pr_auc: float
    roc_auc: float
    alert: bool = False   # True if retraining should be triggered


def simulate_drift(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_score: np.ndarray,
    n_windows: int = 5,
    alert_threshold: float = 0.05,   # alert if PR-AUC drops by this much from peak
    time_col: str = "Time",
) -> List[DriftWindow]:
    """
    Simulate temporal drift detection by evaluating performance on
    successive time windows within the test set.

    Parameters:
        n_windows:        how many time buckets to split the test set into
        alert_threshold:  relative PR-AUC drop that triggers a retraining alert

    Returns:
        List of DriftWindow objects, one per time window
    """
    df = X_test.copy()
    df["__y"]      = y_test.to_numpy()
    df["__score"]  = y_score

    if time_col in df.columns:
        df = df.sort_values(time_col)
    else:
        df = df.reset_index(drop=True)
        df["__time_proxy"] = df.index
        time_col = "__time_proxy"

    window_size = len(df) // n_windows
    windows: List[DriftWindow] = []
    peak_pr_auc = None

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx   = (i + 1) * window_size if i < n_windows - 1 else len(df)
        chunk = df.iloc[start_idx:end_idx]

        y_w = chunk["__y"].to_numpy()
        s_w = chunk["__score"].to_numpy()

        # Need at least some fraud cases to compute PR-AUC
        if y_w.sum() < 2:
            continue

        pr  = float(average_precision_score(y_w, s_w))
        roc = float(roc_auc_score(y_w, s_w))

        if peak_pr_auc is None:
            peak_pr_auc = pr

        alert = (peak_pr_auc - pr) >= alert_threshold

        windows.append(DriftWindow(
            window_id=i,
            time_start=float(chunk[time_col].min()),
            time_end=float(chunk[time_col].max()),
            n_transactions=len(chunk),
            n_fraud=int(y_w.sum()),
            fraud_rate=float(y_w.mean()),
            pr_auc=pr,
            roc_auc=roc,
            alert=alert,
        ))

    return windows


def drift_summary_df(windows: List[DriftWindow]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "window":          w.window_id,
            "time_start":      w.time_start,
            "time_end":        w.time_end,
            "n_transactions":  w.n_transactions,
            "n_fraud":         w.n_fraud,
            "fraud_rate_%":    round(w.fraud_rate * 100, 4),
            "pr_auc":          round(w.pr_auc, 4),
            "roc_auc":         round(w.roc_auc, 4),
            "retrain_alert":   w.alert,
        }
        for w in windows
    ])