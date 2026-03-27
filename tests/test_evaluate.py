"""
tests/test_evaluate.py — Unit tests for the evaluation module.

Run: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from src.evaluate import triage_metrics_at_k, compute_ece, evaluate


def _make_data(n: int = 1000, fraud_rate: float = 0.02, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    # Scores: fraud cases get higher scores on average
    scores = rng.random(n)
    scores[y == 1] = rng.random(y.sum()) * 0.4 + 0.6  # fraud: 0.6–1.0
    return pd.Series(y), scores


class TestTriageMetrics:
    def test_precision_one(self):
        y = pd.Series([1, 1, 1, 0, 0])
        s = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        m = triage_metrics_at_k(y.to_numpy(), s, k=3)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.frauds_caught == 3
        assert m.frauds_missed == 0

    def test_recall_zero_at_small_k(self):
        y = pd.Series([1, 1, 1, 0, 0])
        s = np.array([0.2, 0.1, 0.05, 0.9, 0.8])   # model ranks non-fraud at top
        m = triage_metrics_at_k(y.to_numpy(), s, k=2)
        assert m.precision == 0.0
        assert m.recall == 0.0

    def test_lift_above_one_for_good_model(self):
        y, s = _make_data(n=2000, fraud_rate=0.05)
        m = triage_metrics_at_k(y.to_numpy(), s, k=100)
        assert m.lift > 1.0, "Good model should have lift > 1 at K=100"

    def test_k_clipped_to_dataset_length(self):
        y = pd.Series([1, 0, 1])
        s = np.array([0.9, 0.5, 0.3])
        m = triage_metrics_at_k(y.to_numpy(), s, k=10_000)
        assert m.k == 3   # clipped to dataset size

    def test_expected_cost_all_caught(self):
        # If all fraud is caught and no false positives, cost = 0
        y = pd.Series([1, 1, 0, 0, 0])
        s = np.array([0.9, 0.8, 0.1, 0.05, 0.02])
        m = triage_metrics_at_k(y.to_numpy(), s, k=2, fn_cost=500, fp_cost=10)
        assert m.frauds_caught == 2
        assert m.frauds_missed == 0
        assert m.false_alerts  == 0
        assert m.expected_cost == pytest.approx(0.0)


class TestECE:
    def test_perfect_calibration(self):
        # Predicted probs exactly equal observed rates → ECE ≈ 0
        # Hard to construct exactly, so just test it's low for a sane signal
        rng = np.random.default_rng(1)
        probs = rng.uniform(0, 1, 1000)
        y = (rng.random(1000) < probs).astype(int)
        ece = compute_ece(y, probs, n_bins=10)
        assert ece < 0.1   # should be well-calibrated on average

    def test_overconfident_model(self):
        # Model predicts 0.9 everywhere but only 50% are fraud → high ECE
        y = np.array([1, 0] * 500)
        p = np.full(1000, 0.9)
        ece = compute_ece(y, p)
        assert ece > 0.3


class TestEvaluate:
    def test_evaluate_returns_report(self):
        y, s = _make_data(n=5000, fraud_rate=0.02)
        report = evaluate("test_model", y, s, ks=(100, 500))
        assert report.roc_auc > 0.5
        assert report.pr_auc > 0.0
        assert 100 in report.triage
        assert 500 in report.triage

    def test_evaluate_roc_near_1_for_perfect_ranker(self):
        rng = np.random.default_rng(42)
        n = 2000
        y = pd.Series((rng.random(n) < 0.05).astype(int))
        # Perfect ranker: fraud always scored 1, non-fraud always 0
        s = y.to_numpy().astype(float)
        report = evaluate("perfect", y, s, ks=(100,))
        assert report.roc_auc == pytest.approx(1.0, abs=1e-6)