"""
tests/test_data.py — Tests for data loading and temporal splitting.
"""

import numpy as np
import pandas as pd
import pytest
from src.data import make_temporal_splits, add_engineered_features


def _make_fake_df(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Time":   np.sort(rng.uniform(0, 172800, n)),  # 48h in seconds
        "Amount": rng.exponential(50, n),
        **{f"V{i}": rng.standard_normal(n) for i in range(1, 29)},
        "Class":  (rng.random(n) < 0.002).astype(int),
    })
    return df


class TestTemporalSplits:
    def test_no_overlap(self):
        df = _make_fake_df(1000)
        splits = make_temporal_splits(df)
        train_times = set(splits.X_train["Time"].tolist())
        val_times   = set(splits.X_val["Time"].tolist())
        test_times  = set(splits.X_test["Time"].tolist())
        assert len(train_times & val_times)  == 0
        assert len(train_times & test_times) == 0
        assert len(val_times   & test_times) == 0

    def test_sizes_sum_to_total(self):
        df = _make_fake_df(1000)
        splits = make_temporal_splits(df)
        total = len(splits.X_train) + len(splits.X_val) + len(splits.X_test)
        assert total == len(df)

    def test_temporal_ordering(self):
        """Train max time < val min time < test min time."""
        df = _make_fake_df(1000)
        splits = make_temporal_splits(df)
        assert splits.X_train["Time"].max() <= splits.X_val["Time"].min()
        assert splits.X_val["Time"].max()   <= splits.X_test["Time"].min()

    def test_fractions(self):
        df = _make_fake_df(10000)
        splits = make_temporal_splits(df, train_frac=0.6, val_frac=0.2)
        assert abs(len(splits.X_train) / len(df) - 0.6) < 0.01
        assert abs(len(splits.X_val)   / len(df) - 0.2) < 0.01
        assert abs(len(splits.X_test)  / len(df) - 0.2) < 0.01


class TestFeatureEngineering:
    def test_new_columns_added(self):
        df = _make_fake_df(200)
        df_eng = add_engineered_features(df)
        for col in ["Amount_log1p", "Amount_zscore", "Hour", "Hour_sin", "Hour_cos",
                    "V14_V17", "V12_V14", "V14_sq", "V_norm_l2"]:
            assert col in df_eng.columns, f"Missing: {col}"

    def test_no_nans_introduced(self):
        df = _make_fake_df(500)
        df_eng = add_engineered_features(df)
        assert df_eng.isnull().sum().sum() == 0

    def test_hour_in_range(self):
        df = _make_fake_df(500)
        df_eng = add_engineered_features(df)
        assert df_eng["Hour"].between(0, 23).all()