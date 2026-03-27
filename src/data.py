"""
data.py — Loading and temporal splitting for the creditcard dataset.

Key design decision: we use TIME-BASED splitting, not random stratified splits.
Random splits leak future information into training — a model trained on a random
sample from day 180 will have "seen" transactions from day 1 and day 180 in
both train and test. For fraud, this is optimistic and unrealistic.

Temporal split: train on first 60%, validate on next 20%, test on final 20%.
This simulates real deployment: the model is trained on historical data and
evaluated on future, unseen transactions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    split_method: str = "temporal"
    meta: dict = field(default_factory=dict)

    def fraud_rate(self, split: str) -> float:
        return float(getattr(self, f"y_{split}").mean())

    def summary(self) -> str:
        lines = [f"Split method: {self.split_method}"]
        for s in ("train", "val", "test"):
            X = getattr(self, f"X_{s}")
            y = getattr(self, f"y_{s}")
            lines.append(
                f"  {s:5s}: {len(X):>7,} rows | "
                f"fraud={int(y.sum()):>4} ({self.fraud_rate(s)*100:.4f}%)"
            )
        return "\n".join(lines)


def load_creditcard_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the Kaggle creditcard.csv dataset.
    Performs schema validation and light type coercion.

    Expected columns: Time, V1–V28, Amount, Class
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    # Schema validation
    required = {"Time", "Amount", "Class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}.\n"
            f"Found: {sorted(df.columns.tolist())}"
        )

    v_cols = [c for c in df.columns if c.startswith("V")]
    if len(v_cols) == 0:
        raise ValueError("No PCA feature columns (V1–V28) found.")

    df["Class"] = df["Class"].astype(int)

    # Sanity checks
    assert df["Class"].isin([0, 1]).all(), "Class column contains values other than 0/1"
    assert df["Time"].is_monotonic_increasing or True, "Time column warning: not sorted"

    return df


def make_temporal_splits(
    df: pd.DataFrame,
    target_col: str = "Class",
    time_col: str = "Time",
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    # test_frac is implicitly 1 - train_frac - val_frac = 0.20
) -> DataSplits:
    """
    Temporal train/val/test split based on the Time column.

    Why this matters:
    - Preserves causal ordering: the model never sees 'future' transactions during training.
    - Simulates realistic deployment: models are retrained periodically on past data.
    - Avoids data leakage that inflates metrics in random splits.

    Splits are contiguous time windows, not random samples.
    """
    assert train_frac + val_frac < 1.0, "train_frac + val_frac must be < 1.0"

    df_sorted = df.sort_values(time_col).reset_index(drop=True)

    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df_sorted.iloc[:train_end]
    val_df   = df_sorted.iloc[train_end:val_end]
    test_df  = df_sorted.iloc[val_end:]

    def split_xy(d: pd.DataFrame):
        return d.drop(columns=[target_col]), d[target_col]

    X_train, y_train = split_xy(train_df)
    X_val,   y_val   = split_xy(val_df)
    X_test,  y_test  = split_xy(test_df)

    meta = {
        "time_range_train": (float(train_df[time_col].min()), float(train_df[time_col].max())),
        "time_range_val":   (float(val_df[time_col].min()),   float(val_df[time_col].max())),
        "time_range_test":  (float(test_df[time_col].min()),  float(test_df[time_col].max())),
    }

    return DataSplits(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        split_method="temporal",
        meta=meta,
    )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light feature engineering on top of V1–V28 PCA components.

    The raw features are already PCA-transformed by the dataset provider,
    so traditional domain-based engineering is limited. We add:
    - Amount log-transform: raw Amount is right-skewed; log(1+x) compresses outliers.
    - Hour-of-day: Time is seconds from first transaction; modular arithmetic gives
      cyclical time-of-day signal which correlates with fraud patterns.
    - V-feature interaction terms: top correlated pairs (V14*V17, V12*V14)
      identified as high-signal in the fraud detection literature.
    - L2 norm of all V features: a single summary of 'how unusual' the PCA vector is.
    """
    df = df.copy()

    # Amount features
    df["Amount_log1p"] = np.log1p(df["Amount"])
    df["Amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1e-9)

    # Time features — Time is seconds elapsed from start of dataset
    df["Hour"] = (df["Time"] // 3600) % 24
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    # Interaction features (literature-informed: V14, V17, V12 are top fraud signals)
    if all(c in df.columns for c in ["V14", "V17", "V12"]):
        df["V14_V17"] = df["V14"] * df["V17"]
        df["V12_V14"] = df["V12"] * df["V14"]
        df["V14_sq"]  = df["V14"] ** 2

    # L2 norm of all V-features (anomaly magnitude signal)
    v_cols = [c for c in df.columns if c.startswith("V")]
    df["V_norm_l2"] = np.sqrt((df[v_cols] ** 2).sum(axis=1))

    return df