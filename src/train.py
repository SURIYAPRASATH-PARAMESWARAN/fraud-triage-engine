"""
train.py — Model training suite.

Architecture:
  1. LogisticRegression   — baseline (your original, kept for comparison)
  2. LightGBM             — primary model; handles imbalance natively via scale_pos_weight
  3. XGBoost              — secondary gradient boosted model for ensemble/comparison
  4. CalibratedModel      — wraps any model with Platt/isotonic calibration

Why calibration matters for fraud triage:
  Raw model outputs are scores, not probabilities. On imbalanced data
  (0.17% fraud), uncalibrated scores tend to be compressed near 0.
  Calibration aligns predicted P(fraud) with empirical fraud rate,
  which is critical for threshold-setting and cost-based decision making.
  A reliability diagram (calibration curve) validates this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb


@dataclass
class TrainedModel:
    name: str
    model: Any          # sklearn-compatible: has predict_proba
    meta: Dict[str, Any] = field(default_factory=dict)
    is_calibrated: bool = False

    def predict_proba_fraud(self, X: pd.DataFrame) -> np.ndarray:
        """Returns P(fraud) for each row. Consistent interface across all model types."""
        return self.model.predict_proba(X)[:, 1]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Logistic Regression Baseline
# ──────────────────────────────────────────────────────────────────────────────

def train_logistic_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> TrainedModel:
    """
    Logistic Regression with balanced class weights and StandardScaler.
    Kept identical to original for clean comparison against new models.
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
            max_iter=500,
        )),
    ])
    pipe.fit(X_train, y_train)

    return TrainedModel(
        name="logreg_baseline",
        model=pipe,
        meta={"solver": "liblinear", "class_weight": "balanced"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. LightGBM (Primary Model)
# ──────────────────────────────────────────────────────────────────────────────

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
    n_estimators: int = 1000,
    early_stopping_rounds: int = 50,
) -> TrainedModel:
    """
    LightGBM with:
    - scale_pos_weight: handles severe class imbalance by upweighting fraud class
    - Early stopping on validation PR-AUC (not accuracy — accuracy is useless here)
    - Dart boosting type for regularisation on small positive class

    scale_pos_weight = n_negatives / n_positives
    This tells LightGBM that each fraud sample is worth ~580x a normal transaction.
    """
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos

    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    return TrainedModel(
        name="lightgbm",
        model=clf,
        meta={
            "scale_pos_weight": scale_pos_weight,
            "best_iteration": clf.best_iteration_,
            "n_features": X_train.shape[1],
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. XGBoost
# ──────────────────────────────────────────────────────────────────────────────

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = 42,
    n_estimators: int = 1000,
    early_stopping_rounds: int = 50,
) -> TrainedModel:
    """
    XGBoost with scale_pos_weight and early stopping on AUCPR.
    Used as secondary model for ensemble and head-to-head comparison.
    """
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="aucpr",
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    return TrainedModel(
        name="xgboost",
        model=clf,
        meta={
            "scale_pos_weight": scale_pos_weight,
            "best_iteration": clf.best_iteration,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4. Probability Calibration
# ──────────────────────────────────────────────────────────────────────────────

def _freeze_xgboost(model: Any) -> Any:
    """
    CalibratedClassifierCV refits the estimator on CV folds internally.
    XGBoost with early_stopping_rounds requires eval_set at fit() time,
    which sklearn's calibration wrapper never provides — causing a crash.

    Fix: return a fresh XGBClassifier with early_stopping_rounds=None
    and n_estimators fixed to the best iteration from the original training run.
    All other hyperparameters are preserved exactly.
    """
    if isinstance(model, xgb.XGBClassifier) and model.early_stopping_rounds is not None:
        best_n = getattr(model, "best_iteration", None)
        n_est  = (best_n + 1) if best_n is not None else model.n_estimators
        params = model.get_params()
        params["early_stopping_rounds"] = None
        params["n_estimators"] = n_est
        return xgb.XGBClassifier(**params)
    return model


def calibrate_model(
    trained: TrainedModel,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    method: str = "isotonic",
) -> TrainedModel:
    """
    Post-hoc probability calibration using held-out validation data.

    Why:
    - Tree models produce well-ranked but poorly calibrated probabilities
      on imbalanced data. Calibration aligns P(fraud) with empirical rates,
      essential for cost-based threshold setting.

    Method:
    - isotonic: non-parametric, better for larger val sets
    - sigmoid:  Platt scaling, more stable for small positive class counts

    Compatibility:
    - sklearn >= 1.6 removed cv='prefit'; cv=None is the replacement.
    - XGBoost early_stopping_rounds is stripped before wrapping (see _freeze_xgboost).
    """
    import sklearn, re
    sk_version = tuple(int(x) for x in re.findall(r"\d+", sklearn.__version__)[:2])

    base_model = _freeze_xgboost(trained.model)

    if sk_version >= (1, 6):
        calibrated_clf = CalibratedClassifierCV(
            estimator=base_model,
            method=method,
            cv=None,
        )
    else:
        calibrated_clf = CalibratedClassifierCV(
            estimator=base_model,
            method=method,
            cv="prefit",
        )

    calibrated_clf.fit(X_val, y_val)

    return TrainedModel(
        name=f"{trained.name}_calibrated_{method}",
        model=calibrated_clf,
        meta={**trained.meta, "calibration_method": method},
        is_calibrated=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5. Rank Ensemble (average of calibrated scores)
# ──────────────────────────────────────────────────────────────────────────────

def ensemble_scores(
    models: list[TrainedModel],
    X: pd.DataFrame,
    weights: Optional[list[float]] = None,
) -> np.ndarray:
    """
    Weighted average of calibrated fraud probabilities from multiple models.
    Default: equal weights.

    Using calibrated scores for ensembling is important — averaging raw
    uncalibrated scores from different models is misleading because each
    model's score scale may differ significantly.
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    assert abs(sum(weights) - 1.0) < 1e-6, "Weights must sum to 1.0"

    scores = np.zeros(len(X))
    for m, w in zip(models, weights):
        scores += w * m.predict_proba_fraud(X)

    return scores