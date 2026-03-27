# Methodology Notes

## Why This Is a Ranking Problem, Not a Classification Problem

Most fraud detection tutorials frame the problem as binary classification: is this transaction fraud (1) or not (0)? The model outputs a label, you pick a threshold, done.

This framing is wrong for practical fraud operations, for two reasons:

**1. Analyst capacity is finite.** A bank processes millions of transactions per day. A model flagging 10% of them as "fraud" gives analysts 100,000+ cases to review. That's not actionable. The real constraint is: analysts can review K cases per day. Which K should they be?

**2. False positive tolerance varies by cost.** Whether a false positive is acceptable depends on the relative cost of missing real fraud (chargebacks, regulatory fines) vs. wasting analyst time. A fixed threshold doesn't encode this.

The correct framing: **given model risk scores, rank all transactions and review the top K under capacity budget B**. Evaluate by recall@K and expected cost, not accuracy.

---

## Why Temporal Splitting Is Non-Negotiable

Random train/test splits are used throughout textbooks and Kaggle competitions. For fraud, they are methodologically incorrect.

Fraud transactions and legitimate transactions are time-correlated. A random split puts transactions from day 180 in both training and test. The model can implicitly learn patterns that only emerge later in the dataset.

More concretely: if a specific merchant was involved in a fraud ring that started on day 100, a random split puts some of those transactions in training. The model learns to flag that merchant. In a temporal split, the fraud ring only appears in the test set — as it would in real deployment.

Temporal splitting consistently produces lower (more honest) metrics. Reporting temporally-split metrics is a signal of methodological rigour.

---

## Why PR-AUC and Not ROC-AUC

At 0.17% fraud rate, the dataset has ~580 negative examples for every positive.

ROC-AUC is the probability that the model ranks a random positive above a random negative. This sounds good — but it's heavily influenced by the model's performance on negatives, which are 99.83% of the data. A model that correctly ranks 99% of non-fraud cases as low-risk will have a high ROC-AUC regardless of how well it finds actual fraud.

PR-AUC (Average Precision) integrates precision and recall across all thresholds. It only measures performance on the positive (fraud) class. A random classifier achieves PR-AUC ≈ base rate (0.0017). A good model must substantially beat this.

**Rule of thumb**: use PR-AUC as your primary metric whenever the positive class rate is below 5%.

---

## Calibration: Why It Matters for Threshold Setting

A well-calibrated model outputs predicted P(fraud) = 0.8 for transactions where approximately 80% are actually fraud (in expectation). Most models are not well-calibrated out of the box.

Tree models (LightGBM, XGBoost) are particularly prone to miscalibration on imbalanced data. The scale_pos_weight parameter adjusts the learning objective but doesn't directly calibrate probabilities.

**Isotonic regression** (CalibratedClassifierCV with method='isotonic', cv='prefit') fits a monotonic mapping from raw scores to calibrated probabilities on the held-out validation set. This doesn't change the ranking — just the scale of the probabilities.

Calibrated probabilities are essential for:
1. Setting a threshold based on an absolute probability cutoff (e.g., "review everything above P(fraud) > 0.3")
2. Cost-based optimisation (expected loss = P(fraud) × fraud_cost)
3. Combining scores from multiple models via averaging

---

## SHAP: The Right Tool for Tree Explainability

Feature importance from tree models (e.g., LightGBM's `feature_importances_`) measures how often a feature is used for splitting. This does not tell you the direction of the effect, magnitude, or interactions.

SHAP (SHapley Additive exPlanations) is theoretically grounded in cooperative game theory. For each prediction, SHAP decomposes the model output into a sum of contributions from each feature:

```
f(x) = baseline + φ₁ + φ₂ + ... + φₙ
```

Where φᵢ is the Shapley value of feature i — its marginal contribution to the prediction across all possible feature orderings.

For tree models, `TreeExplainer` computes exact Shapley values in O(TLD) time (T = trees, L = leaves, D = depth) — fast enough for production use.

**Key plots:**
- **Bar (global)**: mean |SHAP| per feature — importance without direction
- **Beeswarm (global)**: distribution of SHAP values per feature — shows direction, magnitude, and spread
- **Waterfall (local)**: single transaction — shows exactly how each feature pushed the prediction up or down from baseline

---

## Concept Drift: The Production Problem Nobody Talks About

Fraud models decay. Fraudsters adapt to detection patterns, payment rails change, cardholder behaviour evolves. A model trained in Q1 may have degraded significantly by Q3.

Drift monitoring requires:
1. A baseline performance metric (PR-AUC on validation)
2. A stream of recent predictions with labels (delayed ground truth from chargebacks)
3. Alerting logic when performance drops below acceptable threshold
4. Retraining cadence

This project simulates drift by evaluating per-window PR-AUC on the test set. In production, this logic runs on a daily or weekly basis against incoming labelled transactions. The retraining trigger (5% PR-AUC drop) is configurable based on business risk tolerance.