# Fraud Triage Engine

> **A production-aware fraud risk scoring and analyst triage system built on the Kaggle credit card fraud dataset.**
> Reframes detection as a capacity-constrained ranking problem — not a binary classifier.

---

## The Core Idea

Fraud teams cannot review every transaction. A model that outputs a probability score is only useful if that score reliably *ranks* fraud above legitimate transactions under real operational constraints.

This project answers: **if analysts can review K transactions per day, what is the optimal policy and how much fraud does it capture?**

The answer is operationalised through a cost-sensitive triage framework that maps model outputs to business decisions, evaluated with metrics that reflect real analyst workflows.

---

## Architecture

```
creditcard.csv
     │
     ▼
┌──────────────────────────────────┐
│  Temporal Split (not random)     │  ← 60% train / 20% val / 20% test
│  + Feature Engineering           │    (causal ordering preserved)
└──────────────────┬───────────────┘
                   │
     ┌─────────────┼──────────────┐
     ▼             ▼              ▼
  LR Baseline   LightGBM      XGBoost
  (comparison)  (primary)     (secondary)
     │             │              │
     └─────────────┴──────────────┘
                   │
         Isotonic Calibration
         (P(fraud) aligned with
          empirical fraud rate)
                   │
          Rank Ensemble (avg)
                   │
     ┌─────────────┼──────────────┐
     ▼             ▼              ▼
  SHAP          Triage        Drift
  Explainer     Queue K=500   Monitor
```

---

## Key Design Decisions

### 1. Temporal Splitting (not random)
Random train/test splitting leaks future information into training. A model trained on a random 80% sample has implicitly seen data points from the same time window as the test set. For fraud — where patterns evolve — this produces optimistic, unrealistic metrics.

**This project uses contiguous time windows**: train on the first 60% of transactions (chronologically), validate on the next 20%, test on the final 20%. This simulates real deployment: the model is trained on historical data and evaluated on future, unseen transactions.

### 2. PR-AUC over ROC-AUC
At 0.17% fraud rate, a model that classifies everything as legitimate achieves 99.83% accuracy. ROC-AUC is also inflated by the large number of true negatives. **PR-AUC** (Average Precision) is the correct primary metric for severely imbalanced binary classification — it focuses entirely on the positive (fraud) class.

### 3. Calibrated Probabilities
Tree models (LightGBM, XGBoost) produce well-ranked scores but poorly calibrated probabilities on imbalanced data. Raw scores from different models are also on incompatible scales, making direct averaging meaningless.

**Isotonic regression calibration** (fitted on the validation set, not training set) aligns predicted P(fraud) with empirical fraud rates. This is validated via reliability diagrams (calibration curves) and Expected Calibration Error (ECE).

### 4. Cost-Sensitive Thresholding
Instead of picking K=500 arbitrarily, the optimal review capacity is derived from a cost matrix:
- **FN cost**: £500 (missed fraud: chargeback + operational loss)
- **FP cost**: £10 (false alert: ~6 min analyst review at £100/hr)

`optimal_k = argmin_K [ missed_frauds(K) × FN_cost + false_alerts(K) × FP_cost ]`

The cost curve plot visualises this across all K values for all models.

### 5. SHAP Explainability
Fraud detection systems that cannot explain individual decisions cannot be deployed under UK FCA guidelines or the EU AI Act. SHAP (TreeExplainer) provides:
- **Global**: feature importance bar chart + beeswarm plot
- **Local**: waterfall plots for each flagged transaction ("why was this flagged?")

This bridges the gap between model performance and operational use.

### 6. Concept Drift Monitoring
The test set is split into 5 chronological windows. PR-AUC is evaluated per window to simulate weekly performance monitoring. A retraining alert is triggered if PR-AUC drops more than 5% from peak — a basic but production-realistic monitoring trigger.

---

## Feature Engineering

The raw dataset provides V1–V28 (pre-PCA transformed for privacy) plus `Time` and `Amount`. We add:

| Feature | Rationale |
|---|---|
| `Amount_log1p` | Raw Amount is right-skewed; log-transform compresses outliers |
| `Amount_zscore` | Standardised amount; large z-scores signal unusual transaction sizes |
| `Hour`, `Hour_sin`, `Hour_cos` | Cyclical time-of-day encoding; fraud peaks overnight |
| `V14_V17`, `V12_V14`, `V14_sq` | Interaction terms; V14, V12, V17 are top fraud signals in literature |
| `V_norm_l2` | L2 norm of V-features; measures overall 'unusualness' of the PCA vector |

---

## Results

### Data Split (Temporal)

| Split | Rows | Fraud Cases | Fraud Rate |
|---|---|---|---|
| Train | 170,884 | 360 | 0.211% |
| Validation | 56,961 | 57 | 0.100% |
| Test | 56,962 | 75 | 0.132% |

> Note: fraud rate drops across time windows — a real-world effect of temporal splitting that random splits would hide.

---

### Model Comparison — Validation Set

| Model | ROC-AUC | PR-AUC | ECE | Recall@500 | Lift@500 |
|---|---|---|---|---|---|
| **Ensemble (LGB+XGB cal)** | **1.000** | **0.989** | **0.00078** | **1.000** | **114×** |
| LightGBM (calibrated) | 1.000 | 0.973 | 0.00084 | 1.000 | 114× |
| XGBoost (calibrated) | 1.000 | 0.971 | 0.00051 | 1.000 | 114× |
| XGBoost (raw) | 0.974 | 0.777 | 0.029 | 0.842 | 96× |
| Logistic Regression | 0.980 | 0.768 | 0.069 | 0.860 | 98× |
| LightGBM (raw) | 0.817 | 0.185 | 0.004 | 0.790 | 90× |

---

### Model Comparison — Test Set (Locked)

| Model | ROC-AUC | PR-AUC | ECE | Recall@500 | Lift@500 |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.984** | **0.801** | **0.078** | **0.867** | **99×** |
| XGBoost (raw) | 0.973 | 0.773 | 0.031 | 0.813 | 93× |
| Ensemble (LGB+XGB cal) | 0.978 | 0.765 | 0.000 | 0.813 | 93× |
| LightGBM (calibrated) | 0.948 | 0.756 | 0.001 | 0.813 | 93× |
| XGBoost (calibrated) | 0.971 | 0.746 | 0.000 | 0.853 | 97× |
| LightGBM (raw) | 0.829 | 0.180 | 0.001 | 0.800 | 91× |

> **Notable finding:** Logistic Regression wins on the test set (PR-AUC 0.801) despite the ensemble dominating validation (PR-AUC 0.989). This is a direct consequence of honest temporal splitting — the calibrated tree models overfit to validation-era patterns. This is exactly the kind of finding that matters in production and would be invisible with random splits.

---

### Triage Policy: Top-500 Daily Reviews (Test Set, Locked)

```
Model used:          Ensemble (LGB + XGB calibrated)
Transactions scored: 56,962
Review budget:       500  (0.88% of all transactions)

Frauds caught:       61 / 75   (81.3% recall)
False alerts:        439
Expected cost:       £11,390   (vs £37,500 with no model)
Cost reduction:      70%

Cost-optimal K:      250       (£8,890 expected cost)
```

---

### Concept Drift — PR-AUC Across Test Windows

| Window | Time Range (s) | Transactions | Fraud Cases | PR-AUC | Retrain Alert |
|---|---|---|---|---|---|
| W0 | 145,248 – 150,016 | 11,392 | 18 | 0.816 | ✅ No |
| W1 | 150,016 – 155,000 | 11,392 | 23 | 0.849 | ✅ No |
| W2 | 155,001 – 160,278 | 11,392 | 16 | 0.754 | ⚠️ Yes |
| W3 | 160,279 – 165,582 | 11,392 | 8 | 0.753 | ⚠️ Yes |
| W4 | 165,582 – 172,792 | 11,394 | 10 | 0.582 | ⚠️ Yes |

> PR-AUC degrades from **0.849 → 0.582** (31% drop) across the test period. Retraining alerts fire in the final 3 windows. This demonstrates why static model deployment without monitoring is risky — and why this system includes drift detection.

---

## Repo Structure

```
fraud-triage-engine/
├── src/
│   ├── data.py           # Temporal splitting + feature engineering
│   ├── train.py          # LR / LightGBM / XGBoost + calibration + ensemble
│   ├── evaluate.py       # ROC-AUC, PR-AUC, Precision@K, Recall@K, ECE, cost
│   ├── explainability.py # SHAP global + local explanations
│   ├── drift.py          # Concept drift simulation + retraining alerts
│   └── viz.py            # All plots (capture curve, PR, calibration, drift, cost)
├── notebooks/
│   └── fraud_triage_walkthrough.ipynb   # End-to-end narrative notebook
├── tests/
│   ├── test_data.py      # Temporal split correctness, feature engineering
│   └── test_evaluate.py  # Triage metrics, ECE, evaluate()
├── outputs/
│   ├── plots/            # All generated figures
│   ├── reports/          # CSVs, model comparison, triage queue, summary.json
│   └── models/           # Serialised model artefacts
├── data/
│   └── raw/              # creditcard.csv goes here (not committed to git)
├── docs/
│   └── methodology.md    # Deep-dive on design decisions
├── main.py               # Full pipeline orchestration
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Place dataset
# Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place at: data/raw/creditcard.csv

# 3. Run full pipeline
python main.py

# 4. Custom cost structure or capacity
python main.py --policy-k 750 --fn-cost 800 --fp-cost 15

# 5. Skip SHAP (faster iteration)
python main.py --skip-shap

# 6. Run tests
python -m pytest tests/ -v
```

---

## Outputs

| File | Description |
|---|---|
| `outputs/reports/model_comparison_test.csv` | Side-by-side model metrics |
| `outputs/reports/triage_test_top500_*.csv` | Analyst review queue with risk tiers |
| `outputs/reports/drift_simulation.csv` | Per-window performance degradation |
| `outputs/reports/shap_values.csv` | SHAP values for all validation samples |
| `outputs/reports/summary.json` | Machine-readable final metrics |
| `outputs/plots/fraud_capture_curve_*.png` | Recall@K vs capacity |
| `outputs/plots/pr_curves_*.png` | Full precision-recall curves |
| `outputs/plots/calibration_curves.png` | Reliability diagram |
| `outputs/plots/model_comparison.png` | Side-by-side metric bar chart |
| `outputs/plots/shap_importance.png` | Global SHAP feature importance |
| `outputs/plots/shap_beeswarm.png` | SHAP value distribution per feature |
| `outputs/plots/shap_waterfalls/` | Per-transaction analyst explanations (top 5) |
| `outputs/plots/drift_curve.png` | PR-AUC degradation over time |
| `outputs/plots/cost_curve.png` | Expected £ cost vs review capacity |

---

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions over 2 days
- 492 frauds (0.172%)
- Features: V1–V28 (PCA-anonymised), Time, Amount, Class

---

## Possible Extensions

- **Hyperparameter optimisation**: Optuna TPE search with PR-AUC objective
- **Isolation Forest / Autoencoder**: Unsupervised anomaly detection as an additional signal
- **Streaming simulation**: Simulate daily batch retraining and score fresh transactions
- **FastAPI serving**: REST endpoint returning `{fraud_prob, risk_tier, top_shap_features}`
- **MLflow tracking**: Log all experiments, metrics, and model artefacts with run comparison

---

## Tech Stack

`Python 3.14` · `LightGBM 4.3` · `XGBoost 2.0` · `scikit-learn 1.6` · `SHAP 0.45` · `pandas 2.2` · `matplotlib 3.8`