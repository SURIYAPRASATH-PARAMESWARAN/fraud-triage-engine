# Fraud Triage Engine

> **A production-aware fraud risk scoring and analyst triage system built on the Kaggle credit card fraud dataset.**
> Reframes detection as a capacity-constrained ranking problem — not a binary classifier.

---

## The Core Idea

Fraud teams cannot review every transaction. A model that outputs a probability score is only useful if that score reliably *ranks* fraud above legitimate transactions under real operational constraints.

This project answers: **if analysts can review K transactions per day, what is the optimal policy and how much fraud does it capture?**

The answer is operationalised through a cost-sensitive triage framework that maps model outputs to business decisions, evaluated with metrics that reflect real analyst workflows. The trained model is served via a FastAPI REST API, containerised with Docker, returning calibrated fraud probabilities, risk tiers, and per-transaction SHAP explanations.

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
                   │
                   ▼
         FastAPI REST API
         (Dockerised, /predict
          /score-batch /health)
```

---

## Key Design Decisions

### 1. Temporal Splitting (not random)
Random train/test splitting leaks future information into training. A model trained on a random 80% sample has implicitly seen data points from the same time window as the test set. For fraud — where patterns evolve — this produces optimistic, unrealistic metrics.

**This project uses contiguous time windows**: train on the first 60% of transactions (chronologically), validate on the next 20%, test on the final 20%. This simulates real deployment: the model is trained on historical data and evaluated on future, unseen transactions.

### 2. PR-AUC over ROC-AUC
At 0.17% fraud rate, a model that classifies everything as legitimate achieves 99.83% accuracy. ROC-AUC is also inflated by the large number of true negatives. **PR-AUC** (Average Precision) is the correct primary metric for severely imbalanced binary classification — it focuses entirely on the positive (fraud) class.

### 3. Calibrated Probabilities
Tree models (LightGBM, XGBoost) produce well-ranked but poorly calibrated probabilities on imbalanced data. Raw scores from different models are also on incompatible scales, making direct averaging meaningless.

**Isotonic regression calibration** (fitted on the validation set, not training set) aligns predicted P(fraud) with empirical fraud rates. Validated via reliability diagrams (calibration curves) and Expected Calibration Error (ECE).

### 4. Cost-Sensitive Thresholding
Instead of picking K=500 arbitrarily, the optimal review capacity is derived from a cost matrix:
- **FN cost**: £500 (missed fraud: chargeback + operational loss)
- **FP cost**: £10 (false alert: ~6 min analyst review at £100/hr)

`optimal_k = argmin_K [ missed_frauds(K) × FN_cost + false_alerts(K) × FP_cost ]`

### 5. SHAP Explainability
Fraud detection systems that cannot explain individual decisions cannot be deployed under UK FCA guidelines or the EU AI Act. SHAP (TreeExplainer) provides:
- **Global**: feature importance bar chart + beeswarm plot
- **Local**: waterfall plots per flagged transaction ("why was this flagged?")
- **API**: top-5 SHAP features returned on every `/predict` call

### 6. Concept Drift Monitoring
The test set is split into 5 chronological windows. PR-AUC is evaluated per window to simulate weekly performance monitoring. A retraining alert fires if PR-AUC drops more than 5% from peak.

---

## Feature Engineering

| Feature | Rationale |
|---|---|
| `Amount_log1p` | Raw Amount is right-skewed; log-transform compresses outliers |
| `Amount_zscore` | Standardised amount; large z-scores signal unusual transaction sizes |
| `Hour`, `Hour_sin`, `Hour_cos` | Cyclical time-of-day encoding; fraud peaks overnight |
| `V14_V17`, `V12_V14`, `V14_sq` | Interaction terms; V14, V12, V17 are top fraud signals in literature |
| `V_norm_l2` | L2 norm of V-features; measures overall unusualness of the PCA vector |

---

## Results

### Data Split (Temporal)

| Split | Rows | Fraud Cases | Fraud Rate |
|---|---|---|---|
| Train | 170,884 | 360 | 0.211% |
| Validation | 56,961 | 57 | 0.100% |
| Test | 56,962 | 75 | 0.132% |

> Fraud rate drops across time windows — a real-world effect of temporal splitting that random splits would hide.

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

> **Notable finding:** Logistic Regression wins on the test set (PR-AUC 0.801) despite the ensemble dominating validation (PR-AUC 0.989). A direct consequence of honest temporal splitting — the calibrated tree models overfit to validation-era patterns. This finding is invisible with random splits.

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

> PR-AUC degrades from **0.849 → 0.582** (31% drop). Retraining alerts fire in the final 3 windows.

---

## Deployment

### Local API (no Docker)

```bash
# 1. Train and save models
python main.py

# 2. Start the server
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# 3. Interactive docs
open http://localhost:8000/docs
```

### Docker

```bash
# Build
docker build -t fraud-triage-api .

# Run — mount models directory so API loads trained weights
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  fraud-triage-api

# Health check
curl http://localhost:8000/health
# → {"status":"ok","model_loaded":true,"error":null}
```

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check — returns 200 even if model not loaded |
| GET | `/model-info` | Training summary metrics from last `main.py` run |
| POST | `/predict` | Score one transaction → `fraud_prob`, `risk_tier`, top-5 SHAP |
| POST | `/score-batch` | Score up to 1,000 transactions → ranked analyst queue |

### Example: single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 406.0, "Amount": 149.62, "V1": -1.36, "V2": -0.07, "V3": 2.54,
       "V4": 1.38, "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
       "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62, "V13": -0.99,
       "V14": -0.31, "V15": 1.47, "V16": -0.47, "V17": 0.21, "V18": 0.03,
       "V19": 0.40, "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11,
       "V24": 0.07, "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02}'
```

```json
{
  "fraud_prob": 0.847312,
  "risk_tier": "CRITICAL",
  "top_shap": [
    {"feature": "V14", "shap_value": -0.312, "direction": "increases_fraud"},
    {"feature": "V4",  "shap_value":  0.198, "direction": "decreases_fraud"},
    {"feature": "V12", "shap_value": -0.187, "direction": "increases_fraud"},
    {"feature": "Amount_log1p", "shap_value": 0.091, "direction": "decreases_fraud"},
    {"feature": "V17", "shap_value": -0.073, "direction": "increases_fraud"}
  ],
  "latency_ms": 4.2
}
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset and place at data/raw/creditcard.csv
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# 3. Run full pipeline (trains, evaluates, saves models)
python main.py

# 4. Custom cost structure or capacity
python main.py --policy-k 750 --fn-cost 800 --fp-cost 15

# 5. Skip SHAP for faster iteration
python main.py --skip-shap

# 6. Run tests
pytest tests/ -v
```

---

## Repo Structure

```
fraud-triage-engine/
├── api/
│   └── app.py                       # FastAPI — /predict, /score-batch, /health
├── src/
│   ├── data.py                      # Temporal split + feature engineering
│   ├── train.py                     # LR / LightGBM / XGBoost + calibration + ensemble
│   ├── evaluate.py                  # ROC-AUC, PR-AUC, Precision@K, ECE, cost
│   ├── explainability.py            # SHAP global + local explanations
│   ├── drift.py                     # Concept drift simulation + retraining alerts
│   └── viz.py                       # All plots
├── tests/
│   ├── test_data.py                 # Temporal split correctness, feature engineering
│   └── test_evaluate.py             # Triage metrics, ECE, evaluate()
├── models/                          # Saved after main.py — gitignored
│   ├── lgbm_cal.pkl
│   ├── xgb_cal.pkl
│   └── feature_names.json
├── notebooks/
│   └── fraud_triage_walkthrough.ipynb
├── outputs/
│   ├── plots/
│   └── reports/
├── data/
│   └── raw/                         # creditcard.csv — not committed
├── docs/
│   └── methodology.md
├── Dockerfile
├── main.py
├── requirements.txt
└── README.md
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
- **Streaming simulation**: Simulate daily batch retraining on fresh transactions
- **MLflow tracking**: Log all experiments, metrics, and model artefacts with run comparison

---

## Tech Stack

`Python 3.11` · `LightGBM 4.3` · `XGBoost 2.0` · `scikit-learn 1.6` · `SHAP 0.45` · `pandas 2.2` · `NumPy` · `matplotlib 3.8` · `FastAPI` · `uvicorn` · `Docker` · `joblib` · `pytest`