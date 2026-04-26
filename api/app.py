"""
api/app.py — FastAPI serving layer for the Fraud Triage Engine.

Endpoints:
  GET  /health          — liveness check (used by Docker / k8s)
  POST /predict         — score a single transaction, returns fraud_prob + risk_tier + SHAP
  POST /score-batch     — score up to 1000 transactions, returns ranked queue

Model loading:
  On startup, loads the serialised ensemble from models/ensemble_scores_model.pkl
  and the LightGBM calibrated model from models/lgbm_cal.pkl (for SHAP).
  If models are not found, the API starts in degraded mode and returns 503
  on prediction endpoints — so the container stays up for health checks.

Usage:
  # 1. Train and save the model first
  python main.py

  # 2. Start the API locally
  uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

  # 3. Or via Docker
  docker build -t fraud-triage-api .
  docker run -p 8000:8000 fraud-triage-api

  # 4. Interactive docs
  http://localhost:8000/docs
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("fraud-api")

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS_DIR  = Path(os.getenv("MODELS_DIR", "models"))
SUMMARY_PATH = Path("outputs/reports/summary.json")

RISK_TIERS = {
    (0.85, 1.01): "CRITICAL",
    (0.60, 0.85): "HIGH",
    (0.30, 0.60): "MEDIUM",
    (0.00, 0.30): "LOW",
}

# V-feature names expected by the model
V_FEATURES   = [f"V{i}" for i in range(1, 29)]
BASE_FEATURES = ["Time", "Amount"] + V_FEATURES

# Engineered feature names added by add_engineered_features()
ENG_FEATURES = [
    "Amount_log1p", "Amount_zscore",
    "Hour", "Hour_sin", "Hour_cos",
    "V14_V17", "V12_V14", "V14_sq", "V_norm_l2",
]

ALL_FEATURES = BASE_FEATURES + ENG_FEATURES


# ── Model store (populated at startup) ────────────────────────────────────────
class ModelStore:
    lgbm_cal: Any = None          # calibrated LightGBM (for SHAP + scoring)
    xgb_cal: Any  = None          # calibrated XGBoost  (for ensemble)
    shap_explainer: Any = None
    feature_names: List[str] = []
    startup_error: Optional[str] = None


store = ModelStore()


# ── Helper: feature engineering (mirrors src/data.py) ─────────────────────────
def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Amount_log1p"] = np.log1p(df["Amount"])
    df["Amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / (df["Amount"].std() + 1e-9)
    df["Hour"]    = (df["Time"] // 3600) % 24
    df["Hour_sin"]= np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"]= np.cos(2 * np.pi * df["Hour"] / 24)

    if all(c in df.columns for c in ["V14", "V17", "V12"]):
        df["V14_V17"] = df["V14"] * df["V17"]
        df["V12_V14"] = df["V12"] * df["V14"]
        df["V14_sq"]  = df["V14"] ** 2

    v_cols = [c for c in df.columns if c.startswith("V")]
    df["V_norm_l2"] = np.sqrt((df[v_cols] ** 2).sum(axis=1))
    return df


def _risk_tier(prob: float) -> str:
    for (lo, hi), tier in RISK_TIERS.items():
        if lo <= prob < hi:
            return tier
    return "LOW"


def _ensemble_score(X: pd.DataFrame) -> np.ndarray:
    """Average of calibrated LightGBM + XGBoost probabilities."""
    p_lgbm = store.lgbm_cal.predict_proba(X)[:, 1]
    if store.xgb_cal is not None:
        p_xgb = store.xgb_cal.predict_proba(X)[:, 1]
        return (p_lgbm + p_xgb) / 2.0
    return p_lgbm


def _top_shap_features(X_row: pd.DataFrame, top_n: int = 5) -> List[Dict[str, Any]]:
    """Return top-N SHAP features for a single transaction row."""
    if store.shap_explainer is None:
        return []
    try:
        sv = store.shap_explainer(X_row)
        vals = sv.values[0] if sv.values.ndim == 2 else sv.values[0, :, 1]
        pairs = sorted(
            zip(X_row.columns.tolist(), vals.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]
        return [
            {"feature": f, "shap_value": round(v, 6), "direction": "increases_fraud" if v > 0 else "decreases_fraud"}
            for f, v in pairs
        ]
    except Exception as e:
        log.warning(f"SHAP computation failed: {e}")
        return []


# ── Lifespan: load models on startup ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading models from %s ...", MODELS_DIR)
    try:
        lgbm_path = MODELS_DIR / "lgbm_cal.pkl"
        xgb_path  = MODELS_DIR / "xgb_cal.pkl"

        if not lgbm_path.exists():
            raise FileNotFoundError(
                f"Model not found: {lgbm_path}. "
                "Run `python main.py` first to train and save models."
            )

        store.lgbm_cal = joblib.load(lgbm_path)
        log.info("Loaded lgbm_cal from %s", lgbm_path)

        if xgb_path.exists():
            store.xgb_cal = joblib.load(xgb_path)
            log.info("Loaded xgb_cal from %s", xgb_path)
        else:
            log.warning("xgb_cal.pkl not found — using lgbm_cal only (no ensemble)")

        # Build SHAP explainer from raw LightGBM estimator inside CalibratedClassifierCV
        raw_lgbm = store.lgbm_cal
        if hasattr(raw_lgbm, "calibrated_classifiers_"):
            raw_lgbm = raw_lgbm.calibrated_classifiers_[0].estimator
        store.shap_explainer = shap.TreeExplainer(raw_lgbm)
        log.info("SHAP TreeExplainer ready")

        log.info("API ready ✓")

    except Exception as e:
        store.startup_error = str(e)
        log.error("Model loading failed: %s", e)
        log.warning("API starting in DEGRADED mode — prediction endpoints will return 503")

    yield  # API runs here

    log.info("Shutting down fraud-triage-api")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Triage Engine API",
    description=(
        "Production REST API for the Fraud Triage Engine. "
        "Scores transactions by fraud probability and returns risk tiers + SHAP explanations."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ─────────────────────────────────────────────────
class Transaction(BaseModel):
    """Single transaction features. V1–V28 are PCA-anonymised by the dataset provider."""
    Time:   float = Field(..., description="Seconds elapsed from start of dataset")
    Amount: float = Field(..., ge=0, description="Transaction amount in GBP")

    # V1–V28 PCA components
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float

    model_config = {"json_schema_extra": {
        "example": {
            "Time": 406.0, "Amount": 149.62,
            "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
            "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
            "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
            "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
            "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
            "V26": -0.19, "V27": 0.13, "V28": -0.02,
        }
    }}


class PredictResponse(BaseModel):
    fraud_prob:      float
    risk_tier:       str
    top_shap:        List[Dict[str, Any]]
    latency_ms:      float


class BatchPredictRequest(BaseModel):
    transactions: List[Transaction] = Field(..., max_length=1000)


class BatchPredictResponse(BaseModel):
    count:        int
    results:      List[Dict[str, Any]]   # ranked by fraud_prob descending
    latency_ms:   float


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    error:         Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness + readiness check. Returns 200 even in degraded mode."""
    return HealthResponse(
        status="degraded" if store.startup_error else "ok",
        model_loaded=store.lgbm_cal is not None,
        error=store.startup_error,
    )


@app.get("/model-info", tags=["ops"])
def model_info():
    """Return training summary metrics from the last main.py run."""
    if SUMMARY_PATH.exists():
        return json.loads(SUMMARY_PATH.read_text())
    raise HTTPException(status_code=404, detail="summary.json not found. Run main.py first.")


@app.post("/predict", response_model=PredictResponse, tags=["scoring"])
def predict(tx: Transaction):
    """
    Score a single transaction.

    Returns:
      - fraud_prob: P(fraud) from calibrated ensemble
      - risk_tier: CRITICAL / HIGH / MEDIUM / LOW
      - top_shap: top-5 features driving this prediction
      - latency_ms: server-side inference time
    """
    if store.lgbm_cal is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {store.startup_error}")

    t0 = time.perf_counter()

    row = pd.DataFrame([tx.model_dump()])
    row_eng = _engineer(row)

    # Only keep features the model was trained on
    feature_cols = [c for c in ALL_FEATURES if c in row_eng.columns]
    X = row_eng[feature_cols]

    prob = float(_ensemble_score(X)[0])
    tier = _risk_tier(prob)
    top_shap = _top_shap_features(X)

    latency = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        fraud_prob=round(prob, 6),
        risk_tier=tier,
        top_shap=top_shap,
        latency_ms=round(latency, 2),
    )


@app.post("/score-batch", response_model=BatchPredictResponse, tags=["scoring"])
def score_batch(req: BatchPredictRequest):
    """
    Score up to 1000 transactions in a single call.

    Returns results ranked by fraud_prob descending — ready to use as an analyst queue.
    SHAP is not computed for batch (use /predict for individual explanations).
    """
    if store.lgbm_cal is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {store.startup_error}")

    t0 = time.perf_counter()

    df = pd.DataFrame([tx.model_dump() for tx in req.transactions])
    df_eng = _engineer(df)
    feature_cols = [c for c in ALL_FEATURES if c in df_eng.columns]
    X = df_eng[feature_cols]

    probs = _ensemble_score(X)

    results = []
    for i, prob in enumerate(probs.tolist()):
        results.append({
            "transaction_index": i,
            "fraud_prob": round(prob, 6),
            "risk_tier": _risk_tier(prob),
            "amount": req.transactions[i].Amount,
        })

    results.sort(key=lambda r: r["fraud_prob"], reverse=True)
    latency = (time.perf_counter() - t0) * 1000

    return BatchPredictResponse(
        count=len(results),
        results=results,
        latency_ms=round(latency, 2),
    )


# ── Global exception handler ───────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )