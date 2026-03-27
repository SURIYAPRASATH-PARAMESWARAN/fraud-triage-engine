"""
main.py — Full fraud triage pipeline orchestration.

Execution order:
  1. Load data + temporal split
  2. Feature engineering
  3. Train all models (LR baseline, LightGBM, XGBoost)
  4. Calibrate LightGBM and XGBoost
  5. Build rank ensemble
  6. Evaluate all models (ROC-AUC, PR-AUC, Precision@K, Recall@K, ECE, Cost)
  7. SHAP explainability on best model
  8. Concept drift simulation
  9. Save all outputs + final triage queue
 10. Print comparison table

Usage:
    python main.py --data data/raw/creditcard.csv --policy-k 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Local imports ──────────────────────────────────────────────────────────────
from src.data import (
    load_creditcard_csv,
    make_temporal_splits,
    add_engineered_features,
)
from src.train import (
    train_logistic_baseline,
    train_lightgbm,
    train_xgboost,
    calibrate_model,
    ensemble_scores,
)
from src.evaluate import (
    evaluate,
    compare_models,
    optimal_k_by_cost,
    save_triage_queue,
)
from src.explainability import (
    compute_shap_values,
    plot_global_importance,
    plot_beeswarm,
    plot_waterfall_top_flagged,
    save_shap_values_csv,
)
from src.drift import simulate_drift, drift_summary_df
from src.viz import (
    plot_fraud_capture_curve,
    plot_pr_curves,
    plot_calibration_curves,
    plot_model_comparison,
    plot_drift_curve,
    plot_cost_curve,
)

OUTPUTS = Path("outputs")


def parse_args():
    p = argparse.ArgumentParser(description="Fraud Triage Engine")
    p.add_argument("--data",     default="data/raw/creditcard.csv", help="Path to creditcard.csv")
    p.add_argument("--policy-k", type=int, default=500,             help="Analyst review capacity")
    p.add_argument("--fn-cost",  type=float, default=500.0,         help="£ cost of missed fraud")
    p.add_argument("--fp-cost",  type=float, default=10.0,          help="£ cost of false alert")
    p.add_argument("--skip-shap", action="store_true",              help="Skip SHAP (faster runs)")
    return p.parse_args()


def banner(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def main():
    args = parse_args()
    t0 = time.time()

    # ── Create output directories (safe on all platforms) ─────────────────
    for _d in ["outputs/reports", "outputs/plots", "outputs/models",
               "outputs/plots/shap_waterfalls"]:
        Path(_d).mkdir(parents=True, exist_ok=True)

    # ── 1. Load + Split ────────────────────────────────────────────────────────
    banner("1. Loading Data")
    df = load_creditcard_csv(args.data)
    print(f"  Loaded {len(df):,} transactions | fraud rate: {df['Class'].mean():.4%}")

    banner("2. Feature Engineering")
    df_eng = add_engineered_features(df)
    new_feats = set(df_eng.columns) - set(df.columns)
    print(f"  Added {len(new_feats)} engineered features: {sorted(new_feats)}")

    banner("3. Temporal Split")
    splits = make_temporal_splits(df_eng)
    print(splits.summary())
    print(f"  Time ranges (train): {splits.meta['time_range_train']}")
    print(f"  Time ranges (test):  {splits.meta['time_range_test']}")

    # ── 2. Train Models ────────────────────────────────────────────────────────
    banner("4. Training Models")

    print("  [1/3] Logistic Regression baseline...")
    lr = train_logistic_baseline(splits.X_train, splits.y_train)
    print("  Done.")

    print("  [2/3] LightGBM (with early stopping on val PR-AUC)...")
    lgbm = train_lightgbm(
        splits.X_train, splits.y_train,
        splits.X_val,   splits.y_val,
    )
    print(f"  Best iteration: {lgbm.meta['best_iteration']}")

    print("  [3/3] XGBoost (with early stopping on val AUCPR)...")
    xgbm = train_xgboost(
        splits.X_train, splits.y_train,
        splits.X_val,   splits.y_val,
    )
    print(f"  Best iteration: {xgbm.meta['best_iteration']}")

    # ── 3. Calibrate ──────────────────────────────────────────────────────────
    banner("5. Probability Calibration (isotonic regression on val set)")
    lgbm_cal = calibrate_model(lgbm, splits.X_val, splits.y_val, method="isotonic")
    xgbm_cal = calibrate_model(xgbm, splits.X_val, splits.y_val, method="isotonic")
    print("  Calibrated: LightGBM, XGBoost")

    # ── 4. Score All Sets ──────────────────────────────────────────────────────
    banner("6. Scoring")
    models = {
        "logreg_baseline":   lr,
        "lightgbm":          lgbm,
        "lightgbm_cal":      lgbm_cal,
        "xgboost":           xgbm,
        "xgboost_cal":       xgbm_cal,
    }

    val_scores, test_scores = {}, {}
    for name, m in models.items():
        val_scores[name]  = m.predict_proba_fraud(splits.X_val)
        test_scores[name] = m.predict_proba_fraud(splits.X_test)
        print(f"  Scored: {name}")

    # Ensemble: average of both calibrated models
    val_scores["ensemble_cal"]  = ensemble_scores([lgbm_cal, xgbm_cal], splits.X_val)
    test_scores["ensemble_cal"] = ensemble_scores([lgbm_cal, xgbm_cal], splits.X_test)
    print(f"  Scored: ensemble_cal")
    print(f"  Total models scored: {len(val_scores)} — val and test sets")

    # ── 5. Evaluate ────────────────────────────────────────────────────────────
    banner("7. Evaluation")
    ks = (100, 250, 500, 750, 1000, 2000)

    val_reports  = [evaluate(n, splits.y_val,  val_scores[n],  ks=ks,
                             fn_cost=args.fn_cost, fp_cost=args.fp_cost)
                    for n in val_scores]
    test_reports = [evaluate(n, splits.y_test, test_scores[n], ks=ks,
                             fn_cost=args.fn_cost, fp_cost=args.fp_cost)
                    for n in test_scores]

    val_comp  = compare_models(val_reports)
    test_comp = compare_models(test_reports)

    print("\n── Validation ──")
    print(val_comp.to_string(index=False))
    print("\n── Test (locked) ──")
    print(test_comp.to_string(index=False))

    val_comp.to_csv(OUTPUTS / "reports" / "model_comparison_val.csv",  index=False)
    test_comp.to_csv(OUTPUTS / "reports" / "model_comparison_test.csv", index=False)

    # ── 6. Select Best Model for Policy ───────────────────────────────────────
    # Best = highest PR-AUC on validation (NOT test — test is locked until the end)
    best_name = val_comp.iloc[0]["model"]
    print(f"\n  Best model (by val PR-AUC): {best_name}")

    # ── 7. Save Triage Queue (locked test policy) ──────────────────────────────
    banner(f"8. Triage Queue — Top-{args.policy_k} (Test Set, Locked Policy)")
    triage_df = save_triage_queue(
        X=splits.X_test,
        y_true=splits.y_test,
        y_score=test_scores[best_name],
        k=args.policy_k,
        out_path=OUTPUTS / "reports" / f"triage_test_top{args.policy_k}_{best_name}.csv",
        model_name=best_name,
    )

    best_test = next(r for r in test_reports if r.model_name == best_name)
    tm = best_test.triage[args.policy_k]
    print(f"  Precision@{args.policy_k}: {tm.precision:.4f}")
    print(f"  Recall@{args.policy_k}:    {tm.recall:.4f}")
    print(f"  Frauds caught:   {tm.frauds_caught}")
    print(f"  Frauds missed:   {tm.frauds_missed}")
    print(f"  False alerts:    {tm.false_alerts}")
    print(f"  Expected cost:   £{tm.expected_cost:,.0f}  (FN=£{args.fn_cost}, FP=£{args.fp_cost})")

    ok, om = optimal_k_by_cost(best_test)
    print(f"\n  Cost-optimal K: {ok}  (expected cost £{om.expected_cost:,.0f})")

    # ── 8. SHAP ────────────────────────────────────────────────────────────────
    if not args.skip_shap:
        banner("9. SHAP Explainability")

        # SHAP requires a single tree model — ensemble_cal is a blend so we
        # always use lightgbm_cal for explanation (best single calibrated tree).
        # Fall back to lightgbm (uncalibrated) if calibrated version unavailable.
        shap_model_name = "lightgbm_cal" if "lightgbm_cal" in models else "lightgbm"
        shap_model_obj  = models[shap_model_name]
        print(f"  Using {shap_model_name} for SHAP (single tree model required)")

        # Unwrap CalibratedClassifierCV to get the raw LightGBM estimator
        raw_model = shap_model_obj.model
        if hasattr(raw_model, "calibrated_classifiers_"):
            raw_model = raw_model.calibrated_classifiers_[0].estimator

        try:
            shap_vals, X_shap = compute_shap_values(
                model=raw_model,
                X=splits.X_val,
                model_type="tree",
                max_samples=3000,
            )
            plot_global_importance(shap_vals, OUTPUTS / "plots" / "shap_importance.png")
            plot_beeswarm(shap_vals,          OUTPUTS / "plots" / "shap_beeswarm.png")
            plot_waterfall_top_flagged(
                shap_vals, X_shap,
                y_score=val_scores[shap_model_name][:len(X_shap)],
                out_dir=OUTPUTS / "plots" / "shap_waterfalls",
                top_n=5,
            )
            save_shap_values_csv(shap_vals, X_shap, OUTPUTS / "reports" / "shap_values.csv")
            print("  SHAP plots saved.")
        except Exception as e:
            print(f"  SHAP skipped (error): {e}")
    else:
        print("  (SHAP skipped via --skip-shap flag)")

    # ── 9. Drift Simulation ────────────────────────────────────────────────────
    banner("10. Concept Drift Simulation (Test Set Windows)")
    drift_windows = simulate_drift(
        splits.X_test, splits.y_test, test_scores[best_name],
        n_windows=5, alert_threshold=0.05,
    )
    drift_df = drift_summary_df(drift_windows)
    print(drift_df.to_string(index=False))
    drift_df.to_csv(OUTPUTS / "reports" / "drift_simulation.csv", index=False)

    # ── 10. Plots ──────────────────────────────────────────────────────────────
    banner("11. Generating Plots")

    score_sets_val  = {n: (splits.y_val,  val_scores[n])  for n in val_scores}
    score_sets_test = {n: (splits.y_test, test_scores[n]) for n in test_scores}

    plot_fraud_capture_curve(
        score_sets_val, OUTPUTS / "plots" / "fraud_capture_curve_val.png",
        policy_k=args.policy_k, title="Fraud Capture Curve (Validation)"
    )
    plot_fraud_capture_curve(
        score_sets_test, OUTPUTS / "plots" / "fraud_capture_curve_test.png",
        policy_k=args.policy_k, title="Fraud Capture Curve (Test — Locked)"
    )
    plot_pr_curves(score_sets_val, OUTPUTS / "plots" / "pr_curves_val.png")
    plot_pr_curves(score_sets_test, OUTPUTS / "plots" / "pr_curves_test.png",
                   title="Precision-Recall Curves (Test — Locked)")

    # Calibration plots
    cal_data = {r.model_name: r.calibration_data for r in val_reports}
    plot_calibration_curves(cal_data, OUTPUTS / "plots" / "calibration_curves.png")

    # Model comparison
    comp_plot_cols = {"roc_auc", "pr_auc", "recall@500", "precision@500"}
    plot_comp_df = test_comp.rename(columns={"recall@500": "recall@500", "precision@500": "precision@500"})
    plot_model_comparison(test_comp, OUTPUTS / "plots" / "model_comparison.png")

    # Drift
    plot_drift_curve(drift_df, OUTPUTS / "plots" / "drift_curve.png")

    # Cost curve
    plot_cost_curve(
        score_sets_test, OUTPUTS / "plots" / "cost_curve.png",
        fn_cost=args.fn_cost, fp_cost=args.fp_cost, policy_k=args.policy_k,
    )

    print("  All plots saved to outputs/plots/")

    # ── 11. Final Summary ──────────────────────────────────────────────────────
    banner("DONE")
    elapsed = time.time() - t0
    print(f"  Total runtime: {elapsed:.1f}s")
    print(f"  Best model: {best_name}")
    print(f"  Val PR-AUC: {val_comp.iloc[0]['pr_auc']}")
    print(f"  Test Recall@{args.policy_k}: {tm.recall:.4f}")
    print(f"  Outputs: {OUTPUTS.resolve()}")

    # Machine-readable summary for CI / MLflow
    summary = {
        "best_model": best_name,
        "val_pr_auc":         float(val_comp.iloc[0]["pr_auc"]),
        "test_roc_auc":       float(test_comp[test_comp["model"] == best_name]["roc_auc"].iloc[0]),
        "test_pr_auc":        float(test_comp[test_comp["model"] == best_name]["pr_auc"].iloc[0]),
        f"test_recall@{args.policy_k}":    round(tm.recall, 4),
        f"test_precision@{args.policy_k}": round(tm.precision, 4),
        "frauds_caught":      tm.frauds_caught,
        "frauds_missed":      tm.frauds_missed,
        "expected_cost_gbp":  round(tm.expected_cost, 0),
        "optimal_k":          ok,
        "runtime_s":          round(elapsed, 1),
    }
    (OUTPUTS / "reports" / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\n  summary.json written.")


if __name__ == "__main__":
    main()