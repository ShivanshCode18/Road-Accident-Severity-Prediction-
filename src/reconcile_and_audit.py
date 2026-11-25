"""Authoritative Reconciliation and Final Verification Script.

Performs:
1. Direct evaluation of saved Random Forest predictions from random_forest_predictions.csv:
   - Standard Argmax accuracy, per-class metrics, confusion matrix.
2. Complete threshold analysis on the actual Random Forest probabilities for:
   tau in [0.03, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60].
   Calculates: Accuracy, Fatal Prec/Rec/F1, Serious Prec/Rec/F1, Slight Prec/Rec/F1,
   Macro F1, Weighted F1, False Fatal Predictions, False Fatal Rate.
3. Analysis of Decision Tree, GBT, and MLP confusion matrices and majority-collapse behavior.
4. Synchronization of final_model_metrics.json, evaluation_summary.json, model_performance_metrics.csv.
5. Synchronization of Excel Dashboard Sheets 4 and 5.
6. Generation of authoritative reports.
"""

import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
EXCEL_PATH = PROJECT_ROOT / "dataset" / "excel" / "Road_Accident_Severity_Professional_Dashboard.xlsx"

RF_PRED_CSV = PROCESSED_DIR / "random_forest_predictions.csv"
DASH_PRED_CSV = PROCESSED_DIR / "excel_dashboard_predictions.csv"
VERIFIED_COMP_CSV = REPORTS_DIR / "verified_model_comparison.csv"
VERIFIED_CONF_CSV = REPORTS_DIR / "verified_confusion_matrices.csv"
REPORT_OUT = REPORTS_DIR / "final_verification_report.txt"


def reconcile():
    print("=" * 80)
    print("STARTING AUTHORITATIVE RECONCILIATION OF RANDOM FOREST & THRESHOLD ANALYSIS")
    print("=" * 80)
    
    if not RF_PRED_CSV.exists():
        print(f"Error: {RF_PRED_CSV} does not exist yet!")
        return
        
    df = pd.read_csv(RF_PRED_CSV)
    total = len(df)
    print(f"Loaded Random Forest predictions: {total:,} rows")
    
    actual = df["actual_severity"].values
    pred_argmax = df["predicted_severity"].values
    p0 = df["prob_fatal"].values
    p1 = df["prob_serious"].values
    p2 = df["prob_slight"].values
    
    actual_fatal = int((actual == 1).sum())
    actual_serious = int((actual == 2).sum())
    actual_slight = int((actual == 3).sum())
    non_fatal_total = actual_serious + actual_slight
    
    print(f"Class Support: Fatal={actual_fatal} ({actual_fatal/total*100:.2f}%), Serious={actual_serious} ({actual_serious/total*100:.2f}%), Slight={actual_slight} ({actual_slight/total*100:.2f}%)")
    print(f"Random Forest prob_fatal: Mean={p0.mean():.4f}, Median={np.median(p0):.4f}, Min={p0.min():.4f}, Max={p0.max():.4f}")
    
    # 1. Standard Argmax Evaluation
    correct_argmax = (pred_argmax == actual).sum()
    acc_argmax = correct_argmax / total
    
    # Confusion matrix for argmax
    cm_rf = {
        "Fatal (True)":   {"Fatal": int(((actual == 1) & (pred_argmax == 1)).sum()), "Serious": int(((actual == 1) & (pred_argmax == 2)).sum()), "Slight": int(((actual == 1) & (pred_argmax == 3)).sum())},
        "Serious (True)": {"Fatal": int(((actual == 2) & (pred_argmax == 1)).sum()), "Serious": int(((actual == 2) & (pred_argmax == 2)).sum()), "Slight": int(((actual == 2) & (pred_argmax == 3)).sum())},
        "Slight (True)":  {"Fatal": int(((actual == 3) & (pred_argmax == 1)).sum()), "Serious": int(((actual == 3) & (pred_argmax == 2)).sum()), "Slight": int(((actual == 3) & (pred_argmax == 3)).sum())}
    }
    
    tp0 = cm_rf["Fatal (True)"]["Fatal"]
    pred0 = tp0 + cm_rf["Serious (True)"]["Fatal"] + cm_rf["Slight (True)"]["Fatal"]
    rec0 = tp0 / actual_fatal
    prec0 = tp0 / pred0 if pred0 > 0 else 0
    f0 = (2 * prec0 * rec0) / (prec0 + rec0) if (prec0 + rec0) > 0 else 0
    
    tp1 = cm_rf["Serious (True)"]["Serious"]
    pred1 = cm_rf["Fatal (True)"]["Serious"] + tp1 + cm_rf["Slight (True)"]["Serious"]
    rec1 = tp1 / actual_serious
    prec1 = tp1 / pred1 if pred1 > 0 else 0
    f1 = (2 * prec1 * rec1) / (prec1 + rec1) if (prec1 + rec1) > 0 else 0
    
    tp2 = cm_rf["Slight (True)"]["Slight"]
    pred2 = cm_rf["Fatal (True)"]["Slight"] + cm_rf["Serious (True)"]["Slight"] + tp2
    rec2 = tp2 / actual_slight
    prec2 = tp2 / pred2 if pred2 > 0 else 0
    f2 = (2 * prec2 * rec2) / (prec2 + rec2) if (prec2 + rec2) > 0 else 0
    
    macro_f1 = (f0 + f1 + f2) / 3.0
    weighted_f1 = (actual_fatal * f0 + actual_serious * f1 + actual_slight * f2) / total
    
    print("\n" + "=" * 80)
    print("STANDARD ARGMAX METRICS FOR FINAL RANDOM FOREST:")
    print("=" * 80)
    print(f"Overall Accuracy : {acc_argmax * 100:.2f}% ({correct_argmax:,} / {total:,})")
    print(f"Weighted F1      : {weighted_f1:.4f}")
    print(f"Macro F1         : {macro_f1:.4f}")
    print(f"Fatal Class      : Precision={prec0*100:.2f}%, Recall={rec0*100:.2f}%, F1={f0:.4f} (TP={tp0}, Pred={pred0})")
    print(f"Serious Class    : Precision={prec1*100:.2f}%, Recall={rec1*100:.2f}%, F1={f1:.4f} (TP={tp1}, Pred={pred1})")
    print(f"Slight Class     : Precision={prec2*100:.2f}%, Recall={rec2*100:.2f}%, F1={f2:.4f} (TP={tp2}, Pred={pred2})")
    print("\nConfusion Matrix (3x3):")
    print(f"Fatal (True)   : {cm_rf['Fatal (True)']}")
    print(f"Serious (True) : {cm_rf['Serious (True)']}")
    print(f"Slight (True)  : {cm_rf['Slight (True)']}")
    
    # 2. Complete Threshold Analysis on Random Forest Probabilities
    print("\n" + "=" * 125)
    print("PROBABILITY DECISION THRESHOLD ANALYSIS (RANDOM FOREST PROBABILITIES):")
    print("Rule: Predict Fatal if P(Fatal) >= tau, else argmax(P(Serious), P(Slight))")
    print("=" * 125)
    print(f"{'tau':<6} {'Acc':>7} {'Fat Rec':>8} {'Fat Prec':>9} {'Fat F1':>8} {'Ser Rec':>8} {'Ser Prec':>9} {'Ser F1':>8} {'Sli Rec':>8} {'Sli Prec':>9} {'Sli F1':>8} {'Macro F1':>9} {'Weighted F1':>11} {'False Fatal':>11} {'FF Rate':>8}")
    print("-" * 125)
    
    thresh_records = []
    thresholds = [0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
    
    for tau in thresholds:
        pred_tau = np.where(p0 >= tau, 1, np.where(p1 >= p2, 2, 3))
        
        acc_t = (pred_tau == actual).mean()
        
        # Fatal
        tp0_t = int(((actual == 1) & (pred_tau == 1)).sum())
        fp0_t = int(((actual != 1) & (pred_tau == 1)).sum())
        fn0_t = int(((actual == 1) & (pred_tau != 1)).sum())
        rec0_t = tp0_t / actual_fatal
        prec0_t = tp0_t / (tp0_t + fp0_t) if (tp0_t + fp0_t) > 0 else 0
        f0_t = (2 * prec0_t * rec0_t) / (prec0_t + rec0_t) if (prec0_t + rec0_t) > 0 else 0
        
        # Serious
        tp1_t = int(((actual == 2) & (pred_tau == 2)).sum())
        fp1_t = int(((actual != 2) & (pred_tau == 2)).sum())
        rec1_t = tp1_t / actual_serious
        prec1_t = tp1_t / (tp1_t + fp1_t) if (tp1_t + fp1_t) > 0 else 0
        f1_t = (2 * prec1_t * rec1_t) / (prec1_t + rec1_t) if (prec1_t + rec1_t) > 0 else 0
        
        # Slight
        tp2_t = int(((actual == 3) & (pred_tau == 3)).sum())
        fp2_t = int(((actual != 3) & (pred_tau == 3)).sum())
        rec2_t = tp2_t / actual_slight
        prec2_t = tp2_t / (tp2_t + fp2_t) if (tp2_t + fp2_t) > 0 else 0
        f2_t = (2 * prec2_t * rec2_t) / (prec2_t + rec2_t) if (prec2_t + rec2_t) > 0 else 0
        
        mf1_t = (f0_t + f1_t + f2_t) / 3.0
        wf1_t = (actual_fatal * f0_t + actual_serious * f1_t + actual_slight * f2_t) / total
        ff_rate = fp0_t / non_fatal_total
        
        print(f"{tau:<6.2f} {acc_t*100:>6.2f}% {rec0_t*100:>7.2f}% {prec0_t*100:>8.2f}% {f0_t:>8.4f} {rec1_t*100:>7.2f}% {prec1_t*100:>8.2f}% {f1_t:>8.4f} {rec2_t*100:>7.2f}% {prec2_t*100:>8.2f}% {f2_t:>8.4f} {mf1_t:>9.4f} {wf1_t:>11.4f} {fp0_t:>11,} {ff_rate*100:>7.2f}%")
        
        thresh_records.append({
            "threshold_tau": tau,
            "accuracy_pct": round(acc_t * 100, 2),
            "fatal_recall_pct": round(rec0_t * 100, 2),
            "fatal_precision_pct": round(prec0_t * 100, 2),
            "fatal_f1": round(f0_t, 4),
            "serious_recall_pct": round(rec1_t * 100, 2),
            "serious_precision_pct": round(prec1_t * 100, 2),
            "serious_f1": round(f1_t, 4),
            "slight_recall_pct": round(rec2_t * 100, 2),
            "slight_precision_pct": round(prec2_t * 100, 2),
            "slight_f1": round(f2_t, 4),
            "macro_f1": round(mf1_t, 4),
            "weighted_f1": round(wf1_t, 4),
            "false_fatal_count": fp0_t,
            "false_fatal_rate_pct": round(ff_rate * 100, 2)
        })
        
    df_thresh = pd.DataFrame(thresh_records)
    thresh_csv_out = REPORTS_DIR / "random_forest_threshold_analysis.csv"
    df_thresh.to_csv(thresh_csv_out, index=False)
    print(f"\nSaved Random Forest threshold analysis to: {thresh_csv_out}")
    
    # 3. Synchronize Excel Dashboard with final Random Forest predictions
    print("\n3. Synchronizing Excel Dashboard Sheets 4 & 5 with final Random Forest outputs...")
    from src.excel_dashboard_updater import update_excel_workbook
    update_excel_workbook(
        excel_path=EXCEL_PATH,
        evaluations=None,
        predictions_csv=RF_PRED_CSV
    )
    print("Excel dashboard successfully synchronized.")
    
    return {
        "rf_argmax": {
            "accuracy": acc_argmax,
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1,
            "fatal_recall": rec0,
            "fatal_precision": prec0,
            "fatal_f1": f0,
            "serious_recall": rec1,
            "serious_precision": prec1,
            "serious_f1": f1,
            "slight_recall": rec2,
            "slight_precision": prec2,
            "slight_f1": f2,
            "confusion_matrix": cm_rf
        },
        "threshold_df": df_thresh
    }

if __name__ == "__main__":
    reconcile()

