"""Recalculate complete threshold metrics directly from saved prediction probabilities.

Performs:
1. Exact sweep across threshold tau values.
2. Complete 3-class assignment:
   if prob_fatal >= tau: predict Fatal (1)
   else: Serious (2) if prob_serious >= prob_slight else Slight (3)
3. Precision, recall, and F1 for Fatal, Serious, and Slight.
4. Macro F1 and Weighted F1.
5. False Fatal count and False Fatal Rate.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRED_PATH = PROJECT_ROOT / "dataset" / "processed" / "excel_dashboard_predictions.csv"

def evaluate_thresholds(csv_path: Path):
    df = pd.read_csv(csv_path)
    actual = df['actual_severity'].values
    p0 = df['prob_fatal'].values
    p1 = df['prob_serious'].values
    p2 = df['prob_slight'].values
    total = len(df)
    
    non_fatal_total = (actual != 1).sum() # 19,793
    actual_fatal = (actual == 1).sum()    # 306
    actual_serious = (actual == 2).sum()  # 4,843
    actual_slight = (actual == 3).sum()   # 14,950
    
    print(f"Dataset: {csv_path.name}")
    print(f"Total rows: {total:,} | Fatal: {actual_fatal} | Serious: {actual_serious} | Slight: {actual_slight}")
    print(f"Mean prob_fatal: {p0.mean():.4f} | Median: {np.median(p0):.4f} | Min: {p0.min():.4f} | Max: {p0.max():.4f}")
    print("\n" + "=" * 125)
    print(f"{'tau':<6} {'Acc':>7} {'Fat Rec':>8} {'Fat Prec':>9} {'Fat F1':>8} {'Ser Rec':>8} {'Ser Prec':>9} {'Ser F1':>8} {'Sli Rec':>8} {'Sli Prec':>9} {'Sli F1':>8} {'Macro F1':>9} {'Weighted F1':>11} {'False Fatal':>11} {'FF Rate':>8}")
    print("-" * 125)
    
    records = []
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.27, 0.30, 0.33, 0.35, 0.40, 0.45, 0.50, 0.60]
    
    for tau in thresholds:
        pred = np.where(p0 >= tau, 1, np.where(p1 >= p2, 2, 3))
        
        acc = (pred == actual).mean()
        
        # Fatal (Class 1)
        tp0 = int(((actual == 1) & (pred == 1)).sum())
        fp0 = int(((actual != 1) & (pred == 1)).sum())
        fn0 = int(((actual == 1) & (pred != 1)).sum())
        rec0 = tp0 / actual_fatal if actual_fatal > 0 else 0.0
        prec0 = tp0 / (tp0 + fp0) if (tp0 + fp0) > 0 else 0.0
        f1_0 = (2 * prec0 * rec0) / (prec0 + rec0) if (prec0 + rec0) > 0 else 0.0
        
        # Serious (Class 2)
        tp1 = int(((actual == 2) & (pred == 2)).sum())
        fp1 = int(((actual != 2) & (pred == 2)).sum())
        fn1 = int(((actual == 2) & (pred != 2)).sum())
        rec1 = tp1 / actual_serious if actual_serious > 0 else 0.0
        prec1 = tp1 / (tp1 + fp1) if (tp1 + fp1) > 0 else 0.0
        f1_1 = (2 * prec1 * rec1) / (prec1 + rec1) if (prec1 + rec1) > 0 else 0.0
        
        # Slight (Class 3)
        tp2 = int(((actual == 3) & (pred == 3)).sum())
        fp2 = int(((actual != 3) & (pred == 3)).sum())
        fn2 = int(((actual == 3) & (pred != 3)).sum())
        rec2 = tp2 / actual_slight if actual_slight > 0 else 0.0
        prec2 = tp2 / (tp2 + fp2) if (tp2 + fp2) > 0 else 0.0
        f1_2 = (2 * prec2 * rec2) / (prec2 + rec2) if (prec2 + rec2) > 0 else 0.0
        
        macro_f1 = (f1_0 + f1_1 + f1_2) / 3.0
        weighted_f1 = (actual_fatal * f1_0 + actual_serious * f1_1 + actual_slight * f1_2) / total
        ff_rate = fp0 / non_fatal_total
        
        print(f"{tau:<6.2f} {acc*100:>6.2f}% {rec0*100:>7.2f}% {prec0*100:>8.2f}% {f1_0:>8.4f} {rec1*100:>7.2f}% {prec1*100:>8.2f}% {f1_1:>8.4f} {rec2*100:>7.2f}% {prec2*100:>8.2f}% {f1_2:>8.4f} {macro_f1:>9.4f} {weighted_f1:>11.4f} {fp0:>11,} {ff_rate*100:>7.2f}%")
        
        records.append({
            "threshold_tau": tau,
            "accuracy_pct": round(acc * 100, 2),
            "fatal_recall_pct": round(rec0 * 100, 2),
            "fatal_precision_pct": round(prec0 * 100, 2),
            "fatal_f1": round(f1_0, 4),
            "serious_recall_pct": round(rec1 * 100, 2),
            "serious_precision_pct": round(prec1 * 100, 2),
            "serious_f1": round(f1_1, 4),
            "slight_recall_pct": round(rec2 * 100, 2),
            "slight_precision_pct": round(prec2 * 100, 2),
            "slight_f1": round(f1_2, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "false_fatal_count": fp0,
            "false_fatal_rate_pct": round(ff_rate * 100, 2)
        })
        
    return pd.DataFrame(records)

if __name__ == "__main__":
    df_res = evaluate_thresholds(PRED_PATH)
    out_csv = PROJECT_ROOT / "reports" / "recalculated_threshold_analysis.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved recalculated threshold metrics to: {out_csv}")

