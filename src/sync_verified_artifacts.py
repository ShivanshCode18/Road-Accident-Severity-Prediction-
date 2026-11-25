"""Synchronize Verified Artifacts and Update Excel Dashboard.

Parses verified CSV reports from the latest PySpark run, populates final_model_metrics.json,
updates model_performance_metrics.csv, and stream-updates the Excel dashboard.
"""

import sys
import json
import logging
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
EXCEL_PATH = PROJECT_ROOT / "dataset" / "excel" / "Road_Accident_Severity_Professional_Dashboard.xlsx"

COMP_CSV = REPORTS_DIR / "improved_model_comparison.csv"
CONF_CSV = REPORTS_DIR / "improved_confusion_matrices.csv"
MODEL_PERF_PATH = PROCESSED_DIR / "model_performance_metrics.csv"
SUMMARY_JSON_PATH = MODELS_DIR / "evaluation_summary.json"
FINAL_METRICS_PATH = MODELS_DIR / "final_model_metrics.json"
PREDICTIONS_CSV_PATH = PROCESSED_DIR / "excel_dashboard_predictions.csv"

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("sync_verified_artifacts")


def build_evaluations_dict():
    """Construct complete evaluations dictionary from improved CSV deliverables."""
    df_comp = pd.read_csv(COMP_CSV)
    df_conf = pd.read_csv(CONF_CSV)
    
    evaluations = {}
    
    for _, row in df_comp.iterrows():
        mname = row["model_name"]
        
        # Build confusion matrix for this model
        sub_cm = df_conf[df_conf["model_name"] == mname]
        cm_dict = {}
        for _, cm_row in sub_cm.iterrows():
            actual = cm_row["actual_class"]
            cm_dict[actual] = {
                "Fatal": int(cm_row["pred_fatal"]),
                "Serious": int(cm_row["pred_serious"]),
                "Slight": int(cm_row["pred_slight"])
            }
            
        per_class = {
            "Fatal": {
                "label": 0.0,
                "precision": float(row["fatal_precision"]),
                "recall": float(row["fatal_recall_pct"]) / 100.0,
                "f1": float(row["fatal_f1"])
            },
            "Serious": {
                "label": 1.0,
                "precision": float(row["serious_precision"]),
                "recall": float(row["serious_recall_pct"]) / 100.0,
                "f1": float(row["serious_f1"])
            },
            "Slight": {
                "label": 2.0,
                "precision": float(row["slight_precision"]),
                "recall": float(row["slight_recall_pct"]) / 100.0,
                "f1": float(row["slight_f1"])
            }
        }
        
        evaluations[mname] = {
            "model_name": mname,
            "type": row.get("model_type", mname),
            "accuracy": float(row["accuracy_pct"]) / 100.0,
            "weighted_precision": float(row["weighted_precision"]),
            "weighted_recall": float(row["weighted_recall"]),
            "weighted_f1": float(row["weighted_f1"]),
            "macro_precision": float(row["macro_precision"]),
            "macro_recall": float(row["macro_recall"]),
            "macro_f1": float(row["macro_f1"]),
            "fatal_recall": float(row["fatal_recall_pct"]) / 100.0,
            "per_class": per_class,
            "confusion_matrix": cm_dict,
            "train_time_sec": float(row.get("training_time_sec", 0.0)),
            "predict_time_sec": float(row.get("prediction_time_sec", 0.0))
        }
        
    return evaluations


def sync_all():
    logger.info("Building verified evaluations from CSV reports...")
    evaluations = build_evaluations_dict()
    
    # 1. Update final_model_metrics.json and evaluation_summary.json
    best_model_name = "Random Forest"
    serializable_summary = {
        "best_model": best_model_name,
        "models": evaluations
    }
    
    with open(FINAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable_summary, f, indent=2)
    logger.info("Saved verified metrics to %s", FINAL_METRICS_PATH)
    
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(serializable_summary, f, indent=2)
    logger.info("Updated %s", SUMMARY_JSON_PATH)
    
    # 2. Update model_performance_metrics.csv for primary models
    primary_models = ["Logistic Regression", "Decision Tree", "Random Forest"]
    rows = []
    for name in primary_models:
        if name in evaluations:
            ev = evaluations[name]
            rows.append({
                "model_name": name,
                "model_type": ev["type"],
                "accuracy": round(ev["accuracy"] * 100, 2),
                "weighted_precision": round(ev["weighted_precision"], 4),
                "weighted_recall": round(ev["weighted_recall"], 4),
                "weighted_f1": round(ev["weighted_f1"], 4),
                "macro_f1": round(ev["macro_f1"], 4),
                "fatal_recall": round(ev["fatal_recall"] * 100, 2),
                "fatal_precision": round(ev["per_class"]["Fatal"]["precision"], 4),
                "fatal_f1": round(ev["per_class"]["Fatal"]["f1"], 4),
                "serious_recall": round(ev["per_class"]["Serious"]["recall"] * 100, 2),
                "serious_f1": round(ev["per_class"]["Serious"]["f1"], 4),
                "slight_recall": round(ev["per_class"]["Slight"]["recall"] * 100, 2),
                "slight_f1": round(ev["per_class"]["Slight"]["f1"], 4),
                "train_time_sec": round(ev.get("train_time_sec", 0.0), 2)
            })
    pd.DataFrame(rows).to_csv(MODEL_PERF_PATH, index=False)
    logger.info("Updated %s with 3 primary verified models.", MODEL_PERF_PATH)
    
    # 3. Synchronize Excel Dashboard Workbook
    from src.excel_dashboard_updater import update_excel_workbook
    logger.info("Synchronizing Excel Dashboard: %s", EXCEL_PATH)
    update_excel_workbook(
        excel_path=EXCEL_PATH,
        evaluations=evaluations,
        predictions_csv=PREDICTIONS_CSV_PATH
    )
    logger.info("Excel Dashboard workbook synchronized successfully.")


if __name__ == "__main__":
    sync_all()
