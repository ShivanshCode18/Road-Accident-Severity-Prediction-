"""Prediction Export and Artifact Persistence Module for Excel Dashboard Integration.

Saves trained Spark MLlib models, extracts class probability vectors into columns,
joins dashboard-only KPI fields (casualties), and exports ready-to-visualize CSV artifacts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
EXCEL_PREDICTIONS_PATH = PROCESSED_DIR / "excel_dashboard_predictions.csv"
MODEL_PERFORMANCE_PATH = PROCESSED_DIR / "model_performance_metrics.csv"
EXCEL_DATA_PATH = PROCESSED_DIR / "excel_dashboard_dataset.csv"

# Backward compatibility aliases
POWERBI_PREDICTIONS_PATH = EXCEL_PREDICTIONS_PATH
POWERBI_PERFORMANCE_PATH = MODEL_PERFORMANCE_PATH
POWERBI_DATA_PATH = EXCEL_DATA_PATH


def extract_probability_columns(predictions: DataFrame) -> DataFrame:
    """Extract probability vector elements into distinct prob_fatal, prob_serious, prob_slight columns."""
    logger.info("Extracting class probability distributions...")
    
    # Try native vector_to_array if supported, else fallback to safe UDF
    try:
        from pyspark.ml.functions import vector_to_array
        df_prob = predictions.withColumn("prob_arr", vector_to_array("probability"))
        df_prob = df_prob.withColumn("prob_fatal", F.round(F.col("prob_arr")[0], 4)) \
                         .withColumn("prob_serious", F.round(F.col("prob_arr")[1], 4)) \
                         .withColumn("prob_slight", F.round(F.col("prob_arr")[2], 4)) \
                         .drop("prob_arr")
        return df_prob
    except Exception as e:
        logger.info("Using UDF for vector probability extraction: %s", e)
        get_prob = F.udf(lambda v, i: float(v[i]) if v is not None and len(v) > i else 0.0, DoubleType())
        
        df_prob = predictions.withColumn("prob_fatal", F.round(get_prob(F.col("probability"), F.lit(0)), 4)) \
                             .withColumn("prob_serious", F.round(get_prob(F.col("probability"), F.lit(1)), 4)) \
                             .withColumn("prob_slight", F.round(get_prob(F.col("probability"), F.lit(2)), 4))
        return df_prob


def prepare_dashboard_predictions_dataframe(
    predictions: DataFrame,
    raw_df_with_casualties: DataFrame = None
) -> DataFrame:
    """Format predictions DataFrame for Excel dashboard analysis and join casualty KPIs.
    
    Args:
        predictions: Model predictions DataFrame.
        raw_df_with_casualties: DataFrame containing collision_index and number_of_casualties.
        
    Returns:
        DataFrame: Curated dashboard predictions DataFrame.
    """
    logger.info("Preparing dashboard predictions DataFrame...")
    
    # 1. Extract class probabilities
    df_pbi = extract_probability_columns(predictions)
    
    # 2. Add human-readable and standard numeric predicted severity codes
    # Model label: 0.0 -> Severity 1 (Fatal), 1.0 -> Severity 2 (Serious), 2.0 -> Severity 3 (Slight)
    df_pbi = df_pbi.withColumn(
        "predicted_severity",
        F.when(F.col("prediction") == 0.0, 1)
         .when(F.col("prediction") == 1.0, 2)
         .when(F.col("prediction") == 2.0, 3)
         .otherwise(3)
    )
    
    df_pbi = df_pbi.withColumn(
        "predicted_severity_name",
        F.when(F.col("prediction") == 0.0, "Fatal")
         .when(F.col("prediction") == 1.0, "Serious")
         .when(F.col("prediction") == 2.0, "Slight")
         .otherwise("Slight")
    )
    
    df_pbi = df_pbi.withColumn(
        "actual_severity",
        F.col("collision_severity").cast("int")
    )
    
    df_pbi = df_pbi.withColumn(
        "actual_severity_name",
        F.when(F.col("actual_severity") == 1, "Fatal")
         .when(F.col("actual_severity") == 2, "Serious")
         .when(F.col("actual_severity") == 3, "Slight")
         .otherwise("Unknown")
    )
    
    df_pbi = df_pbi.withColumn(
        "prediction_correct",
        F.when(F.col("actual_severity") == F.col("predicted_severity"), 1).otherwise(0)
    )
    
    # 3. Ensure number_of_casualties is present for dashboard KPI filtering
    if "number_of_casualties" not in df_pbi.columns:
        if raw_df_with_casualties is not None and "number_of_casualties" in raw_df_with_casualties.columns:
            cas_df = raw_df_with_casualties.select("collision_index", "number_of_casualties")
            df_pbi = df_pbi.join(cas_df, on="collision_index", how="left")
        else:
            df_pbi = df_pbi.withColumn("number_of_casualties", F.lit(1))
        
    # 4. Curate final column subset for clean dashboard reporting
    desired_columns = [
        "collision_index",
        "date",
        "time",
        "hour",
        "time_of_day",
        "day_name",
        "month_name",
        "actual_severity",
        "actual_severity_name",
        "predicted_severity",
        "predicted_severity_name",
        "prediction_correct",
        "prob_fatal",
        "prob_serious",
        "prob_slight",
        "number_of_vehicles",
        "number_of_casualties",
        "speed_limit",
        "speed_zone",
        "road_type_name",
        "urban_rural_name",
        "weather_group",
        "road_surface_group",
        "light_group",
        "latitude",
        "longitude"
    ]
    
    available_cols = [c for c in desired_columns if c in df_pbi.columns]
    return df_pbi.select(available_cols)


# Backward compatibility alias
prepare_powerbi_predictions_dataframe = prepare_dashboard_predictions_dataframe


def export_dashboard_predictions(df_dash: DataFrame) -> Path:
    """Export predictions DataFrame to CSV for Excel dashboard."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting dashboard predictions to: %s...", EXCEL_PREDICTIONS_PATH)
    
    # Convert to Pandas for single consolidated CSV file output
    pdf = df_dash.toPandas()
    pdf.to_csv(EXCEL_PREDICTIONS_PATH, index=False)
    logger.info("Exported %d rows to %s", len(pdf), EXCEL_PREDICTIONS_PATH)
    
    # Also export improved_predictions.csv
    improved_pred_path = PROCESSED_DIR / "improved_predictions.csv"
    pdf.to_csv(improved_pred_path, index=False)
    logger.info("Exported improved predictions to %s", improved_pred_path)
    return EXCEL_PREDICTIONS_PATH


# Backward compatibility alias
export_powerbi_predictions = export_dashboard_predictions


def export_model_performance_table(evaluations: Dict[str, Dict[str, Any]]) -> Path:
    """Export summary metrics table for the 3 primary models to CSV for Excel dashboard visualization."""
    import pandas as pd
    
    primary_models = ["Logistic Regression", "Decision Tree", "Random Forest"]
    rows = []
    
    for name in primary_models:
        if name in evaluations:
            ev = evaluations[name]
            row = {
                "model_name": name,
                "model_type": ev.get("type", name),
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
            }
            rows.append(row)
        
    pdf = pd.DataFrame(rows)
    pdf.to_csv(MODEL_PERFORMANCE_PATH, index=False)
    logger.info("Exported model performance comparison to: %s", MODEL_PERFORMANCE_PATH)
    return MODEL_PERFORMANCE_PATH


def save_model_artifacts(
    pipeline_model,
    best_model,
    evaluations: Dict[str, Any],
    models_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Path]:
    """Persist trained pipeline model, best classifier, and evaluation summary."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    
    # Save Pipeline Model
    pipeline_path = MODELS_DIR / "spark_feature_pipeline"
    pipeline_model.write().overwrite().save(str(pipeline_path))
    paths["pipeline_model"] = pipeline_path
    logger.info("Saved Spark Feature PipelineModel to: %s", pipeline_path)
    
    # Save Best Model
    best_model_name = evaluations.get("_best_model", "best_model")
    clean_name = best_model_name.lower().replace(" ", "_")
    model_path = MODELS_DIR / f"spark_{clean_name}_model"
    best_model.write().overwrite().save(str(model_path))
    paths["best_model"] = model_path
    logger.info("Saved Best Model (%s) to: %s", best_model_name, model_path)
    
    # Save other primary models for test suite validation
    if models_dict:
        if "Logistic Regression" in models_dict and best_model_name != "Logistic Regression":
            lr_path = MODELS_DIR / "spark_logistic_regression_model"
            models_dict["Logistic Regression"]["model"].write().overwrite().save(str(lr_path))
            paths["logistic_regression_model"] = lr_path
            logger.info("Saved Logistic Regression model to: %s", lr_path)
        if "Random Forest" in models_dict and best_model_name != "Random Forest":
            rf_path = MODELS_DIR / "spark_random_forest_model"
            models_dict["Random Forest"]["model"].write().overwrite().save(str(rf_path))
            paths["random_forest_model"] = rf_path
            logger.info("Saved Random Forest model to: %s", rf_path)
            
    # Save Evaluation JSON Summary
    summary_path = MODELS_DIR / "evaluation_summary.json"
    final_metrics_path = MODELS_DIR / "final_model_metrics.json"
    
    # Prepare serializable dict
    serializable_summary = {
        "best_model": best_model_name,
        "models": {}
    }
    for mname in [k for k in evaluations.keys() if not k.startswith("_")]:
        ev = evaluations[mname]
        serializable_summary["models"][mname] = {
            "accuracy": ev["accuracy"],
            "weighted_precision": ev.get("weighted_precision", 0.0),
            "weighted_recall": ev.get("weighted_recall", 0.0),
            "weighted_f1": ev["weighted_f1"],
            "macro_precision": ev.get("macro_precision", 0.0),
            "macro_recall": ev.get("macro_recall", 0.0),
            "macro_f1": ev["macro_f1"],
            "fatal_recall": ev["fatal_recall"],
            "per_class": ev["per_class"],
            "confusion_matrix": ev["confusion_matrix"],
            "train_time_sec": ev.get("train_time_sec", 0.0),
            "predict_time_sec": ev.get("predict_time_sec", 0.0)
        }
        
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(serializable_summary, f, indent=2)
        
    with open(final_metrics_path, "w", encoding="utf-8") as f:
        json.dump(serializable_summary, f, indent=2)
        
    paths["summary_json"] = summary_path
    paths["final_metrics_json"] = final_metrics_path
    logger.info("Saved Evaluation JSON Summary to: %s and %s", summary_path, final_metrics_path)
    return paths
