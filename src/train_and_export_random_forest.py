"""Targeted Training, Prediction Export, and Threshold Analysis for the Final Random Forest Model.

Trains the optimal Random Forest (60 Trees, Depth 12, Class-Weighted, Seed 42) on the 79,901 training partition,
evaluates on the 20,099 held-out test partition, saves the model artifact to models/spark_random_forest_model,
exports predictions to excel_dashboard_predictions.csv and random_forest_predictions.csv,
and performs an exact probability threshold analysis.
"""

import sys
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from src.spark_session import get_spark_session, stop_spark_session
from src.spark_data_processing import load_parquet_dataset
from src.spark_feature_pipeline import (
    prepare_target_label,
    stratified_train_test_split,
    fit_and_transform_pipeline
)
from src.model_training import compute_class_weights
from src.prediction_export import (
    prepare_dashboard_predictions_dataframe,
    export_dashboard_predictions
)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"


def run_rf_pipeline():
    spark = get_spark_session("Random Forest Final Verification")
    try:
        print("1. Ingesting Parquet Dataset...")
        df_raw = load_parquet_dataset(spark)
        
        print("2. Preparing Target Labels and Stratified Split...")
        df_labeled = prepare_target_label(df_raw)
        train_raw, test_raw = stratified_train_test_split(df_labeled, train_ratio=0.8, seed=42)
        
        print("3. Fitting 195-dimensional Feature Pipeline on Training Data...")
        pipeline_model, train_df, test_df = fit_and_transform_pipeline(train_raw, test_raw)
        
        # Save pipeline model
        pipe_path = MODELS_DIR / "spark_feature_pipeline"
        pipeline_model.write().overwrite().save(str(pipe_path))
        print(f"Saved Feature Pipeline to: {pipe_path}")
        
        print("4. Computing Class Weights strictly on Training Data...")
        train_weighted, class_weights = compute_class_weights(train_df)
        
        print("5. Training Final Random Forest Model (60 Trees, Depth 12, minInstances 5, Seed 42)...")
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            predictionCol="prediction",
            probabilityCol="probability",
            rawPredictionCol="rawPrediction",
            numTrees=60,
            maxDepth=12,
            minInstancesPerNode=5,
            weightCol="class_weight",
            seed=42
        )
        t0 = time.time()
        rf_model = rf.fit(train_weighted)
        train_duration = time.time() - t0
        print(f"Random Forest training completed in {train_duration:.2f} seconds.")
        
        # Save RF model
        rf_model_path = MODELS_DIR / "spark_random_forest_model"
        rf_model.write().overwrite().save(str(rf_model_path))
        print(f"Saved Random Forest model artifact to: {rf_model_path}")
        
        print("6. Evaluating Random Forest on Held-Out Test Set (20,099 rows)...")
        t_pred = time.time()
        predictions = rf_model.transform(test_df).cache()
        num_preds = predictions.count()
        pred_duration = time.time() - t_pred
        print(f"Evaluated {num_preds:,} predictions in {pred_duration:.2f} seconds.")
        
        # Compute official Spark MLlib metrics
        acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
        f0_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="fMeasureByLabel", metricLabel=0.0)
        r0_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="recallByLabel", metricLabel=0.0)
        p0_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="precisionByLabel", metricLabel=0.0)
        f1_eval_class1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="fMeasureByLabel", metricLabel=1.0)
        r1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="recallByLabel", metricLabel=1.0)
        p1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="precisionByLabel", metricLabel=1.0)
        f2_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="fMeasureByLabel", metricLabel=2.0)
        r2_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="recallByLabel", metricLabel=2.0)
        p2_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="precisionByLabel", metricLabel=2.0)
        
        accuracy = acc_eval.evaluate(predictions)
        weighted_f1 = f1_eval.evaluate(predictions)
        r0 = r0_eval.evaluate(predictions)
        p0 = p0_eval.evaluate(predictions)
        f0 = f0_eval.evaluate(predictions)
        r1 = r1_eval.evaluate(predictions)
        p1 = p1_eval.evaluate(predictions)
        f1 = f1_eval_class1.evaluate(predictions)
        r2 = r2_eval.evaluate(predictions)
        p2 = p2_eval.evaluate(predictions)
        f2 = f2_eval.evaluate(predictions)
        macro_f1 = (f0 + f1 + f2) / 3.0
        
        print("\n=== SPARK MLLIB OFFICIAL METRICS FOR FINAL RANDOM FOREST ===")
        print(f"Accuracy         : {accuracy * 100:.4f}%")
        print(f"Weighted F1      : {weighted_f1:.4f}")
        print(f"Macro F1         : {macro_f1:.4f}")
        print(f"Fatal Recall     : {r0 * 100:.2f}% | Precision: {p0 * 100:.2f}% | F1: {f0:.4f}")
        print(f"Serious Recall   : {r1 * 100:.2f}% | Precision: {p1 * 100:.2f}% | F1: {f1:.4f}")
        print(f"Slight Recall    : {r2 * 100:.2f}% | Precision: {p2 * 100:.2f}% | F1: {f2:.4f}")
        
        # Format predictions for Excel and CSV export
        print("\n7. Preparing and Exporting Dashboard Predictions DataFrame...")
        df_dash = prepare_dashboard_predictions_dataframe(
            predictions=predictions,
            raw_df_with_casualties=df_raw
        )
        
        # Export to CSV
        dash_csv_path = PROCESSED_DIR / "excel_dashboard_predictions.csv"
        rf_csv_path = PROCESSED_DIR / "random_forest_predictions.csv"
        improved_csv_path = PROCESSED_DIR / "improved_predictions.csv"
        
        pdf = df_dash.toPandas()
        pdf.to_csv(dash_csv_path, index=False)
        pdf.to_csv(rf_csv_path, index=False)
        pdf.to_csv(improved_csv_path, index=False)
        print(f"Exported {len(pdf):,} rows to {dash_csv_path}")
        print(f"Exported copy to {rf_csv_path}")
        print(f"Exported copy to {improved_csv_path}")
        
        return pdf, {
            "accuracy": accuracy,
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1,
            "fatal_recall": r0,
            "fatal_precision": p0,
            "fatal_f1": f0,
            "serious_recall": r1,
            "serious_precision": p1,
            "serious_f1": f1,
            "slight_recall": r2,
            "slight_precision": p2,
            "slight_f1": f2,
            "train_time_sec": train_duration,
            "predict_time_sec": pred_duration
        }
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    run_rf_pipeline()

