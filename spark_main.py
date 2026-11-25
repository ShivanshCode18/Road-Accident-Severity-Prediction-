"""Apache Spark Machine Learning Pipeline Driver Script (Phase 2).

Orchestrates end-to-end PySpark data processing, MLlib feature indexing,
stratified train-test partitioning, multiclass model training (Logistic Regression,
Decision Tree, Random Forest), diagnostic evaluation, artifact saving, and Excel dashboard data export.

Usage:
    python spark_main.py
"""

import sys
import logging
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.spark_session import (
    get_spark_session,
    stop_spark_session,
    display_dataframe_summary
)
from src.spark_data_processing import (
    load_parquet_dataset,
    audit_spark_dataset,
    verify_against_phase1,
    generate_spark_processing_report
)
from src.spark_feature_pipeline import (
    prepare_target_label,
    stratified_train_test_split,
    fit_and_transform_pipeline
)
from src.model_training import train_all_models
from src.model_evaluation import (
    evaluate_all_models,
    generate_model_evaluation_report
)
from src.prediction_export import (
    prepare_dashboard_predictions_dataframe,
    export_dashboard_predictions,
    export_model_performance_table,
    save_model_artifacts
)
from src.excel_dashboard_updater import (
    update_excel_workbook,
    EXCEL_WORKBOOK_PATH
)

LOG_FILE = BASE_DIR / "logs" / "spark_pipeline.log"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"


def setup_logging():
    """Configure console and file logging handlers."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    while logger.handlers:
        logger.handlers.pop()
        
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger


def run_spark_pipeline():
    """Execute the complete PySpark machine learning pipeline."""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("STARTING PYSPARK MACHINE LEARNING PIPELINE (PHASE 2)")
    logger.info("=" * 80)
    
    spark = None
    try:
        # STEP 1: Initialize SparkSession
        logger.info("Step 1: Initializing Apache Spark Session...")
        spark = get_spark_session("Road Accident Severity Prediction")
        
        # STEP 2: Load Parquet Dataset
        logger.info("Step 2: Loading accident_final_100k.parquet into PySpark DataFrame...")
        df_raw = load_parquet_dataset(spark)
        display_dataframe_summary(df_raw, name="Ingested Parquet Dataset", num_rows=3)
        
        # STEP 3: Data Processing & Phase 1 Conformance Audit
        logger.info("Step 3: Auditing dataset schema, nulls, and duplicates in PySpark...")
        audit_results = audit_spark_dataset(df_raw)
        verified, verif_msg = verify_against_phase1(audit_results)
        
        if not verified:
            raise ValueError(f"Phase 1 Conformance Verification Failed: {verif_msg}")
            
        spark_report_path = generate_spark_processing_report(audit_results, verif_msg)
        logger.info("Generated Spark Processing Report: %s", spark_report_path)
        
        # STEP 4: Target Label Preparation & Stratified Split
        logger.info("Step 4: Mapping target collision_severity to zero-indexed MLlib label...")
        df_labeled = prepare_target_label(df_raw)
        
        logger.info("Step 5: Executing stratified 80/20 train-test partition...")
        train_raw, test_raw = stratified_train_test_split(df_labeled, train_ratio=0.8, seed=42)
        
        # STEP 5: Feature Pipeline Fitting (On Training Data Only)
        logger.info("Step 6: Fitting feature indexing and assembly pipeline on training data...")
        pipeline_model, train_df, test_df = fit_and_transform_pipeline(train_raw, test_raw)
        
        # STEP 6: Multiclass Model Training
        logger.info("Step 7: Training Multiclass Classifiers (Logistic Regression, Decision Tree, Random Forest)...")
        models_dict = train_all_models(train_df)
        
        # STEP 7: Comprehensive Model Evaluation on Untouched Test Set
        logger.info("Step 8: Evaluating all models on the untouched test set (20,000 rows)...")
        evaluations = evaluate_all_models(models_dict, test_df)
        
        eval_report_path = generate_model_evaluation_report(evaluations)
        logger.info("Generated Model Evaluation Report: %s", eval_report_path)
        
        # STEP 8: Model Artifact Persistence
        best_model_name = evaluations["_best_model"]
        best_model = models_dict[best_model_name]["model"]
        
        logger.info("Step 9: Persisting feature pipeline and best model (%s) artifacts...", best_model_name)
        saved_paths = save_model_artifacts(pipeline_model, best_model, evaluations, models_dict=models_dict)
        
        # STEP 9: Dashboard Predictions and Performance Export
        logger.info("Step 10: Generating Excel dashboard predictions CSV and model performance table...")
        best_predictions = evaluations[best_model_name]["predictions_df"]
        
        df_dash = prepare_dashboard_predictions_dataframe(
            predictions=best_predictions,
            raw_df_with_casualties=df_raw
        )
        
        dash_pred_path = export_dashboard_predictions(df_dash)
        model_perf_path = export_model_performance_table(evaluations)
        
        # STEP 10: Synchronize Excel Dashboard Workbook (Sheets 4 & 5)
        logger.info("Step 11: Synchronizing Excel Dashboard Workbook (%s)...", EXCEL_WORKBOOK_PATH)
        excel_status = update_excel_workbook(
            excel_path=EXCEL_WORKBOOK_PATH,
            evaluations=evaluations,
            predictions_csv_path=dash_pred_path
        )
        logger.info("Excel Dashboard synchronization completed: %s", excel_status)
        
        # FINAL CONSOLE SUMMARY
        logger.info("=" * 80)
        logger.info("PYSPARK PHASE 2 & IMPROVEMENT PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        print("\n" + "=" * 90)
        print("          PYSPARK MACHINE LEARNING PIPELINE & MODEL OPTIMIZATION SUMMARY")
        print("=" * 90)
        print(f"Total Records Ingested   : {audit_results['total_records']:,} rows")
        print(f"Training Partition (80%) : {train_raw.count():,} rows")
        print(f"Testing Partition (20%)  : {test_raw.count():,} rows (Strictly Disjoint & Untouched)")
        print("-" * 90)
        print("MULTICLASS MODEL BENCHMARK (UNTOUCHED TEST DATASET: 20,099 ROWS):")
        print(f"{'Model Name':<28} {'Accuracy':>9} {'Weighted F1':>12} {'Macro F1':>9} {'Fatal Rec':>10} {'Fatal Prec':>11}")
        print("-" * 90)
        for mname in [k for k in evaluations.keys() if not k.startswith("_")]:
            ev = evaluations[mname]
            fprec = ev['per_class']['Fatal']['precision'] * 100
            print(f"{mname:<28} {ev['accuracy']*100:>8.2f}% {ev['weighted_f1']:>12.4f} {ev['macro_f1']:>9.4f} {ev['fatal_recall']*100:>9.2f}% {fprec:>10.2f}%")
        print("-" * 90)
        print("ORIGINAL VS. IMPROVED PERFORMANCE (RANDOM FOREST):")
        print(f"{'Metric':<25} {'Original Run':>15} {'Improved Run':>15} {'Absolute Gain':>15}")
        print("-" * 75)
        rf_ev = evaluations.get("Random Forest", {})
        rf_acc = rf_ev.get("accuracy", 0) * 100
        rf_wf1 = rf_ev.get("weighted_f1", 0)
        rf_mf1 = rf_ev.get("macro_f1", 0)
        rf_fprec = rf_ev.get("per_class", {}).get("Fatal", {}).get("precision", 0) * 100
        rf_ff1 = rf_ev.get("per_class", {}).get("Fatal", {}).get("f1", 0)
        
        acc_diff = rf_acc - 54.50
        wf1_diff = rf_wf1 - 0.5985
        mf1_diff = rf_mf1 - 0.3667
        fprec_diff = rf_fprec - 4.39
        ff1_diff = rf_ff1 - 0.0814
        
        acc_diff_str = f"+{acc_diff:.2f}%" if acc_diff >= 0 else f"{acc_diff:.2f}%"
        wf1_diff_str = f"+{wf1_diff:.4f}" if wf1_diff >= 0 else f"{wf1_diff:.4f}"
        mf1_diff_str = f"+{mf1_diff:.4f}" if mf1_diff >= 0 else f"{mf1_diff:.4f}"
        fprec_diff_str = f"+{fprec_diff:.2f}%" if fprec_diff >= 0 else f"{fprec_diff:.2f}%"
        ff1_diff_str = f"+{ff1_diff:.4f}" if ff1_diff >= 0 else f"{ff1_diff:.4f}"

        print(f"{'Accuracy':<25} {'54.50%':>15} {rf_acc:>14.2f}% {acc_diff_str:>15}")
        print(f"{'Weighted F1':<25} {'0.5985':>15} {rf_wf1:>15.4f} {wf1_diff_str:>15}")
        print(f"{'Macro F1':<25} {'0.3667':>15} {rf_mf1:>15.4f} {mf1_diff_str:>15}")
        print(f"{'Fatal Precision':<25} {'4.39%':>15} {rf_fprec:>14.2f}% {fprec_diff_str:>15}")
        print(f"{'Fatal F1-Score':<25} {'0.0814':>15} {rf_ff1:>15.4f} {ff1_diff_str:>15}")
        print("-" * 75)
        print(f"Selected Best Model      : {best_model_name.upper()} (Optimal Macro F1 & Safety Balance)")
        print("-" * 90)
        print("Generated Deliverables & Reports:")
        print(f"  - [Spark Processing Report     ]: {spark_report_path}")
        print(f"  - [Model Evaluation Report     ]: {eval_report_path}")
        print(f"  - [Model Improvement Report    ]: {REPORTS_DIR / 'model_improvement_report.txt'}")
        print(f"  - [Feature Engineering Report  ]: {REPORTS_DIR / 'feature_engineering_report.txt'}")
        print(f"  - [Improved Model Comparison   ]: {REPORTS_DIR / 'improved_model_comparison.csv'}")
        print(f"  - [Improved Class Report       ]: {REPORTS_DIR / 'improved_classification_report.csv'}")
        print(f"  - [Improved Confusion Matrices ]: {REPORTS_DIR / 'improved_confusion_matrices.csv'}")
        print(f"  - [Hyperparameter Tuning CSV   ]: {REPORTS_DIR / 'hyperparameter_tuning_results.csv'}")
        print(f"  - [Dashboard Predictions CSV   ]: {dash_pred_path}")
        print(f"  - [Model Performance Table CSV ]: {model_perf_path}")
        print(f"  - [Final Metrics JSON          ]: {MODELS_DIR / 'final_model_metrics.json'}")
        print(f"  - [Synchronized Excel Workbook ]: {EXCEL_WORKBOOK_PATH}")
        print(f"  - [Saved Pipeline Artifact     ]: {saved_paths.get('pipeline_model')}")
        print(f"  - [Saved Best Model Artifact   ]: {saved_paths.get('best_model')}")
        print(f"  - [Pipeline Execution Log      ]: {LOG_FILE}")
        print("=" * 90 + "\n")
        
    except Exception as e:
        logger.exception("FATAL PYSPARK PIPELINE ERROR: %s", e)
        sys.exit(1)
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    run_spark_pipeline()

