"""Main Pipeline Execution Script.

Orchestrates end-to-end STATS19 data ingestion, validation, cleaning,
feature engineering, stratified sampling, and export of modeling and Power BI datasets.

Usage:
    python main.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path for direct invocation
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    LOGS_DIR,
    PIPELINE_LOG_PATH,
    RANDOM_SEED,
    TARGET_SAMPLE_SIZE,
    TARGET_COLUMN,
    EXCLUDED_FEATURES_EXPLANATION,
    get_raw_dataset_path
)
from src.data_validation import load_raw_dataset, validate_dataset, print_validation_summary
from src.data_preprocessing import (
    remove_duplicates,
    clean_missing_and_invalid_data,
    drop_unnecessary_columns
)
from src.feature_engineering import engineer_features
from src.data_export import (
    select_final_records,
    export_processed_datasets,
    generate_preprocessing_report,
    generate_data_quality_report
)


def setup_logging():
    """Configure console and file logging handlers."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Root logger configuration
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates on rerun
    while logger.handlers:
        logger.handlers.pop()
        
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    fh = logging.FileHandler(PIPELINE_LOG_PATH, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger


def run_pipeline():
    """Execute the full data preparation pipeline."""
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("STARTING ROAD ACCIDENT SEVERITY PREPROCESSING PIPELINE (STATS19 2025)")
    logger.info("=" * 80)
    
    try:
        # STEP 1: Locate and Load Raw Dataset
        raw_csv_path = get_raw_dataset_path()
        logger.info("Step 1: Located raw dataset at: %s", raw_csv_path)
        raw_df = load_raw_dataset(raw_csv_path)
        raw_shape = raw_df.shape
        
        # STEP 2: Validate Dataset
        logger.info("Step 2: Performing schema verification and data integrity checks...")
        val_results = validate_dataset(raw_df)
        print_validation_summary(val_results)
        
        # STEP 3: Clean Data
        logger.info("Step 3: Removing duplicates and handling invalid values...")
        df_dedup, dups_removed = remove_duplicates(raw_df)
        df_clean_geo, clean_audit = clean_missing_and_invalid_data(df_dedup)
        df_cleaned, dropped_cols = drop_unnecessary_columns(df_clean_geo)
        cleaned_shape = df_cleaned.shape
        
        # STEP 4: Feature Engineering
        logger.info("Step 4: Engineering temporal, calendar, and environmental features...")
        df_engineered, created_features = engineer_features(df_cleaned)
        
        # STEP 5: Stratified Record Selection
        logger.info("Step 5: Selecting final records (Target: %d, Seed: %d)...", TARGET_SAMPLE_SIZE, RANDOM_SEED)
        df_final = select_final_records(
            df_engineered,
            target_size=TARGET_SAMPLE_SIZE,
            random_seed=RANDOM_SEED
        )
        final_shape = df_final.shape
        
        # STEP 6: Export Processed Datasets
        logger.info("Step 6: Exporting processed CSV and Parquet files...")
        exported_paths = export_processed_datasets(df_engineered, df_final)
        
        # STEP 7: Generate Data Quality and Preprocessing Reports
        logger.info("Step 7: Compiling formal academic text reports...")
        target_distribution = df_final[TARGET_COLUMN].value_counts().to_dict()
        
        prep_report_path = generate_preprocessing_report(
            raw_shape=raw_shape,
            cleaned_shape=cleaned_shape,
            final_shape=final_shape,
            duplicates_removed=dups_removed,
            missing_handled=clean_audit,
            features_created=created_features,
            columns_excluded=EXCLUDED_FEATURES_EXPLANATION,
            target_distribution=target_distribution
        )
        
        quality_report_path = generate_data_quality_report(
            validation_results=val_results,
            clean_audit=clean_audit,
            final_df=df_final
        )
        
        # FINAL SUMMARY
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        print("\n" + "=" * 80)
        print("                        PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Original Records       : {raw_shape[0]:,} rows across {raw_shape[1]} columns")
        print(f"Cleaned Records        : {cleaned_shape[0]:,} rows across {cleaned_shape[1]} columns")
        print(f"Final Selected Records : {final_shape[0]:,} rows across {final_shape[1]} columns")
        print("-" * 80)
        print("Final Target Distribution (collision_severity):")
        total_final = len(df_final)
        for cat in sorted(target_distribution.keys()):
            cnt = target_distribution[cat]
            label = {1: "Fatal", 2: "Serious", 3: "Slight"}.get(cat, "Other")
            print(f"  Class {cat} ({label:<7}): {cnt:>7,} rows ({cnt/total_final*100:6.2f}%)")
        print("-" * 80)
        print("Generated Files:")
        for name, p in exported_paths.items():
            print(f"  - [{name:<18}]: {p}")
        print(f"  - [preprocessing_report]: {prep_report_path}")
        print(f"  - [data_quality_report ]: {quality_report_path}")
        print(f"  - [execution_log       ]: {PIPELINE_LOG_PATH}")
        print("=" * 80)
        
    except Exception as e:
        logger.exception("FATAL PIPELINE ERROR: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()

