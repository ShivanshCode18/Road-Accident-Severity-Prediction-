"""Data Validation Module for STATS19 Collisions Dataset.

Performs schema verification, data integrity checks, duplicate audits,
missing value calculations, and target distribution validation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
from .config import TARGET_COLUMN, ID_COLUMNS, get_raw_dataset_path

logger = logging.getLogger(__name__)


def load_raw_dataset(file_path: Path = None) -> pd.DataFrame:
    """Load the raw STATS19 CSV file ensuring ID columns remain strings.
    
    Args:
        file_path: Path to the raw CSV file. If None, dynamically resolved.
        
    Returns:
        pd.DataFrame: Loaded raw dataset.
    """
    if file_path is None:
        file_path = get_raw_dataset_path()
        
    logger.info("Loading raw dataset from: %s", file_path)
    
    # Read identifier columns as strings to prevent scientific notation or precision loss
    dtype_dict = {col: str for col in ID_COLUMNS}
    
    df = pd.read_csv(file_path, dtype=dtype_dict, low_memory=False)
    logger.info("Raw dataset loaded successfully with shape: %s", df.shape)
    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Execute comprehensive validation checks on the dataset.
    
    Args:
        df: Raw or loaded DataFrame.
        
    Returns:
        Dict[str, Any]: Detailed dictionary of validation results.
        
    Raises:
        ValueError: If required target column is missing.
    """
    logger.info("Starting dataset validation...")
    
    # Check target column existence
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"CRITICAL ERROR: Target column '{TARGET_COLUMN}' is missing from the dataset! "
            f"Available columns: {list(df.columns)}"
        )
        
    total_records = len(df)
    total_columns = len(df.columns)
    
    # 1. Column Names and Types
    dtypes_summary = df.dtypes.astype(str).to_dict()
    
    # 2. Missing values per column
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / total_records) * 100
    missing_summary = {
        col: {"count": int(missing_counts[col]), "pct": float(missing_pcts[col])}
        for col in df.columns
    }
    
    # 3. Duplicate checks
    exact_duplicates = int(df.duplicated().sum())
    
    has_collision_index = "collision_index" in df.columns
    index_duplicates = int(df.duplicated(subset=["collision_index"]).sum()) if has_collision_index else None
    unique_collision_indices = int(df["collision_index"].nunique()) if has_collision_index else None
    
    # 4. Collision Year validation
    has_collision_year = "collision_year" in df.columns
    year_distribution = df["collision_year"].value_counts().to_dict() if has_collision_year else {}
    
    # 5. Target variable distribution
    target_counts = df[TARGET_COLUMN].value_counts(dropna=False).to_dict()
    target_pcts = (df[TARGET_COLUMN].value_counts(normalize=True, dropna=False) * 100).to_dict()
    
    # 6. Numerical vs Categorical feature identification
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    validation_results = {
        "total_records": total_records,
        "total_columns": total_columns,
        "column_names": list(df.columns),
        "dtypes": dtypes_summary,
        "numerical_features": numerical_cols,
        "categorical_features": categorical_cols,
        "missing_summary": missing_summary,
        "exact_duplicates": exact_duplicates,
        "has_collision_index": has_collision_index,
        "unique_collision_indices": unique_collision_indices,
        "index_duplicates": index_duplicates,
        "collision_year_distribution": year_distribution,
        "target_counts": target_counts,
        "target_pcts": target_pcts,
        "head_5": df.head(5).to_dict(orient="records"),
        "tail_5": df.tail(5).to_dict(orient="records")
    }
    
    logger.info("Validation completed. Total Records: %d, Total Columns: %d", total_records, total_columns)
    logger.info("Exact Duplicates: %d, Collision Index Duplicates: %s", exact_duplicates, index_duplicates)
    logger.info("Target Variable '%s' Distribution: %s", TARGET_COLUMN, target_counts)
    
    return validation_results


def print_validation_summary(results: Dict[str, Any]) -> None:
    """Print an academic, human-readable summary of validation checks."""
    print("=" * 80)
    print("                     STATS19 DATASET VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total Number of Records  : {results['total_records']:,}")
    print(f"Total Number of Columns  : {results['total_columns']}")
    print(f"Exact Duplicate Records  : {results['exact_duplicates']}")
    if results['has_collision_index']:
        print(f"Unique Collision Indices : {results['unique_collision_indices']:,}")
        print(f"Duplicate Collision IDs  : {results['index_duplicates']}")
    print(f"Collision Year Values    : {results['collision_year_distribution']}")
    print("-" * 80)
    print(f"Target Column: '{TARGET_COLUMN}' Distribution:")
    for cat, count in results['target_counts'].items():
        pct = results['target_pcts'].get(cat, 0.0)
        label = {1: "Fatal", 2: "Serious", 3: "Slight"}.get(cat, "Other")
        print(f"  Class {cat} ({label:<7}): {count:>7,} rows ({pct:6.2f}%)")
    print("-" * 80)
    print(f"Numerical Features ({len(results['numerical_features'])}): {results['numerical_features'][:8]} ...")
    print(f"Categorical Features ({len(results['categorical_features'])}): {results['categorical_features']}")
    
    # Columns with missing values
    missing_cols = {col: data for col, data in results['missing_summary'].items() if data['count'] > 0}
    print("-" * 80)
    if missing_cols:
        print(f"Columns with Missing (NaN) Values ({len(missing_cols)}):")
        for col, data in missing_cols.items():
            print(f"  - {col}: {data['count']:,} missing ({data['pct']:.4f}%)")
    else:
        print("No NaN missing values found in raw dataset.")
    print("=" * 80)

