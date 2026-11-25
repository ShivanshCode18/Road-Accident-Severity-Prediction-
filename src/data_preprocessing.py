"""Data Preprocessing Module for STATS19 Collisions Dataset.

Handles duplicate filtering, missing coordinate elimination, domain validation
of geographic coordinates and speed limits, column filtering with leakage prevention,
and categorical code consistency.
"""

import logging
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from .config import (
    TARGET_COLUMN,
    ID_COLUMNS,
    UK_BOUNDS,
    PERMANENTLY_DROPPED_COLUMNS,
    EXCLUDED_FEATURES_EXPLANATION
)

logger = logging.getLogger(__name__)


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Identify and remove exact duplicate rows while auditing collision identifiers.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Tuple[pd.DataFrame, int]: Deduplicated DataFrame and count of removed duplicates.
    """
    initial_count = len(df)
    exact_dups = int(df.duplicated().sum())
    
    if exact_dups > 0:
        logger.warning("Found %d exact duplicate rows. Removing duplicates...", exact_dups)
        df_cleaned = df.drop_duplicates().copy()
    else:
        logger.info("Zero exact duplicate rows detected.")
        df_cleaned = df.copy()
        
    # Check uniqueness of collision_index
    if "collision_index" in df_cleaned.columns:
        index_dups = int(df_cleaned.duplicated(subset=["collision_index"]).sum())
        if index_dups > 0:
            logger.warning("Found %d duplicate collision_index records. Investigating...", index_dups)
        else:
            logger.info("All collision_index identifiers are strictly unique.")
            
    removed_count = initial_count - len(df_cleaned)
    return df_cleaned, removed_count


def clean_missing_and_invalid_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Filter records with invalid coordinates, speed limits, or target severities.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned DataFrame and quality audit metrics.
    """
    logger.info("Auditing missing and invalid records...")
    initial_rows = len(df)
    audit = {}
    
    # 1. Coordinate Validation (Drop NaN or outside UK bounding box)
    missing_coords = df["latitude"].isnull() | df["longitude"].isnull()
    audit["missing_coordinates_count"] = int(missing_coords.sum())
    
    valid_coords = ~missing_coords
    out_of_bounds = (
        (df["latitude"] < UK_BOUNDS["min_lat"]) |
        (df["latitude"] > UK_BOUNDS["max_lat"]) |
        (df["longitude"] < UK_BOUNDS["min_lon"]) |
        (df["longitude"] > UK_BOUNDS["max_lon"])
    ) & valid_coords
    audit["out_of_uk_bounds_count"] = int(out_of_bounds.sum())
    
    # Retain only valid geographic records
    valid_geo_mask = valid_coords & (~out_of_bounds)
    df_cleaned = df[valid_geo_mask].copy()
    
    # 2. Target Variable Validation (Must be 1, 2, or 3)
    valid_target_mask = df_cleaned[TARGET_COLUMN].isin([1, 2, 3])
    invalid_target_count = int((~valid_target_mask).sum())
    audit["invalid_target_count"] = invalid_target_count
    df_cleaned = df_cleaned[valid_target_mask].copy()
    
    # 3. Speed Limit Validation (Must be standard UK limits: 20, 30, 40, 50, 60, 70)
    valid_speed_limits = [20, 30, 40, 50, 60, 70]
    valid_speed_mask = df_cleaned["speed_limit"].isin(valid_speed_limits)
    invalid_speed_count = int((~valid_speed_mask).sum())
    audit["invalid_speed_count"] = invalid_speed_count
    
    audit["initial_rows"] = initial_rows
    audit["cleaned_rows"] = len(df_cleaned)
    audit["dropped_rows"] = initial_rows - len(df_cleaned)
    
    logger.info(
        "Data cleaning complete: %d rows retained, %d rows removed (Missing coords: %d, Invalid bounds: %d).",
        len(df_cleaned),
        audit["dropped_rows"],
        audit["missing_coordinates_count"],
        audit["out_of_uk_bounds_count"]
    )
    
    return df_cleaned, audit


def drop_unnecessary_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Drop permanently deprecated, junk, and direct target leakage columns.
    
    Args:
        df: Cleaned DataFrame.
        
    Returns:
        Tuple[pd.DataFrame, list]: Filtered DataFrame and list of dropped columns.
    """
    cols_to_drop = [col for col in PERMANENTLY_DROPPED_COLUMNS if col in df.columns]
    df_filtered = df.drop(columns=cols_to_drop).copy()
    
    logger.info(
        "Dropped %d deprecated/redundant/leakage columns: %s",
        len(cols_to_drop),
        cols_to_drop
    )
    return df_filtered, cols_to_drop
