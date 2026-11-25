"""Feature Engineering Module for STATS19 Collisions Dataset.

Derives temporal, calendar, categorical grouping, and environmental features
strictly adhering to the STATS19 coding guide without data fabrication.
"""

import logging
from typing import Tuple, List
import pandas as pd
import numpy as np

from .config import (
    SEVERITY_MAP,
    DAY_OF_WEEK_MAP,
    WEATHER_CONDITIONS_MAP,
    ROAD_SURFACE_CONDITIONS_MAP,
    LIGHT_CONDITIONS_MAP,
    ROAD_TYPE_MAP,
    URBAN_RURAL_MAP
)

logger = logging.getLogger(__name__)


def assign_time_of_day(hour: int) -> str:
    """Classify 24-hour integer into academic time-of-day categories.
    
    Morning: 06:00 - 11:59 (6 to 11)
    Afternoon: 12:00 - 16:59 (12 to 16)
    Evening: 17:00 - 21:59 (17 to 21)
    Night: 22:00 - 05:59 (22, 23, 0 to 5)
    """
    if pd.isnull(hour):
        return "Unknown"
    hour = int(hour)
    if 6 <= hour <= 11:
        return "Morning"
    elif 12 <= hour <= 16:
        return "Afternoon"
    elif 17 <= hour <= 21:
        return "Evening"
    else:
        return "Night"


def assign_weather_group(code: int) -> str:
    """Consolidate granular STATS19 weather codes into macro groupings."""
    if code in [1, 4]:
        return "Fine"
    elif code in [2, 5]:
        return "Raining"
    elif code in [3, 6, 7]:
        return "Snow/Ice/Fog"
    else:
        return "Other/Unknown"


def assign_road_surface_group(code: int) -> str:
    """Consolidate granular STATS19 road surface codes into macro groupings."""
    if code == 1:
        return "Dry"
    elif code in [2, 5]:
        return "Wet/Flood"
    elif code in [3, 4]:
        return "Snow/Ice"
    else:
        return "Other/Unknown"


def assign_light_group(code: int) -> str:
    """Consolidate granular STATS19 lighting condition codes into macro groupings."""
    if code == 1:
        return "Daylight"
    elif code == 4:
        return "Darkness - Lights Lit"
    elif code in [5, 6, 7]:
        return "Darkness - Unlit/No Lights"
    else:
        return "Other/Unknown"


def assign_speed_zone(speed: int) -> str:
    """Group speed limits into intuitive speed zones."""
    if speed <= 20:
        return "<=20 mph"
    elif speed == 30:
        return "30 mph"
    elif speed in [40, 50]:
        return "40-50 mph"
    elif speed >= 60:
        return "60-70 mph"
    else:
        return "Unknown"


def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Create all required temporal, calendar, and environmental features.
    
    Args:
        df: Input cleaned DataFrame.
        
    Returns:
        Tuple[pd.DataFrame, List[str]]: Augmented DataFrame and list of created features.
    """
    logger.info("Starting feature engineering...")
    df_feat = df.copy()
    created_features = []
    
    # 1. Temporal Parsing
    # Date parsing (UK format: DD/MM/YYYY)
    dt_series = pd.to_datetime(df_feat["date"], format="%d/%m/%Y", errors="coerce")
    
    df_feat["month"] = dt_series.dt.month.astype("int32")
    created_features.append("month")
    
    df_feat["month_name"] = dt_series.dt.strftime("%B")
    created_features.append("month_name")
    
    # Time parsing (HH:MM)
    time_series = pd.to_datetime(df_feat["time"], format="%H:%M", errors="coerce")
    df_feat["hour"] = time_series.dt.hour.astype("int32")
    created_features.append("hour")
    
    # Time of Day Classification
    df_feat["time_of_day"] = df_feat["hour"].apply(assign_time_of_day)
    created_features.append("time_of_day")
    
    # Day name and Weekend Indicators
    # STATS19 day_of_week: 1=Sunday, 2=Monday, ..., 7=Saturday
    df_feat["day_name"] = df_feat["day_of_week"].map(DAY_OF_WEEK_MAP).fillna("Unknown")
    created_features.append("day_name")
    
    df_feat["weekend_indicator"] = df_feat["day_of_week"].apply(lambda d: 1 if d in [1, 7] else 0).astype("int32")
    created_features.append("weekend_indicator")
    
    df_feat["weekday_weekend"] = df_feat["weekend_indicator"].apply(lambda w: "Weekend" if w == 1 else "Weekday")
    created_features.append("weekday_weekend")
    
    # 2. Meaningful String Labels from STATS19 Dictionaries
    df_feat["severity_name"] = df_feat["collision_severity"].map(SEVERITY_MAP).fillna("Unknown")
    created_features.append("severity_name")
    
    df_feat["road_type_name"] = df_feat["road_type"].map(ROAD_TYPE_MAP).fillna("Unknown")
    created_features.append("road_type_name")
    
    df_feat["urban_rural_name"] = df_feat["urban_or_rural_area"].map(URBAN_RURAL_MAP).fillna("Unknown")
    created_features.append("urban_rural_name")
    
    # 3. Environmental Categorical Groupings
    df_feat["weather_group"] = df_feat["weather_conditions"].apply(assign_weather_group)
    created_features.append("weather_group")
    
    df_feat["road_surface_group"] = df_feat["road_surface_conditions"].apply(assign_road_surface_group)
    created_features.append("road_surface_group")
    
    df_feat["light_group"] = df_feat["light_conditions"].apply(assign_light_group)
    created_features.append("light_group")
    
    df_feat["speed_zone"] = df_feat["speed_limit"].apply(assign_speed_zone)
    created_features.append("speed_zone")
    
    logger.info("Successfully engineered %d features: %s", len(created_features), created_features)
    return df_feat, created_features

