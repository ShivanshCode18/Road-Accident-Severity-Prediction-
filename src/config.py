"""Configuration module for Road Accident Severity Prediction project.

Centralizes file paths, column definitions, STATS19 reference mappings,
and hyperparameters for reproducible execution.
"""

from pathlib import Path
import os
import tempfile

# Ensure matplotlib cache directory is writable
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARENT_DIR = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DOCS_DIR = PROJECT_ROOT / "docs"
ARCHIVE_DIR = PROJECT_ROOT / "archive"
POWERBI_DIR = ARCHIVE_DIR / "powerbi"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
EXCEL_DATA_DIR = DATA_DIR / "excel"

# Expected Raw Dataset Filename
RAW_DATASET_NAME = "dft-road-casualty-statistics-collision-2025.csv"

# Target & Sampling Configuration
TARGET_COLUMN = "collision_severity"
RANDOM_SEED = 42
TARGET_SAMPLE_SIZE = 100000
MIN_SAMPLE_SIZE = 50000
MAX_SAMPLE_SIZE = 150000

# String Identifier Columns (to prevent scientific notation / precision loss)
ID_COLUMNS = ["collision_index", "collision_ref_no"]

# Bounding Box for Valid UK Geographic Coordinates
UK_BOUNDS = {
    "min_lat": 49.5,
    "max_lat": 61.0,
    "min_lon": -8.5,
    "max_lon": 2.0
}

# Output File Paths
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "accident_cleaned.csv"
FINAL_100K_CSV_PATH = PROCESSED_DATA_DIR / "accident_final_100k.csv"
FINAL_100K_PARQUET_PATH = PROCESSED_DATA_DIR / "accident_final_100k.parquet"
EXCEL_DASHBOARD_DATA_PATH = PROCESSED_DATA_DIR / "excel_dashboard_dataset.csv"
EXCEL_DASHBOARD_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "excel_dashboard_predictions.csv"
MODEL_PERFORMANCE_METRICS_PATH = PROCESSED_DATA_DIR / "model_performance_metrics.csv"
EXCEL_WORKBOOK_PATH = EXCEL_DATA_DIR / "Road_Accident_Severity_Professional_Dashboard.xlsx"

# Backward compatibility alias
POWERBI_DATA_PATH = EXCEL_DASHBOARD_DATA_PATH

PREPROCESSING_REPORT_PATH = REPORTS_DIR / "preprocessing_report.txt"
DATA_QUALITY_REPORT_PATH = REPORTS_DIR / "data_quality_report.txt"
PIPELINE_LOG_PATH = LOGS_DIR / "pipeline.log"

# Official STATS19 Categorical Mappings
SEVERITY_MAP = {
    1: "Fatal",
    2: "Serious",
    3: "Slight"
}

DAY_OF_WEEK_MAP = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday"
}

WEATHER_CONDITIONS_MAP = {
    1: "Fine no high winds",
    2: "Raining no high winds",
    3: "Snowing no high winds",
    4: "Fine + high winds",
    5: "Raining + high winds",
    6: "Snowing + high winds",
    7: "Fog or mist",
    8: "Other",
    9: "Unknown",
    -1: "Missing"
}

ROAD_SURFACE_CONDITIONS_MAP = {
    1: "Dry",
    2: "Wet or damp",
    3: "Snow",
    4: "Frost or ice",
    5: "Flood over 3cm deep",
    6: "Oil or diesel",
    7: "Mud",
    9: "Unknown",
    -1: "Missing"
}

LIGHT_CONDITIONS_MAP = {
    1: "Daylight",
    4: "Darkness - lights lit",
    5: "Darkness - lights unlit",
    6: "Darkness - no lighting",
    7: "Darkness - lighting unknown",
    -1: "Missing"
}

ROAD_TYPE_MAP = {
    1: "Roundabout",
    2: "One way street",
    3: "Dual carriageway",
    6: "Single carriageway",
    7: "Slip road",
    9: "Unknown",
    12: "One way street/Slip road",
    -1: "Missing"
}

URBAN_RURAL_MAP = {
    1: "Urban",
    2: "Rural",
    3: "Unallocated",
    -1: "Missing"
}

JUNCTION_DETAIL_MAP = {
    0: "Not at junction",
    1: "Roundabout",
    2: "Mini-roundabout",
    3: "T or staggered junction",
    5: "Slip road",
    6: "Crossroads",
    7: "Multiple junction",
    8: "Private drive/entrance",
    9: "Other junction",
    13: "T or staggered junction",
    16: "Crossroads",
    17: "Multiple junction",
    18: "Private drive",
    19: "Other junction",
    99: "Unknown",
    -1: "Missing/Unknown"
}

JUNCTION_CONTROL_MAP = {
    1: "Authorised person",
    2: "Auto traffic signal",
    3: "Stop sign",
    4: "Give way or uncontrolled",
    9: "Unknown",
    -1: "Not at junction / Missing"
}

FIRST_ROAD_CLASS_MAP = {
    1: "Motorway",
    2: "A(M)",
    3: "A",
    4: "B",
    5: "C",
    6: "Unclassified",
    -1: "Missing"
}

# Permanently Dropped Columns (Deprecated, Junk, or Direct Target Leakage)
PERMANENTLY_DROPPED_COLUMNS = [
    "local_authority_district",
    "junction_detail_historic",
    "pedestrian_crossing_human_control_historic",
    "pedestrian_crossing_physical_facilities_historic",
    "special_conditions_at_site",
    "carriageway_hazards_historic",
    "local_authority_highway",
    "local_authority_highway_current",
    "location_easting_osgr",
    "location_northing_osgr",
    "first_road_number",
    "second_road_number",
    "second_road_class",
    "lsoa_of_accident_location",
    "collision_ref_no",
    "enhanced_severity_collision",
    "collision_injury_based",
    "collision_adjusted_severity_serious",
    "collision_adjusted_severity_slight"
]

# Excluded Features Documentation
EXCLUDED_FEATURES_EXPLANATION = {
    "local_authority_district": "100% missing values (-1); deprecated in official DfT STATS19 in favor of ONS district codes.",
    "junction_detail_historic": ">86% missing (-1); deprecated historic field replaced by modernized junction_detail.",
    "pedestrian_crossing_human_control_historic": ">86% missing (-1); deprecated historic field.",
    "pedestrian_crossing_physical_facilities_historic": ">86% missing (-1); deprecated historic field.",
    "special_conditions_at_site": ">86% missing (-1); deprecated historic field.",
    "carriageway_hazards_historic": ">86% missing (-1); deprecated historic field.",
    "local_authority_highway": "Administrative identifier redundant with local_authority_ons_district.",
    "local_authority_highway_current": "Administrative identifier redundant with local_authority_ons_district.",
    "collision_year": "Constant value (2025) across all records; provides zero predictive variance.",
    "collision_index": "Unique incident identifier; retained in Excel dashboard and final dataset for audit trail, excluded from predictive ML features.",
    "collision_ref_no": "Secondary accident identifier redundant with collision_index.",
    "location_easting_osgr": "Spatial coordinate redundant with WGS84 longitude.",
    "location_northing_osgr": "Spatial coordinate redundant with WGS84 latitude.",
    "first_road_number": "High cardinality road route number; leads to categorical overfitting.",
    "second_road_class": "Largely unpopulated or 0 when collision is not at a multi-road junction.",
    "second_road_number": "High cardinality road route number; mostly 0 or missing.",
    "lsoa_of_accident_location": "Census lower layer super output area; excessively high cardinality (>30,000 distinct codes).",
    "did_police_officer_attend_scene_of_accident": "Post-accident response variable; not known prior to or at time of incident.",
    "number_of_casualties": "Post-accident outcome information (Data Leakage for severity prediction). Excluded from ML modeling; retained in Excel dashboard dataset for KPI reporting.",
    "enhanced_severity_collision": "Direct target leakage; represents an alternative post-hoc severity metric.",
    "collision_injury_based": "Direct target leakage; records whether severity was injury-based.",
    "collision_adjusted_severity_serious": "Direct target leakage; DfT statistical model adjustment probability for serious severity.",
    "collision_adjusted_severity_slight": "Direct target leakage; DfT statistical model adjustment probability for slight severity."
}


def get_raw_dataset_path() -> Path:
    """Locate the raw STATS19 CSV file dynamically.
    
    Checks dataset/raw first, then checks the workspace parent directory.
    
    Returns:
        Path: Absolute path to the raw CSV file.
        
    Raises:
        FileNotFoundError: If the raw dataset cannot be found.
    """
    candidate_locations = [
        RAW_DATA_DIR / RAW_DATASET_NAME,
        RAW_DATA_DIR / "accident_collisions_2025.csv",
        PARENT_DIR / RAW_DATASET_NAME,
        RAW_DATA_DIR / "collisions_2025.csv",
        PROJECT_ROOT / RAW_DATASET_NAME
    ]
    
    # Also check if any CSV file exists in RAW_DATA_DIR
    if RAW_DATA_DIR.exists():
        csv_files = list(RAW_DATA_DIR.glob("*.csv"))
        if csv_files:
            candidate_locations.insert(0, csv_files[0])
            
    # Also check parent directory for any dft-*.csv
    if PARENT_DIR.exists():
        parent_csvs = list(PARENT_DIR.glob("*collision*2025*.csv"))
        if parent_csvs:
            candidate_locations.append(parent_csvs[0])
            
    for loc in candidate_locations:
        if loc.exists() and loc.is_file():
            return loc.resolve()
            
    raise FileNotFoundError(
        f"Could not locate raw dataset. Checked: {[str(p) for p in candidate_locations]}"
    )
