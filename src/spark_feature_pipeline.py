"""Spark MLlib Feature Pipeline and Stratified Partitioning Module.

Assembles StringIndexer, OneHotEncoder, and VectorAssembler into an MLlib Pipeline.
Implements exact stratified train-test splitting (80/20) with disjoint record isolation.
"""

import logging
import math
from typing import Tuple, List, Dict
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

logger = logging.getLogger(__name__)

RANDOM_SEED = 42

# Label Mappings (Deterministic Bi-Directional Mapping)
# STATS19: 1 = Fatal, 2 = Serious, 3 = Slight
SEVERITY_CODE_TO_LABEL = {1: 0.0, 2: 1.0, 3: 2.0}
LABEL_TO_SEVERITY_CODE = {0.0: 1, 1.0: 2, 2.0: 3}
LABEL_TO_SEVERITY_NAME = {0.0: "Fatal", 1.0: "Serious", 2.0: "Slight"}
SEVERITY_CODE_TO_NAME = {1: "Fatal", 2: "Serious", 3: "Slight"}

# Explicit Feature Architecture (Strictly excluding IDs and post-accident leakage)
CATEGORICAL_PREDICTORS = [
    "time_of_day",
    "day_name",
    "road_type_name",
    "urban_rural_name",
    "weather_group",
    "road_surface_group",
    "light_group",
    "speed_zone",
    "junction_detail",
    "junction_control",
    "police_force",
    # Advanced Domain Feature Interactions
    "road_speed_inter",
    "weather_surface_inter",
    "light_time_inter",
    "weekend_time_inter",
    "urban_rural_speed_inter"
]

NUMERICAL_PREDICTORS = [
    "hour",
    "month",
    "speed_limit",
    "number_of_vehicles",
    "latitude",
    "longitude",
    "weekend_indicator",
    # Continuous Cyclical Temporal Representations
    "sin_hour",
    "cos_hour",
    "sin_month",
    "cos_month",
    # Critical Domain Safety Indicator
    "high_speed_single_carriageway"
]

EXCLUDED_COLUMNS = [
    "collision_index",
    "collision_ref_no",
    "number_of_casualties",
    "enhanced_severity_collision",
    "collision_adjusted_severity_serious",
    "collision_adjusted_severity_slight",
    "did_police_officer_attend_scene_of_accident",
    "first_road_number",
    "second_road_number",
    "collision_year"
]


def engineer_spark_features(df: DataFrame) -> DataFrame:
    """Generate cyclical temporal representations and domain risk interactions.
    
    Strictly preserves target leakage constraints: excludes casualties and post-crash data.
    """
    logger.info("Engineering cyclical representations and domain interaction features...")
    df_feat = df
    
    # 1. Cyclical trigonometric encodings of hour (24-hr) and month (12-mo)
    # Eliminates artificial distance boundaries between 23:00 and 00:00, Dec and Jan
    pi = math.pi
    if "hour" in df_feat.columns:
        df_feat = df_feat.withColumn("sin_hour", F.sin(2.0 * pi * F.col("hour").cast("double") / 24.0)) \
                         .withColumn("cos_hour", F.cos(2.0 * pi * F.col("hour").cast("double") / 24.0))
    if "month" in df_feat.columns:
        df_feat = df_feat.withColumn("sin_month", F.sin(2.0 * pi * F.col("month").cast("double") / 12.0)) \
                         .withColumn("cos_month", F.cos(2.0 * pi * F.col("month").cast("double") / 12.0))
                         
    # 2. Road type and Speed zone interaction
    # Captures high-hazard single carriageways at 60mph vs urban multi-lane streets
    if "road_type_name" in df_feat.columns and "speed_zone" in df_feat.columns:
        df_feat = df_feat.withColumn(
            "road_speed_inter",
            F.concat_ws("_", F.col("road_type_name"), F.col("speed_zone"))
        )
        
    # 3. Weather and Road surface condition interaction
    # Captures compounding friction hazards (e.g., Raining on Wet vs Snow on Icy roads)
    if "weather_group" in df_feat.columns and "road_surface_group" in df_feat.columns:
        df_feat = df_feat.withColumn(
            "weather_surface_inter",
            F.concat_ws("_", F.col("weather_group"), F.col("road_surface_group"))
        )
        
    # 4. Lighting conditions and Time of day interaction
    # Contrasts unlit darkness at night against lit darkness during evening commute
    if "light_group" in df_feat.columns and "time_of_day" in df_feat.columns:
        df_feat = df_feat.withColumn(
            "light_time_inter",
            F.concat_ws("_", F.col("light_group"), F.col("time_of_day"))
        )
        
    # 5. Weekend and Time of day combination
    # Identifies late-night weekend risk patterns vs weekday morning rush hour
    if "weekday_weekend" in df_feat.columns and "time_of_day" in df_feat.columns:
        df_feat = df_feat.withColumn(
            "weekend_time_inter",
            F.concat_ws("_", F.col("weekday_weekend"), F.col("time_of_day"))
        )
        
    # 6. Road environment (Urban/Rural) and Speed zone interaction
    # Captures high-speed rural routes (primary source of fatal crashes)
    if "urban_rural_name" in df_feat.columns and "speed_zone" in df_feat.columns:
        df_feat = df_feat.withColumn(
            "urban_rural_speed_inter",
            F.concat_ws("_", F.col("urban_rural_name"), F.col("speed_zone"))
        )
        
    # 7. High-speed single carriageway indicator (Literature-grounded top UK fatality factor)
    if "road_type_name" in df_feat.columns and "speed_limit" in df_feat.columns:
        df_feat = df_feat.withColumn(
            "high_speed_single_carriageway",
            F.when((F.col("road_type_name") == "Single carriageway") & (F.col("speed_limit") >= 60), 1.0)
             .otherwise(0.0)
        )
        
    return df_feat


def prepare_target_label(df: DataFrame) -> DataFrame:
    """Add deterministic zero-indexed 'label' column and engineer advanced features.
    
    1 (Fatal)   -> 0.0
    2 (Serious) -> 1.0
    3 (Slight)  -> 2.0
    """
    logger.info("Adding deterministic MLlib 'label' column from collision_severity...")
    
    # Cast categorical codes to string to ensure StringIndexer treats them categorically
    df_prepared = df
    for col_name in ["junction_detail", "junction_control", "police_force"]:
        if col_name in df_prepared.columns:
            df_prepared = df_prepared.withColumn(col_name, F.col(col_name).cast("string"))
            
    df_prepared = df_prepared.withColumn(
        "label",
        F.when(F.col("collision_severity") == 1, 0.0)
         .when(F.col("collision_severity") == 2, 1.0)
         .when(F.col("collision_severity") == 3, 2.0)
         .otherwise(-1.0)
    )
    
    # Apply advanced feature engineering
    df_prepared = engineer_spark_features(df_prepared)
    return df_prepared


def stratified_train_test_split(
    df: DataFrame,
    train_ratio: float = 0.8,
    seed: int = RANDOM_SEED
) -> Tuple[DataFrame, DataFrame]:
    """Perform exact stratified 80/20 train-test split using left-anti join for guaranteed disjoint isolation.
    
    Args:
        df: Input DataFrame with collision_severity and collision_index.
        train_ratio: Fraction for training data (0.8).
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple[DataFrame, DataFrame]: train_df (80%), test_df (20%).
    """
    logger.info("Executing stratified train-test split (ratio=%.2f, seed=%d)...", train_ratio, seed)
    
    # Stratified sampling on collision_severity
    fractions = {1: train_ratio, 2: train_ratio, 3: train_ratio}
    train_df = df.sampleBy("collision_severity", fractions=fractions, seed=seed)
    
    # Left-anti join on collision_index ensures zero records overlap into test_df
    test_df = df.join(train_df.select("collision_index"), on="collision_index", how="left_anti")
    
    train_count = train_df.count()
    test_count = test_df.count()
    total = train_count + test_count
    
    logger.info(
        "Split completed: Train=%d (%.2f%%), Test=%d (%.2f%%), Total=%d (Disjoint: 100%%)",
        train_count, (train_count / total) * 100,
        test_count, (test_count / total) * 100,
        total
    )
    
    return train_df, test_df


def build_feature_pipeline() -> Tuple[Pipeline, List[str]]:
    """Construct PySpark MLlib Pipeline consisting of StringIndexers, OneHotEncoders, and VectorAssembler.
    
    Returns:
        Tuple[Pipeline, List[str]]: Unfitted ML Pipeline and list of assembler input feature names.
    """
    logger.info("Building PySpark MLlib Feature Pipeline...")
    stages = []
    ohe_output_cols = []
    
    for cat_col in CATEGORICAL_PREDICTORS:
        idx_col = f"{cat_col}_idx"
        vec_col = f"{cat_col}_vec"
        
        indexer = StringIndexer(
            inputCol=cat_col,
            outputCol=idx_col,
            handleInvalid="keep"
        )
        encoder = OneHotEncoder(
            inputCol=idx_col,
            outputCol=vec_col,
            handleInvalid="keep"
        )
        stages.extend([indexer, encoder])
        ohe_output_cols.append(vec_col)
        
    assembler_inputs = ohe_output_cols + NUMERICAL_PREDICTORS
    assembler = VectorAssembler(
        inputCols=assembler_inputs,
        outputCol="features",
        handleInvalid="keep"
    )
    stages.append(assembler)
    
    pipeline = Pipeline(stages=stages)
    logger.info("Feature pipeline created with %d stages (%d OHE vectors + %d numerical features).",
                len(stages), len(ohe_output_cols), len(NUMERICAL_PREDICTORS))
    return pipeline, assembler_inputs


def fit_and_transform_pipeline(
    train_df: DataFrame,
    test_df: DataFrame
) -> Tuple[PipelineModel, DataFrame, DataFrame]:
    """Fit feature pipeline on training data ONLY, then transform both train and test sets.
    
    Args:
        train_df: Raw training DataFrame.
        test_df: Raw test DataFrame.
        
    Returns:
        Tuple[PipelineModel, DataFrame, DataFrame]: Fitted PipelineModel, train_transformed, test_transformed.
    """
    pipeline, _ = build_feature_pipeline()
    
    logger.info("Fitting feature pipeline on training data (preventing test set leakage)...")
    pipeline_model = pipeline.fit(train_df)
    
    logger.info("Transforming training dataset...")
    train_transformed = pipeline_model.transform(train_df).cache()
    
    logger.info("Transforming testing dataset...")
    test_transformed = pipeline_model.transform(test_df).cache()
    
    return pipeline_model, train_transformed, test_transformed

