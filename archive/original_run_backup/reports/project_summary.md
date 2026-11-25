# End-to-End Project Summary
## Road Accident Severity Prediction Using Apache Spark and Machine Learning with Excel Dashboard

---

## 1. Project Overview & Objectives

This academic capstone project presents an end-to-end data engineering and predictive machine learning solution developed to analyze personal injury road collisions across Great Britain and predict accident severity (`Fatal`, `Serious`, `Slight`). 

The technical architecture spans three cohesive phases:
1. **Phase 1: Data Collection, Hygiene & Preprocessing** (Python / Pandas / PyArrow)
2. **Phase 2: Spark Data Processing & Machine Learning Model Development** (Apache Spark 3.5.9 / PySpark MLlib in local mode)
3. **Phase 3: Analytical Visualization & Dashboard Development** (Microsoft Excel)

The project replaces the originally contemplated Power BI visualization tool with an authoritative, self-contained **Microsoft Excel Professional Dashboard** containing 8 worksheets, 41 native charts, and an automated formula calculation engine.

---

## 2. Dataset Provenance & Ingestion Statistics

- **Source**: Official UK Department for Transport (DfT) STATS19 Road Casualty Statistics (2025 Collisions Dataset).
- **Original Raw Population**: `101,525` rows across `44` attributes.
- **Geographic Cleaning**: Validated against Great Britain WGS84 bounding box ($49.5^\circ\text{N} \le \text{Lat} \le 61.0^\circ\text{N}$, $-8.5^\circ\text{W} \le \text{Lon} \le 2.0^\circ\text{E}$). Exactly **2 invalid coordinate records** were removed.
- **Cleaned Population**: `101,523` rows across `25` retained attributes.
- **Duplicate Verification**: `collision_index` uniqueness tested; zero duplicate records existed.
- **Missing Value Audit**: Final processed dataset contains zero `NaN` or unhandled missing values across all columns.
- **Stratified Final Sample**: Exactly `100,000` records selected via stratified random sampling on `collision_severity` with fixed random seed `42`.
- **Target Severity Distribution (100,000 Records)**:
  - **Fatal (Class 1)**: `1,431` records (`1.43%`)
  - **Serious (Class 2)**: `24,812` records (`24.81%`)
  - **Slight (Class 3)**: `73,757` records (`73.76%`)

---

## 3. Preprocessing & Data Leakage Prevention

To ensure academic and methodological validity, 18 post-accident attributes were strictly eliminated from predictive feature sets:
1. **Consequential Outcomes**: `number_of_casualties` (retained in dashboard for casualty volume reporting, excluded from model training).
2. **Direct Target Leakage**: `enhanced_severity_collision`, `collision_injury_based`, `collision_adjusted_severity_serious`, `collision_adjusted_severity_slight`.
3. **Post-Crash Administrative Data**: `did_police_officer_attend_scene_of_accident`.
4. **Deprecated Fields (>86% missing)**: `local_authority_district`, `junction_detail_historic`, `pedestrian_crossing_human_control_historic`, `pedestrian_crossing_physical_facilities_historic`, `special_conditions_at_site`, `carriageway_hazards_historic`.
5. **Collinear & High-Cardinality Identifiers**: `collision_year` (constant 2025), `collision_index`, `collision_ref_no`, `location_easting_osgr`, `location_northing_osgr`, `first_road_number`, `second_road_number`, `second_road_class`, `lsoa_of_accident_location`.

---

## 4. Feature Engineering

Fourteen dedicated features were engineered to capture spatial, temporal, and hazard dynamics:
- **Temporal & Calendar (7)**: `month`, `month_name`, `hour`, `time_of_day` (Morning Rush, Midday, Evening Rush, Night), `day_name`, `weekend_indicator` (0/1), `weekday_weekend`.
- **Categorical Mappings & Groupings (7)**: `severity_name`, `road_type_name`, `urban_rural_name`, `weather_group` (Fine, Raining, Snowing, Fog/Mist, Other), `road_surface_group` (Dry, Wet, Snow/Ice, Flood), `light_group` (Daylight, Darkness lit, Darkness unlit, Darkness no lighting), `speed_zone` (Urban Low Speed, Moderate Speed, High Speed).

---

## 5. Apache Spark Processing & ML Pipeline

- **Spark Environment**: Apache Spark 3.5.9 (PySpark MLlib) executed in local multi-threaded mode (`master="local[*]"`) on Windows with Oracle Java 8 and Hadoop 3.2.4 `winutils`.
- **Storage Engine**: Apache Parquet with Snappy compression (`accident_final_100k.parquet`).
- **Feature Pipeline (23 Stages)**:
  - `StringIndexer` fitted on 10 categorical string columns (`handleInvalid="keep"`).
  - `OneHotEncoder` generating sparse binary indicator vectors.
  - `VectorAssembler` consolidating 24 numerical predictors and encoded vectors into unified feature vectors.
- **Stratified Partitioning**:
  - **Training Partition (80%)**: `79,901` records.
  - **Testing Partition (20%)**: `20,099` records (disjoint held-out validation set).
- **Class Imbalance Strategy**: Applied mathematical inverse-frequency class weighting ($w_{\text{Fatal}} \approx 23.67$, $w_{\text{Serious}} \approx 1.33$, $w_{\text{Slight}} \approx 0.45$) to prioritize fatal casualty detection.

---

## 6. Machine Learning Evaluation Results

Three classifiers were trained and benchmarked across the untouched 20,099 held-out test partition:

| Model | Accuracy | Weighted F1 | Macro F1 | Fatal Recall | Fatal Precision | Fatal F1 | Train Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** (Class-Weighted) | 47.43% | 0.5464 | 0.3383 | **61.11%** | 0.0367 | 0.0692 | **9.54s** |
| **Decision Tree** (maxDepth=8) | 46.02% | 0.5346 | 0.3257 | 59.48% | 0.0333 | 0.0631 | 10.04s |
| **Random Forest** (50 Trees, maxDepth=10) | **54.50%** | **0.5985** | **0.3667** | 54.90% | **0.0439** | **0.0814** | 57.59s |

### Trade-off Analysis & The Accuracy Paradox:
- **Random Forest** delivers the best overall classification fidelity (54.50% accuracy, 0.5985 weighted F1).
- **Class-Weighted Logistic Regression** delivers the best public safety performance, successfully identifying **187 out of 306 fatal crashes** (61.11% Recall).
- Because a standard unweighted model achieves ~74% accuracy while detecting 0 fatalities, the class-weighted model is explicitly chosen to maximize life safety over raw accuracy.

---

## 7. Microsoft Excel Professional Dashboard

The final Phase 3 deliverable is located at:
`dataset/excel/Road_Accident_Severity_Professional_Dashboard.xlsx`

The workbook contains **8 cohesive worksheets** and **41 embedded native Excel charts**:
1. **Executive Dashboard**: Executive KPI cards, collision severity distribution, 12-month trend, diurnal commuting profiles, and 7 interactive dropdown filters.
2. **Road & Environment**: Infrastructure risk (single vs. dual carriageway), speed zone analysis, road surface friction, and nighttime illumination hazards.
3. **Geographical Analysis**: Police force jurisdiction volumes, urban (66.1%) vs. rural (33.9%) collision disparity, and nationwide coordinate scatter mapping.
4. **ML Model Performance**: Comparative evaluation matrices, safety recall benchmarking, and 3x3 confusion matrices for all evaluated models.
5. **Prediction Analysis**: Itemized audit of all 20,099 test predictions, displaying actual vs. predicted severity, prediction correctness flags, and calibrated class probabilities (`prob_fatal`, `prob_serious`, `prob_slight`).
6. **Accident Data**: Full 100,000-record data repository with 32 attributes.
7. **Calc Engine**: Dynamic formula-driven calculation matrix that evaluates `COUNTIFS`/`SUMIFS` aggregations and updates charts based on active dropdown selections.
8. **Map Sample**: Stratified geographic sample (7,431 coordinates) with `#N/A` conditional masking for responsive Excel scatter mapping.

---

## 8. Final Conclusions & Key Insights

1. **Environmental Determinants**: Rural single carriageways with 60 mph speed limits under darkness without street lighting represent the most lethal accident profile in the UK.
2. **Commuting Dynamics**: Collision volumes peak during morning rush (08:00–09:00) and evening rush (17:00–18:00), with Fridays experiencing the highest weekly volume.
3. **Methodological Rigor**: Preventing data leakage and addressing class imbalance via inverse frequency weighting is essential for building trustworthy transportation safety models.
4. **Practical Delivery**: Microsoft Excel serves as a highly effective, universally accessible analytical visualization platform for academic capstone presentation and transportation stakeholder engagement.

