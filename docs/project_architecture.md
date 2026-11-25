# System Architecture & Technical Design
## Road Accident Severity Prediction Using Apache Spark and Machine Learning with Excel Dashboard

---

## 1. High-Level Architectural Overview

The system implements a production-grade, three-tier data engineering and analytical intelligence architecture:

```
+-------------------------------------------------------------------------------+
|                       UK STATS19 (2025) RAW DATASET                           |
|                        101,525 Records, 44 Attributes                         |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| PHASE 1: DATA INGESTION, HYGIENE & PREPROCESSING (Python / Pandas / PyArrow)  |
|  * Missing coordinate removal (2 invalid records -> 101,523)                  |
|  * Zero duplicate row verification                                            |
|  * Strict Target Leakage Elimination (18 post-crash fields dropped)           |
|  * 14 Engineered Features (Temporal, Calendar, Spatial, Environmental)       |
|  * Stratified Sampling (100,000 records, Seed 42, Severity Ratio Preserved)   |
|  * Formats Exported: CSV & Columnar Snappy-Compressed Parquet                 |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| PHASE 2: DISTRIBUTED SPARK PROCESSING & MLlib MODELING (Apache Spark 3.5.9)   |
|  * Execution Engine: PySpark local[*] (Multi-threaded single-node execution)  |
|  * 23-Stage Pipeline: StringIndexer -> OneHotEncoder -> VectorAssembler       |
|  * Stratified Split: 80% Train (79,901) / 20% Test (20,099)                   |
|  * Class Imbalance Mitigation: Inverse Frequency Class Weighting              |
|  * Multiclass Models: Logistic Regression, Decision Tree, Random Forest       |
|  * Selected Safety Model: Class-Weighted Logistic Regression (61.11% Recall)  |
|  * Model Artifacts Persisted: PipelineModel, BestModel, Evaluation Summary    |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
| PHASE 3: BUSINESS INTELLIGENCE & EXCEL DASHBOARD (Microsoft Excel)            |
|  * Authoritative Workbook: Road_Accident_Severity_Professional_Dashboard.xlsx  |
|  * 8 Structured Worksheets:                                                   |
|    - Executive Dashboard (KPIs, Trends, Diurnal Peaks, 7 Dropdown Filters)    |
|    - Road & Environment (Infrastructure, Speed Limits, Surface, Lighting)     |
|    - Geographical Analysis (Police Forces, Urban vs. Rural, Map Scatter)      |
|    - ML Model Performance (Metrics Benchmark, ROC/F1, 3 Confusion Matrices)   |
|    - Prediction Analysis (Held-out Test Prediction Audit, Probability Vector) |
|    - Accident Data (100,000 Clean Collision Records Data Repository)          |
|    - Calc Engine (Formula Aggregation Matrix & Dynamic Lookup Tables)         |
|    - Map Sample (7,431 Stratified Geographic Coordinates Sample)              |
|  * 41 Native Excel Visualizations, 0 External Add-in Dependencies            |
+-------------------------------------------------------------------------------+
```

---

## 2. Phase 1: Data Engineering & Preprocessing Tier

### 2.1 Technology & Tooling
- **Language**: Python 3.11.9 (64-bit)
- **Data Manipulation**: Pandas (`pandas>=2.0.0`), NumPy (`numpy>=1.24.0`)
- **Columnar Serialization**: Apache PyArrow (`pyarrow>=12.0.0`)
- **Sampling Strategy**: Scikit-learn (`sklearn.model_selection.train_test_split`)

### 2.2 Ingestion & Hygiene Logic
1. **Coordinate Verification**: Raw coordinates are validated against the standard WGS84 UK bounding box ($49.5^\circ\text{N} \le \text{Latitude} \le 61.0^\circ\text{N}$, $-8.5^\circ\text{W} \le \text{Longitude} \le 2.0^\circ\text{E}$). Records with missing or corrupt coordinates are removed (2 records dropped out of 101,525).
2. **Duplicate Audit**: Uniqueness of `collision_index` is strictly validated. The dataset contains zero duplicate rows.
3. **Data Leakage Elimination**: Post-accident outcome indicators (such as `number_of_casualties`, `enhanced_severity_collision`, `collision_injury_based`, `collision_adjusted_severity_serious`, and `collision_adjusted_severity_slight`) are isolated. These represent consequences of severity rather than antecedent causes.
4. **Stratified Sampling**: A representative sample of exactly 100,000 records is selected using stratified sampling on `collision_severity` with fixed random seed `42`, preserving exact national injury proportions:
   - Fatal: 1,431 (1.43%)
   - Serious: 24,812 (24.81%)
   - Slight: 73,757 (73.76%)
5. **Storage Architecture**: Exported as columnar Snappy-compressed Parquet (`accident_final_100k.parquet`) for fast parallel ingestion into Apache Spark.

---

## 3. Phase 2: Apache Spark & PySpark MLlib Tier

### 3.1 Local Spark Execution Architecture
Rather than requiring a multi-node cluster (which introduces substantial operational overhead for evaluation), the pipeline is configured for **Apache Spark 3.5.9 in local multi-threaded mode (`local[*]`)**:
- **Spark Master**: `local[*]` (utilizes all available CPU cores).
- **Driver Memory**: `4g`.
- **Executor Memory**: `2g`.
- **Shuffle Partitions**: `8` (optimized for local multi-core parallelism).
- **Hadoop Integration**: Hadoop 3.2.4 `winutils.exe` configured on Windows to handle native local filesystem I/O.
- **Java Virtual Machine**: 64-bit Oracle Java 8 (`jdk1.8.0_481`).

### 3.2 Feature Engineering Pipeline (PySpark MLlib)
The ML pipeline constructs a 23-stage feature transformation workflow:
1. `StringIndexer`: Transforms 10 categorical string dimensions (`local_authority_ons_district`, `month_name`, `time_of_day`, `day_name`, `weekday_weekend`, `road_type_name`, `urban_rural_name`, `weather_group`, `road_surface_group`, `light_group`, `speed_zone`) into indexed numeric features (`handleInvalid="keep"`).
2. `OneHotEncoder`: Maps indexed categorical features into sparse binary indicator vectors.
3. `VectorAssembler`: Combines 24 numerical predictors and one-hot encoded categorical vectors into a single consolidated feature vector `features`.

### 3.3 Class Imbalance Mitigation Strategy
Because Fatal accidents represent only 1.43% of records, standard unweighted classifiers suffer from the "accuracy paradox" (predicting the majority class for 74% theoretical accuracy while completely failing to identify fatalities). The pipeline computes mathematical inverse class weights:
$$w_c = \frac{N}{3 \cdot N_c}$$
These weights are applied during loss minimization in **Multinomial Logistic Regression**, achieving a **61.11% Fatal Recall rate**.

---

## 4. Phase 3: Analytical Visualization & Excel Dashboard Tier

### 4.1 Transition from Power BI to Excel
While Power BI was initially explored for dashboard prototyping, Microsoft Excel was selected as the final production BI technology. This delivers:
1. **Portability & Accessibility**: Evaluators can open, inspect, and evaluate the deliverable on any standard workstation without specialized Power BI Desktop licenses or cloud service configurations.
2. **Deterministic Formula Governance**: Every KPI, aggregate chart, and metric is powered by native, transparent Excel formulas rather than opaque binary data caches.
3. **Comprehensive Embedded Data**: The workbook encapsulates the entire 100,000-record collision dataset and 20,099-record test prediction partition directly in separate worksheets.

### 4.2 Workbook Internal Architecture
- **Worksheets 1–5**: Presentation dashboards (Executive KPIs, Infrastructure & Environmental Risks, Geographical Intelligence, ML Model Comparisons, Prediction Diagnostics).
- **Worksheet 6 (`Accident Data`)**: Full 100,000-row repository.
- **Worksheet 7 (`Calc Engine`)**: Dynamic formula calculation engine responding to 7 Executive Dashboard filter dropdowns.
- **Worksheet 8 (`Map Sample`)**: Stratified 7,431-point sample with conditional `#N/A` masking for responsive coordinate scatter charting.

