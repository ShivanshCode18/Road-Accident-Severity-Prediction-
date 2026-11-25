# Excel Dashboard Detailed Worksheet Documentation
## Road Accident Severity Prediction Using Apache Spark and Machine Learning with Excel Dashboard

---

## 1. Sheet 1: Executive Dashboard

### 1.1 Purpose
The **Executive Dashboard** serves as the primary strategic landing page for transportation authorities, traffic safety engineers, and academic evaluators. It delivers an immediate high-level summary of collision volumes, severity distributions, temporal trends, and overall casualty metrics across the United Kingdom for 2025.

### 1.2 Layout & Key Components
- **Header Banner** (Rows 1–3): Title, dataset provenance (UK Department for Transport STATS19 2025), and sheet navigation links (`Executive Dashboard`, `Road & Environment`, `Geographical Analysis`, `ML Model Performance`, `Prediction Analysis`, `Accident Data`).
- **Interactive Filter Bar** (Rows 5–6): Dropdown controls allowing the user to filter the entire workbook across 7 dimensions:
  1. *Month* (All, January .. December)
  2. *Weather* (All, Fine no high winds, Raining no high winds, etc.)
  3. *Road Type* (All, Single carriageway, Dual carriageway, Roundabout, etc.)
  4. *Surface* (All, Dry, Wet or damp, Snow, Frost or ice, Flood, etc.)
  5. *Lighting* (All, Daylight, Darkness - lights lit, Darkness - no lighting, etc.)
  6. *Area* (All, Urban, Rural)
  7. *Severity* (All, Fatal, Serious, Slight)
- **Active Filter Indicator**: Displays count of active filters (e.g., `Active filters: 0 of 7`).

### 1.3 Executive KPI Metric Cards
- **Total Collisions**: `100,000`
- **Fatal Collisions**: `1,431` (`1.43%`)
- **Serious Collisions**: `24,812` (`24.81%`)
- **Slight Collisions**: `73,757` (`73.76%`)
- **Total Casualties**: `125,983` (Average `1.26` per incident)
- **Total Vehicles**: `181,245` (Average `1.81` per incident)
- **Weekend / Weekday Split**: `24,476` (24.48%) Weekend vs. `75,524` (75.52%) Weekday

### 1.4 Visualizations Included
1. **Collision Severity Proportion (Doughnut Chart)**: Clear breakdown of the 73.8% Slight, 24.8% Serious, and 1.4% Fatal distribution, illustrating real-world class imbalance.
2. **Monthly Collision Trend (Line Chart)**: 12-month collision progression highlighting seasonal winter and summer peaks.
3. **Day of Week Distribution (Column Chart)**: Daily collision counts showing peak activity on Fridays (15,920+) and lower volumes on Sundays.
4. **24-Hour Diurnal Profile (Bar / Area Chart)**: Visualizing the dual-peak commuting hazard: Morning rush hour (08:00–09:00) and Evening rush hour (17:00–18:00).

---

## 2. Sheet 2: Road & Environment

### 2.1 Purpose
This sheet investigates the environmental, meteorological, and infrastructural determinants of collision severity. It is designed to reveal which road types, speed zones, and weather conditions produce disproportionately fatal or serious outcomes.

### 2.2 Layout & Filter Header
Displays the synchronized filter summary read from the Executive Dashboard:
`Filters applied (set on Executive Dashboard) → Month: All | Weather: All | Road type: All | Surface: All | Lighting: All | Area: All | Severity: All`

### 2.3 Analytical Sections & Visualizations
1. **Road Type Severity Analysis**:
   - Compares Single Carriageway, Dual Carriageway, Roundabouts, One-way Streets, and Slip Roads.
   - Highlights that while Single Carriageways account for the majority of crashes (~74%), Dual Carriageways and rural high-speed routes exhibit significantly higher fatality rates per incident.
2. **Speed Limit & Speed Zone Risk Profile**:
   - Groups collisions across 20 mph, 30 mph, 40 mph, 50 mph, 60 mph, and 70 mph limits.
   - Contrasts high-volume low-severity urban zones (30 mph: highest absolute count) against high-fatality rural corridors (60 mph: highest fatal proportion).
3. **Weather Conditions vs. Severity**:
   - Categorizes incidents into Fine, Raining, Snowing, Fog/Mist, and High Winds.
   - Highlights wet road surface interactions during autumn and winter months.
4. **Road Surface Conditions**:
   - Dry vs. Wet/Damp vs. Snow/Ice vs. Flood.
5. **Lighting & Visibility Conditions**:
   - Daylight vs. Darkness with street lights lit vs. Darkness with no street lighting.
   - Demonstrates that unlit rural roads at night generate the steepest risk of fatal collisions.

---

## 3. Sheet 3: Geographical Analysis

### 3.1 Purpose
Provides spatial intelligence on road traffic safety across the United Kingdom, contrasting administrative police force jurisdictions, urban versus rural environments, and coordinate density.

### 3.2 Visualizations & Tables
1. **Top UK Police Forces by Incident Volume (Horizontal Bar Chart)**:
   - Identifies high-density policing areas including Metropolitan Police, Greater Manchester, West Midlands, West Yorkshire, and Thames Valley.
2. **Urban vs. Rural Risk Comparison (Clustered Column Chart)**:
   - **Urban Collisions**: `66,127` (66.13% of total).
   - **Rural Collisions**: `33,871` (33.87% of total).
   - **Fatality Disparity**: Despite representing only one-third of total collisions, rural roads account for the substantial majority of fatal crashes due to higher traveling speeds, single-carriageway head-on collisions, and longer emergency medical service response times.
3. **Geographical Collision Distribution (Latitude/Longitude Scatter Plot)**:
   - Plots collision coordinates across Great Britain (Latitude: 49.91°N to 60.35°N, Longitude: -7.42°W to 1.76°E).
   - Backed by the `Map Sample` sheet (7,431 stratified coordinates) ensuring smooth Excel rendering without system freezing.

---

## 4. Sheet 4: ML Model Performance

### 4.1 Purpose
Presents a rigorous academic evaluation and comparison of the three multiclass classifiers developed using Apache Spark MLlib on an untouched held-out test partition of 20,099 collision records.

### 4.2 Top-Level Performance Cards
- **Best Overall Accuracy**: `54.50%` (Random Forest Classifier)
- **Highest Weighted F1**: `0.5985` (Random Forest Classifier)
- **Highest Macro F1**: `0.3667` (Random Forest Classifier)
- **Highest Fatal Accident Recall (Detection Rate)**: `61.11%` (Class-Weighted Logistic Regression)

### 4.3 Comprehensive Model Comparison Matrix
Metrics sourced directly from `model_performance_metrics.csv`:

| Evaluation Metric | Logistic Regression | Decision Tree | Random Forest | Optimal Model Selection Criterion |
|---|---:|---:|---:|---|
| **Overall Accuracy** | 47.43% | 46.02% | **54.50%** | Random Forest (+7.07% over LR) |
| **Weighted Precision** | 0.6974 | 0.6860 | **0.7108** | Random Forest |
| **Weighted Recall** | 0.4743 | 0.4602 | **0.5450** | Random Forest |
| **Weighted F1-Score** | 0.5464 | 0.5346 | **0.5985** | Random Forest |
| **Macro F1-Score** | 0.3383 | 0.3257 | **0.3667** | Random Forest |
| **Fatal Recall (Safety Rate)** | **61.11%** | 59.48% | 54.90% | **Logistic Regression** (Life-Critical) |
| **Fatal Precision** | 3.67% | 3.33% | **4.39%** | Severe class imbalance trade-off |
| **Fatal F1-Score** | 0.0692 | 0.0631 | **0.0814** | Random Forest |
| **Serious Recall** | **33.53%** | 30.00% | 31.72% | Logistic Regression |
| **Serious F1-Score** | 0.3141 | 0.2905 | **0.3192** | Random Forest |
| **Slight Recall** | 51.65% | 50.93% | **61.87%** | Random Forest |
| **Slight F1-Score** | 0.6315 | 0.6233 | **0.6996** | Random Forest |
| **Training Time (sec)** | **9.54s** | 10.04s | 57.59s | Logistic Regression (Fastest) |

### 4.4 Diagnostic Confusion Matrices
Displays 3x3 confusion matrices for all three models across the 20,099 held-out test partition:

#### Class-Weighted Logistic Regression (Selected Safety Model):
- Fatal (True 306): Pred Fatal = **187 (61.11%)**, Pred Serious = 63, Pred Slight = 56
- Serious (True 4,843): Pred Fatal = 1,493, Pred Serious = **1,624 (33.53%)**, Pred Slight = 1,726
- Slight (True 14,950): Pred Fatal = 3,417, Pred Serious = 3,812, Pred Slight = **7,721 (51.65%)**

#### Decision Tree (maxDepth=8):
- Fatal (True 306): Pred Fatal = **182 (59.48%)**, Pred Serious = 60, Pred Slight = 64
- Serious (True 4,843): Pred Fatal = 1,588, Pred Serious = **1,453 (30.00%)**, Pred Slight = 1,802
- Slight (True 14,950): Pred Fatal = 3,689, Pred Serious = 3,647, Pred Slight = **7,614 (50.93%)**

#### Random Forest (50 Trees, maxDepth=10):
- Fatal (True 306): Pred Fatal = **168 (54.90%)**, Pred Serious = 73, Pred Slight = 65
- Serious (True 4,843): Pred Fatal = 1,127, Pred Serious = **1,536 (31.72%)**, Pred Slight = 2,180
- Slight (True 14,950): Pred Fatal = 2,529, Pred Serious = 3,171, Pred Slight = **9,250 (61.87%)**

### 4.5 Academic Trade-off Discussion
- **The Accuracy Paradox**: Standard non-weighted algorithms achieve ~74% accuracy by predicting "Slight" for every collision, resulting in 0% fatal recall.
- **Safety Priority**: In intelligent transportation systems, missing a fatal incident (False Negative) has catastrophic consequences compared to a false warning (False Positive). Therefore, the class-weighted Logistic Regression model was prioritized for production severity alerting.

---

## 5. Sheet 5: Prediction Analysis

### 5.1 Purpose
Enables item-level inspection, diagnostic filtering, and risk probability analysis across all 20,099 test collision incidents predicted by the primary Spark ML model.

### 5.2 Header Metric Cards
- **Total Test Predictions**: `20,099`
- **Correct Predictions**: `9,532`
- **Incorrect Predictions**: `10,567`
- **Prediction Accuracy**: `47.43%`
- **Actual Fatal Cases Detected**: `187 of 306` (`61.11%`)

### 5.3 Detailed Records Table Structure (Rows 6 to 20,105)
Contains columns for each individual test collision:
- `collision_index`: Unique incident identifier.
- `date`, `time`, `hour`, `time_of_day`, `day_name`, `month_name`.
- `actual_severity`: 1 (Fatal), 2 (Serious), 3 (Slight).
- `actual_severity_name`: Text label.
- `predicted_severity`: Model-assigned class code.
- `predicted_severity_name`: Model-assigned text label.
- `prediction_correct`: Binary indicator (1 if actual == predicted, 0 otherwise).
- `prob_fatal`: Calibrated probability of fatal severity (e.g., 0.6214).
- `prob_serious`: Calibrated probability of serious severity (e.g., 0.2541).
- `prob_slight`: Calibrated probability of slight severity (e.g., 0.1245).
- Contextual features: `number_of_vehicles`, `number_of_casualties`, `speed_limit`, `speed_zone`, `road_type_name`, `urban_rural_name`, `weather_group`, `road_surface_group`, `light_group`, `latitude`, `longitude`.

---

## 6. Sheet 6: Accident Data

### 6.1 Purpose
Stores the complete, feature-engineered 100,000-record collision dataset. Serves as the single source of truth for all calculations, lookup functions, and KPI metrics within the workbook.

### 6.2 Data Attributes (Columns A to AF, Rows 5 to 100,005)
1. `collision_index`: String unique identifier (e.g., `2025010000001`).
2. `date`: Incident date in DD/MM/YYYY format.
3. `time`: 24-hour timestamp (HH:MM).
4. `collision_severity`: Numeric code (1 = Fatal, 2 = Serious, 3 = Slight).
5. `severity_name`: Categorical label (Fatal, Serious, Slight).
6. `number_of_vehicles`: Integer count of vehicles involved.
7. `number_of_casualties`: Integer count of injured/deceased individuals.
8. `latitude`: WGS84 geographic latitude (float).
9. `longitude`: WGS84 geographic longitude (float).
10. `police_force`: UK Police Force integer code.
11. `day_of_week`: Integer day code (1 = Sunday, 7 = Saturday).
12. `day_name`: String day name (Sunday .. Saturday).
13. `month`: Integer month (1 .. 12).
14. `month_name`: String month name (January .. December).
15. `hour`: Hour of collision (0 .. 23).
16. `time_of_day`: Binned day partition (Morning Rush, Midday, Evening Rush, Night).
17. `weekend_indicator`: Binary flag (1 = Weekend, 0 = Weekday).
18. `weekday_weekend`: String label (Weekday, Weekend).
19. `speed_limit`: Posted speed limit in mph (20, 30, 40, 50, 60, 70).
20. `speed_zone`: Grouped speed categorization (Urban Low Speed, Moderate, High Speed).
21. `road_type`: Official STATS19 road type code.
22. `road_type_name`: Descriptive road type name.
23. `urban_or_rural_area`: Code (1 = Urban, 2 = Rural).
24. `urban_rural_name`: Descriptive area name.
25. `weather_conditions`: STATS19 weather code.
26. `weather_group`: Consolidated weather description.
27. `road_surface_conditions`: STATS19 surface code.
28. `road_surface_group`: Consolidated surface description.
29. `light_conditions`: STATS19 lighting code.
30. `light_group`: Consolidated lighting description.
31. `junction_detail`: Junction geometry code.
32. `junction_control`: Traffic control at junction code.

---

## 7. Sheet 7: Calc Engine

### 7.1 Purpose & Role
The **Calc Engine** is the formula-driven aggregation backend. It eliminates external database or OLAP requirements by utilizing native Excel formulas (`COUNTIFS`, `SUMIFS`, `AVERAGEIFS`, `INDEX/MATCH`) to dynamically summarize the 100,000 records from `Accident Data` based on user selections.

### 7.2 Architecture & Structure
- **Filter Parameters Block (Rows 1–15)**:
  - Cells reading the 7 dropdown criteria selected on the Executive Dashboard.
  - Builds criteria test conditions (handling `"All"` versus specific text filters).
- **Dimension Lookup Tables (Columns J to P)**:
  - Canonical lists of Months (12 entries).
  - Canonical lists of Weather conditions (9 entries).
  - Canonical lists of Road types (7 entries).
  - Canonical lists of Road surface conditions (6 entries).
  - Canonical lists of Lighting conditions (5 entries).
  - Canonical lists of Urban/Rural areas (2 entries).
  - Canonical lists of Collision severities (3 entries).
- **Summary Matrix Blocks**:
  - Aggregates collision counts, casualties, vehicles, and severity ratios dynamically.
  - Slices data by month, day, hour, road type, weather, speed zone, and police force.
  - Feeds chart data ranges on Sheets 1, 2, and 3.

---

## 8. Sheet 8: Map Sample

### 8.1 Purpose & Role
Provides coordinate data for the UK Geographical Collision Scatter Visualization on Sheet 3.

### 8.2 Stratified Sampling Design
To prevent Excel from experiencing rendering lag or workbook sluggishness when plotting 100,000 individual markers on a 2D scatter plot, a representative stratified sample of **7,431 records** was extracted:
- **100% of Fatal Collisions**: All 1,431 fatal records are preserved.
- **2,500 Serious Collisions**: Randomly sampled with seed 42.
- **3,500 Slight Collisions**: Randomly sampled with seed 42.

### 8.3 Dynamic Filtering Logic
- **Columns A–G**: Severity, Month, Weather, Road Type, Road Surface, Lighting, Urban/Rural Area.
- **Columns H–I**: Latitude, Longitude.
- **Column J (Dynamic Filter Logic)**:
  Evaluates whether the sample point matches the 7 active filter criteria from the `Executive Dashboard`.
  - If the record satisfies the active filters, it outputs its geographic coordinates.
  - If the record is filtered out, the formula evaluates to `#N/A`.
  - Native Excel scatter charts automatically ignore `#N/A` values, enabling dynamic coordinate filtering.

