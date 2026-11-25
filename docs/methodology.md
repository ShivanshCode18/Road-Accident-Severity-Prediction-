# Research Methodology & Technical Formulation
## Road Accident Severity Prediction Using Apache Spark and Machine Learning with Excel Dashboard

---

## 1. Dataset Acquisition & Provenance

### 1.1 Data Source
The empirical foundation of this project is the official **UK Department for Transport (DfT) STATS19 Road Casualty Statistics (2025 Collisions Dataset)**. STATS19 is the United Kingdom's national standard system for recording road accidents resulting in personal injury. Reports are completed by attending police officers or submitted by members of the public through official reporting portals.

### 1.2 Raw Dataset Dimensions
- **Initial Ingested Records**: `101,525` rows.
- **Initial Attributes**: `44` raw columns.
- **Geographic Coverage**: England, Scotland, and Wales (Great Britain).
- **Temporal Horizon**: Calendar year 2025.

---

## 2. Data Cleaning & Quality Assurance

### 2.1 Coordinate Validation & Spatial Filtering
Road collision coordinates in STATS19 are recorded in both British National Grid (OSGR Easting/Northing) and WGS84 Geodetic coordinates (Latitude/Longitude). To eliminate invalid geographical records:
- Each record was evaluated against the bounding box of Great Britain ($49.5^\circ\text{N} \le \text{Latitude} \le 61.0^\circ\text{N}$, $-8.5^\circ\text{W} \le \text{Longitude} \le 2.0^\circ\text{E}$).
- Exactly **2 records** contained missing coordinates (`NaN`) and were permanently removed, leaving `101,523` valid records.

### 2.2 Duplicate Record Analysis
Uniqueness testing was conducted on `collision_index`. Zero duplicate rows or duplicate keys were identified, confirming 100% incident uniqueness.

### 2.3 Domain Validation
- **Target Severity Domain**: Checked against official categories $\{1: \text{Fatal}, 2: \text{Serious}, 3: \text{Slight}\}$. Zero out-of-domain severity values were observed.
- **Speed Limit Domain**: Validated against statutory UK limits $\{20, 30, 40, 50, 60, 70\text{ mph}\}$. 100% of records complied with valid speed limits.
- **Missing Value Handling**: Deprecated historic fields ($>86\%$ missing codes) were eliminated. The final processed dataset contains zero `NaN` values across all columns.

---

## 3. Strict Target Data Leakage Elimination

In traffic safety modeling, data leakage is a pervasive methodological error where information generated *during* or *after* the collision is inadvertently used to predict the crash severity. Such models yield artificially high accuracy during training but fail completely in real-world decision support.

The pipeline strictly isolates and excludes the following 18 columns from predictive ML modeling:

| Column Name | Category | Academic Justification for Exclusion |
|---|---|---|
| `number_of_casualties` | Post-Crash Outcome | The number of casualties is a direct outcome of crash severity. Retained in dashboard KPI reporting, excluded from predictive features. |
| `enhanced_severity_collision` | Direct Target Leakage | DfT post-hoc alternative severity classification. |
| `collision_injury_based` | Direct Target Leakage | Records whether severity was verified through hospital injury coding. |
| `collision_adjusted_severity_serious` | Target Leakage Probability | Statistical model adjustment probability for serious severity. |
| `collision_adjusted_severity_slight` | Target Leakage Probability | Statistical model adjustment probability for slight severity. |
| `did_police_officer_attend_scene_of_accident` | Post-Crash Response | Reflects post-crash administrative attendance rather than antecedent risk. |
| `local_authority_district` | Deprecated Field | 100% missing values (-1); replaced by modern ONS district codes. |
| `junction_detail_historic` | Deprecated Field | >86% missing; superseded by modern junction detail. |
| `pedestrian_crossing_human_control_historic` | Deprecated Field | >86% missing; obsolete historical attribute. |
| `pedestrian_crossing_physical_facilities_historic` | Deprecated Field | >86% missing; obsolete historical attribute. |
| `special_conditions_at_site` | Deprecated Field | >86% missing; superseded by carriageway hazard fields. |
| `carriageway_hazards_historic` | Deprecated Field | >86% missing; obsolete historical attribute. |
| `collision_year` | Zero Variance Constant | Constant value (2025) across all records; provides zero predictive signal. |
| `collision_index` | Unique Identifier | High-cardinality primary key; retained in dashboard for auditability, excluded from ML features. |
| `collision_ref_no` | Redundant Identifier | Secondary key redundant with `collision_index`. |
| `location_easting_osgr` / `northing_osgr` | Collinear Spatial Field | Perfectly collinear with WGS84 longitude and latitude. |
| `first_road_number` / `second_road_number` | High Cardinality Identifier | Specific route numbers causing severe overfitting; abstracted to `first_road_class`. |
| `lsoa_of_accident_location` | Extreme Cardinality Identifier | Census Lower Layer Super Output Area (>30,000 distinct codes). |

---

## 4. Feature Engineering

Fourteen dedicated spatial, temporal, calendar, and environmental features were engineered:

1. `month`: Integer extraction (1 to 12) from `date`.
2. `month_name`: Full string month name (`January` .. `December`).
3. `hour`: Integer extraction (0 to 23) from `time`.
4. `time_of_day`: Binned day partition (`Morning Rush`, `Midday`, `Evening Rush`, `Night`).
5. `day_name`: Full string day name (`Sunday` .. `Saturday`).
6. `weekend_indicator`: Binary flag (1 for Saturday/Sunday, 0 for Monday–Friday).
7. `weekday_weekend`: String label (`Weekday`, `Weekend`).
8. `severity_name`: Categorical mapping (`Fatal`, `Serious`, `Slight`).
9. `road_type_name`: Descriptive road geometry classification.
10. `urban_rural_name`: Descriptive area classification (`Urban`, `Rural`).
11. `weather_group`: Consolidated meteorological category.
12. `road_surface_group`: Consolidated pavement condition.
13. `light_group`: Consolidated ambient/artificial illumination category.
14. `speed_zone`: Grouped speed bands (`Urban Low Speed`, `Moderate Speed`, `High Speed`).

---

## 5. Stratified Sampling Design

To establish a standard benchmark dataset for distributed machine learning and Excel visualization:
- Cleaned valid population: `101,523` rows.
- Target sample size: Exactly `100,000` rows.
- Sampling algorithm: Scikit-learn stratified random sampling on `collision_severity` with fixed random seed `42`.
- Preserved Class Distribution:
  - **Fatal (Class 1)**: `1,431` rows (`1.43%`)
  - **Serious (Class 2)**: `24,812` rows (`24.81%`)
  - **Slight (Class 3)**: `73,757` rows (`73.76%`)

---

## 6. Machine Learning Formulation & The Accuracy Paradox

### 6.1 The Accuracy Paradox in Public Safety
In real-world collision datasets, class imbalance is extreme: less than 1.5% of crashes result in fatalities. A naive or unweighted machine learning algorithm easily maximizes raw accuracy by predicting the majority class ("Slight") for 100% of cases:
$$\text{Naive Accuracy} = \frac{73,757}{100,000} \approx 73.76\%$$
However, this naive classifier achieves **0.0% Fatal Accident Recall**, completely failing to detect a single fatal incident. In life-critical transportation safety, missing a fatal crash (False Negative) has catastrophic consequences, whereas misclassifying a serious crash as potentially fatal (False Positive) merely warrants heightened caution.

### 6.2 Inverse Frequency Class Weighting
To force the learning algorithms to pay proportional attention to the rare fatal class, mathematical class weights were computed:
$$w_c = \frac{N}{K \cdot N_c}$$
Where:
- $N = 79,901$ (Total training instances)
- $K = 3$ (Number of severity classes)
- $N_c$ = Total training instances belonging to class $c$

Computed weights:
- $w_{\text{Fatal}} = \frac{79,901}{3 \times 1,125} \approx 23.67$
- $w_{\text{Serious}} = \frac{79,901}{3 \times 19,969} \approx 1.33$
- $w_{\text{Slight}} = \frac{79,901}{3 \times 58,807} \approx 0.45$

These weights were incorporated into the Multinomial Logistic Regression loss objective, shifting the classification boundary to aggressively flag high-risk fatal scenarios.

### 6.3 Train-Test Partitioning
The 100,000 records were partitioned into disjoint subsets using stratified random splitting:
- **Training Partition (80%)**: `79,901` records.
- **Testing Partition (20%)**: `20,099` records (Held-out and untouched during training).

### 6.4 Model Evaluation Results on Held-Out Test Set (20,099 Records)

| Model Name | Accuracy | Weighted F1 | Macro F1 | Fatal Recall | Fatal Precision | Fatal F1 | Train Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** (Class-Weighted) | 47.43% | 0.5464 | 0.3383 | **61.11%** | 0.0367 | 0.0692 | **9.54s** |
| **Decision Tree** (maxDepth=8) | 46.02% | 0.5346 | 0.3257 | 59.48% | 0.0333 | 0.0631 | 10.04s |
| **Random Forest** (50 Trees, maxDepth=10) | **54.50%** | **0.5985** | **0.3667** | 54.90% | **0.0439** | **0.0814** | 57.59s |

### 6.5 Model Selection Justification
- **Random Forest** achieved the highest overall accuracy (54.50%), highest weighted F1 (0.5985), and highest macro F1 (0.3667).
- However, **Multinomial Logistic Regression** achieved the highest **Fatal Accident Recall (61.11%)**, successfully detecting 187 out of 306 actual fatal incidents in the test set.
- Consequently, Class-Weighted Logistic Regression was selected as the **primary public safety model**, and its predictions were exported to drive the diagnostic analysis in the Excel dashboard.

