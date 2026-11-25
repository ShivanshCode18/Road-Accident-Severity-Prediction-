# Dashboard Data Dictionary
## UK STATS19 Road Safety Analytics & ML Prediction Attributes

This document defines the schema, data types, value domains, and analytical definitions of all attributes used across the **Road Accident Severity Prediction Excel Dashboard** (`Road_Accident_Severity_Professional_Dashboard.xlsx`) and its supporting processed datasets (`excel_dashboard_dataset.csv`, `excel_dashboard_predictions.csv`, `model_performance_metrics.csv`).

All data definitions adhere strictly to the **UK Department for Transport (DfT) STATS19 Road Safety Data Standards** and project-specific feature engineering rules.

---

## 1. Primary Incident Identification & Temporal Fields

### `collision_index`
- **Data Type**: String (`object` / Text)
- **Description**: Unique 13-character alphanumeric identifier assigned by the UK police force or Department for Transport to uniquely distinguish each collision event.
- **Example**: `"2025010000001"`
- **Integrity**: 100% unique across all 100,000 records; zero nulls; formatted as text to prevent scientific notation conversion.
- **Role in Dashboard**: Primary key and audit trail linkage between `Accident Data` and `Prediction Analysis`. Excluded from machine learning predictors to avoid identifier leakage.

### `date`
- **Data Type**: String / Date (`DD/MM/YYYY`)
- **Description**: Calendar date on which the road traffic collision occurred.
- **Example**: `"15/01/2025"`
- **Domain**: All calendar dates within year 2025 (01/01/2025 to 31/12/2025).
- **Role in Dashboard**: Slicing and temporal aggregation.

### `time`
- **Data Type**: String / Time (`HH:MM`)
- **Description**: Recorded 24-hour time of the collision reported by attending police officers.
- **Example**: `"08:35"`, `"17:42"`
- **Domain**: `00:00` to `23:59`. Zero missing values.

### `month_name`
- **Data Type**: Categorical String
- **Description**: Full English calendar name of the month in which the incident occurred, derived from the `date` attribute.
- **Domain**: `["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]`
- **Role in Dashboard**: Interactive dropdown filter and monthly trend line visualization.

### `day_name`
- **Data Type**: Categorical String
- **Description**: Full English name of the day of the week, derived from `day_of_week` (STATS19 mapping: 1 = Sunday .. 7 = Saturday).
- **Domain**: `["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]`
- **Role in Dashboard**: Day-of-week risk analysis chart; Friday exhibits the highest collision count (15,920+).

### `time_of_day`
- **Data Type**: Categorical String
- **Description**: Binned temporal segment derived from the collision `hour` to reflect diurnal traffic and commuting patterns.
- **Classification Rules**:
  - `Morning Rush`: Hours `07:00` to `09:59`
  - `Midday`: Hours `10:00` to `15:59`
  - `Evening Rush`: Hours `16:00` to `18:59`
  - `Night`: Hours `19:00` to `06:59`
- **Role in Dashboard**: Diurnal risk visualization highlighting peak commuter casualty windows.

---

## 2. Collision Characteristics & Severity Target

### `severity_name`
- **Data Type**: Categorical String
- **Description**: The human-readable classification of the most severe injury sustained in the collision, based on official STATS19 definitions.
- **Domain**:
  - `Fatal`: Collisions resulting in the death of at least one person within 30 days of the incident (Code `1`).
  - `Serious`: Collisions resulting in severe injuries requiring hospital admission (e.g., fractures, severe concussion, internal injuries, severe lacerations) with no deaths (Code `2`).
  - `Slight`: Collisions resulting in minor injuries (e.g., sprains, light bruises, shock) not requiring prolonged medical admission (Code `3`).
- **Distribution in 100k Dataset**:
  - `Fatal`: 1,431 records (1.43%)
  - `Serious`: 24,812 records (24.81%)
  - `Slight`: 73,757 records (73.76%)
- **Role in Dashboard**: Primary target metric across executive KPI cards, charts, and filter controls.

### `number_of_vehicles`
- **Data Type**: Integer (`int64`)
- **Description**: The total number of vehicles involved in the collision (including motor vehicles, bicycles, and horses).
- **Domain**: Integer $\ge 1$. Observed range in dataset: `1` to `17` vehicles. Mean: `1.81` vehicles per incident.
- **Role in Dashboard**: Operational complexity metric and predictive feature in ML modeling.

### `number_of_casualties`
- **Data Type**: Integer (`int64`)
- **Description**: Total number of individuals who sustained fatal, serious, or slight injuries as a direct result of the collision.
- **Domain**: Integer $\ge 1$. Observed range in dataset: `1` to `11` casualties. Total sum across 100k dataset: `125,983` casualties. Mean: `1.26` casualties per incident.
- **Academic Governance & Leakage Note**: This field records post-accident casualty outcomes. It is **strictly excluded** from machine learning model training features to prevent direct target leakage. It is retained exclusively in dashboard datasets for KPI aggregation and trauma burden reporting.

---

## 3. Infrastructure, Environmental & Spatial Fields

### `road_type_name`
- **Data Type**: Categorical String
- **Description**: Official categorization of the roadway geometry at the point of collision.
- **Domain**:
  - `Single carriageway` (Code `6`): Undivided roadway with traffic moving in opposite directions (~74.8% of crashes).
  - `Dual carriageway` (Code `3`): Separated roadway with a physical median strip.
  - `Roundabout` (Code `1`): Circular intersection.
  - `One way street` (Code `2`): Single direction roadway.
  - `Slip road` (Code `7`): Entry or exit ramp connecting grade-separated highways.
  - `Unknown` / `Missing` (Codes `9`, `-1`): Unclassified roadway types.
- **Role in Dashboard**: Infrastructure risk analysis and dropdown filter.

### `urban_rural_name`
- **Data Type**: Categorical String
- **Description**: Settlement classification of the collision site based on the official UK Office for National Statistics (ONS) urban/rural definition.
- **Domain**:
  - `Urban`: Built-up settlements with a resident population of 10,000 or more (`66,127` collisions, 66.13%).
  - `Rural`: Small towns, villages, hamlets, and open country roads (`33,871` collisions, 33.87%).
- **Role in Dashboard**: Core geographical risk comparison; demonstrates that rural routes produce disproportionately high fatal crash rates due to speed differentials.

### `weather_group`
- **Data Type**: Categorical String
- **Description**: Consolidated meteorological conditions at the scene during the collision, mapped from STATS19 detailed weather codes.
- **Domain**:
  - `Fine`: Clear conditions without severe atmospheric precipitation.
  - `Raining`: Active rain or drizzle (with or without high winds).
  - `Snowing`: Active snowfall or blizzard conditions.
  - `Fog or mist`: Reduced visibility atmospheric fog.
  - `Other` / `Unknown`: Unclassified or missing meteorological observations.
- **Role in Dashboard**: Environmental hazard diagnosis and filter control.

### `road_surface_group`
- **Data Type**: Categorical String
- **Description**: Physical condition and traction state of the carriageway surface at the moment of impact.
- **Domain**:
  - `Dry`: Normal, dry road surface without moisture.
  - `Wet or damp`: Rainwater or moisture on pavement.
  - `Snow`: Packed or loose snow on carriageway.
  - `Frost or ice`: Glaze, black ice, or frost reducing tire adhesion.
  - `Flood`: Standing water deeper than 3 cm.
  - `Other` / `Unknown`: Mud, oil, diesel spillages, or missing records.
- **Role in Dashboard**: Road safety maintenance analysis and filter control.

### `light_group`
- **Data Type**: Categorical String
- **Description**: Natural ambient and artificial lighting conditions at the collision scene.
- **Domain**:
  - `Daylight`: Natural daytime lighting conditions.
  - `Darkness - lights lit`: Nighttime with operating street luminaires.
  - `Darkness - lights unlit`: Nighttime with defective or switched-off street lights.
  - `Darkness - no lighting`: Nighttime along unlit rural roads or motorways.
  - `Darkness - lighting unknown`: Nighttime with unverified lighting state.
- **Role in Dashboard**: Visibility hazard evaluation and filter control.

### `speed_zone`
- **Data Type**: Categorical String
- **Description**: Engineered speed band grouping posted speed limits into analytical danger categories.
- **Classification Rules**:
  - `Urban Low Speed`: Speed limits of `20 mph` or `30 mph`. Characterized by high pedestrian density and lower collision kinetic energy.
  - `Moderate Speed`: Speed limits of `40 mph` or `50 mph`. Semi-urban arterial corridors.
  - `High Speed`: Speed limits of `60 mph` (rural single carriageway national limit) or `70 mph` (dual carriageway and motorway national limit). High fatality rate corridors.
- **Role in Dashboard**: Speed limit impact analysis.

### `latitude` and `longitude`
- **Data Types**: Floating-point numeric (`float64`)
- **Description**: WGS84 geographic coordinates of the collision point, transformed from British National Grid Easting/Northing coordinates.
- **Domain**:
  - `latitude`: `49.91253` to `60.34525` (Valid UK bounding box: `49.5` to `61.0`)
  - `longitude`: `-7.41772` to `1.75559` (Valid UK bounding box: `-8.5` to `2.0`)
- **Integrity**: Zero missing values; 2 raw invalid coordinate records were removed during Phase 1 cleaning.
- **Role in Dashboard**: Drives the Great Britain collision scatter visualization on Sheet 3 (`Geographical Analysis`) via the stratified sample on Sheet 8 (`Map Sample`).

---

## 4. Machine Learning & Prediction Diagnostics Fields

*(Applies to `excel_dashboard_predictions.csv` and Sheet 5 `Prediction Analysis` across the 20,099 held-out test partition)*

### `actual_severity_name`
- **Data Type**: Categorical String
- **Description**: Ground-truth severity outcome recorded in the official collision report.
- **Domain**: `["Fatal", "Serious", "Slight"]`
- **Test Partition Distribution**: Fatal = 306 (1.52%), Serious = 4,843 (24.10%), Slight = 14,950 (74.38%). Total = 20,099.

### `predicted_severity_name`
- **Data Type**: Categorical String
- **Description**: Discrete class predicted by the trained Apache Spark MLlib classifier (Multinomial Logistic Regression with inverse class weighting).
- **Domain**: `["Fatal", "Serious", "Slight"]`
- **Predictions in Test Partition**: Fatal = 5,097 (25.36%), Serious = 5,499 (27.36%), Slight = 9,503 (47.28%).
- **Safety Orientation**: Demonstrates that class weighting intentionally shifts boundary thresholds to over-flag potential fatal scenarios, avoiding fatal False Negatives.

### `prediction_correct`
- **Data Type**: Binary Integer (`1` or `0`)
- **Description**: Correctness indicator for individual predictions.
- **Formula**: `IF(actual_severity == predicted_severity, 1, 0)`
- **Aggregate Values in Test Set**:
  - Correct (`1`): `9,532` records (`47.43%`)
  - Incorrect (`0`): `10,567` records (`52.57%`)
- **Role in Dashboard**: Model accuracy KPI and audit filtering on Sheet 5.

### `prob_fatal`
- **Data Type**: Float (`0.0000` to `1.0000`)
- **Description**: Soft softmax output probability assigned by the classifier indicating the estimated likelihood of the collision being fatal.
- **Test Set Mean**: `0.2703` (27.03%)
- **Interpretation**: Higher values signal extreme risk profiles (e.g., 60 mph unlit rural road head-on collisions).

### `prob_serious`
- **Data Type**: Float (`0.0000` to `1.0000`)
- **Description**: Soft softmax output probability assigned by the classifier indicating the estimated likelihood of serious injury.
- **Test Set Mean**: `0.3452` (34.52%)

### `prob_slight`
- **Data Type**: Float (`0.0000` to `1.0000`)
- **Description**: Soft softmax output probability assigned by the classifier indicating the estimated likelihood of slight injury.
- **Test Set Mean**: `0.3845` (38.45%)

### Simplex Invariant
Across all 20,099 records, the predicted probability distribution satisfies the unit simplex constraint:
$$\text{prob\_fatal} + \text{prob\_serious} + \text{prob\_slight} = 1.0000 \pm 10^{-4}$$

