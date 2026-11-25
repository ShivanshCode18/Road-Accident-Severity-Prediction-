# Master Power BI Implementation Guide: UK Road Accident Severity Analytics & Machine Learning Dashboard

**Project**: Road Accident Severity Prediction Using Apache Spark (PySpark) and Power BI  
**Author**: Data Science & Big Data Engineering Team  
**Tool**: Microsoft Power BI Desktop (Version 2024 / 2025 / 2026 Compatible)  
**Theme**: Custom Dark Navy / Charcoal Executive Theme (`uk_road_safety_theme.json`)  

---

## PART 1: Getting Started & Data Ingestion

### Step 1.1: Launch Power BI Desktop
1. Open **Microsoft Power BI Desktop** on your computer.
2. If the startup splash screen appears, click **Close** (X) to go directly to the blank canvas.

---

### Step 1.2: Import Data Source 1 — `FactCollisions` (`powerbi_dataset.csv`)
1. In the top ribbon, click **Home** > **Get Data** > **Text/CSV**.
2. Navigate to your project folder:
   `D:\ApacheSpark Project\Road_Accident_Severity_Prediction\dataset\processed\`
3. Select **`powerbi_dataset.csv`** and click **Open**.
4. In the preview window:
   - File Origin: `65001: Unicode (UTF-8)`.
   - Delimiter: `Comma`.
   - Data Type Detection: `Based on first 200 rows`.
5. Click **Transform Data** (this opens the Power Query Editor).
6. In Power Query, rename the query on the right pane from `powerbi_dataset` to **`FactCollisions`**.
7. Confirm data types:
   - `collision_index`: **Text**
   - `date`: **Date** (or Text in `DD/MM/YYYY`)
   - `time`: **Text**
   - `collision_severity`: **Whole Number** (1, 2, 3)
   - `number_of_vehicles`: **Whole Number**
   - `number_of_casualties`: **Whole Number**
   - `latitude`: **Decimal Number**
   - `longitude`: **Decimal Number**
   - `speed_limit`: **Whole Number**

---

### Step 1.3: Import Data Source 2 — `FactPredictions` (`powerbi_predictions.csv`)
1. In the Power Query Editor ribbon, click **New Source** > **Text/CSV**.
2. Select **`powerbi_predictions.csv`** and click **Open**.
3. Rename the query to **`FactPredictions`**.
4. Confirm data types:
   - `collision_index`: **Text**
   - `actual_severity`: **Whole Number**
   - `predicted_severity`: **Whole Number**
   - `prediction_correct`: **Whole Number** (1 or 0)
   - `prob_fatal`: **Decimal Number**
   - `prob_serious`: **Decimal Number**
   - `prob_slight`: **Decimal Number**
   - `latitude`: **Decimal Number**
   - `longitude`: **Decimal Number**
   - `number_of_casualties`: **Whole Number**

---

### Step 1.4: Import Data Source 3 — `DimModelPerformance` (`powerbi_model_performance.csv`)
1. Click **New Source** > **Text/CSV**.
2. Select **`powerbi_model_performance.csv`** and click **Open**.
3. Rename the query to **`DimModelPerformance`**.
4. Confirm that numerical metrics (`accuracy`, `weighted_f1`, `macro_f1`, `fatal_recall`, `train_time_sec`) are set to **Decimal Number**.

---

### Step 1.5: Close & Apply
1. Click **Close & Apply** in the top-left corner of Power Query.
2. Wait a few seconds while Power BI loads the 100,000 rows for `FactCollisions`, 20,099 rows for `FactPredictions`, and 3 rows for `DimModelPerformance`.

---

## PART 2: Star Schema Modeling & Dimension Creation

### Step 2.1: Create Dimension Table `DimSeverity`
In Power BI Desktop, click **Modeling** > **New Table**, and enter the following DAX:

```dax
DimSeverity = 
DATATABLE(
    "SeverityCode", INTEGER,
    "SeverityName", STRING,
    "SeverityColorHex", STRING,
    "SortOrder", INTEGER,
    {
        {1, "Fatal", "#EF4444", 1},
        {2, "Serious", "#F59E0B", 2},
        {3, "Slight", "#10B981", 3}
    }
)
```

1. Select the `SeverityName` column in the Data pane.
2. In the top ribbon, click **Column Tools** > **Sort by Column** > **`SortOrder`**.

---

### Step 2.2: Create Dedicated `_Measures` Table
1. Click **Home** > **Enter Data**.
2. Name the table **`_Measures`**.
3. Name Column1 `Placeholder` and click **Load**.
*(Once you create your first DAX measure in this table, right-click `Placeholder` and select **Delete from model** so that `_Measures` displays a calculator icon at the very top of your Field list).*

---

### Step 2.3: Set Up Relationships in Model View
Switch to the **Model View** (icon on the far left navigation bar). Drag and drop fields to configure the following relationships:

1. **`DimSeverity` to `FactCollisions`**:
   - Drag `DimSeverity[SeverityCode]` $\rightarrow$ `FactCollisions[collision_severity]`.
   - Cardinality: **One to many (1:*)**.
   - Cross filter direction: **Single**.
   - Active: **Checked**.

2. **`DimSeverity` to `FactPredictions`**:
   - Drag `DimSeverity[SeverityCode]` $\rightarrow$ `FactPredictions[actual_severity]`.
   - Cardinality: **One to many (1:*)**.
   - Cross filter direction: **Single**.
   - Active: **Checked**.

3. **`FactCollisions` to `FactPredictions`**:
   - Drag `FactCollisions[collision_index]` $\rightarrow$ `FactPredictions[collision_index]`.
   - Cardinality: **One to one (1:1)** or **One to many (1:*)**.
   - Cross filter direction: **Single** (`FactCollisions` filters `FactPredictions`).
   - Active: **Checked**.

---

### Step 2.4: Configure Geographic Data Categories
1. Switch to the **Data View** (table icon on the left bar).
2. Select table **`FactCollisions`**:
   - Click column **`latitude`** $\rightarrow$ in the ribbon, set **Data Category** to **Latitude**.
   - Click column **`longitude`** $\rightarrow$ in the ribbon, set **Data Category** to **Longitude**.
3. Select table **`FactPredictions`**:
   - Set **`latitude`** $\rightarrow$ **Latitude**.
   - Set **`longitude`** $\rightarrow$ **Longitude**.

---

## PART 3: Import Custom Dark Navy Theme

1. In the top ribbon, navigate to **View** > click the dropdown arrow under **Themes**.
2. Click **Browse for themes...**.
3. Select the file:
   `D:\ApacheSpark Project\Road_Accident_Severity_Prediction\powerbi\uk_road_safety_theme.json`
4. Click **Open**. Power BI will display: *"Theme imported successfully"*.
*(All canvas pages will now instantly adopt a dark navy background `#0F172A`, slate containers `#1E293B`, rounded cards, and high-contrast typography).*

---

## PART 4: Enter Core DAX Measures

In the **_Measures** table, click **New Measure** and add the following essential DAX measures (refer to [`powerbi/DAX_MEASURES.md`](file:///D:/ApacheSpark%20Project/Road_Accident_Severity_Prediction/powerbi/DAX_MEASURES.md) for the complete library of 39 measures):

### Core KPIs:
```dax
Total Collisions = COUNTROWS('FactCollisions')
```
```dax
Total Casualties = SUM('FactCollisions'[number_of_casualties])
```
```dax
Fatal Collisions = CALCULATE([Total Collisions], 'FactCollisions'[collision_severity] = 1)
```
```dax
Serious Collisions = CALCULATE([Total Collisions], 'FactCollisions'[collision_severity] = 2)
```
```dax
Slight Collisions = CALCULATE([Total Collisions], 'FactCollisions'[collision_severity] = 3)
```
```dax
Fatal Collision % = DIVIDE([Fatal Collisions], [Total Collisions], 0)
```
```dax
Serious Collision % = DIVIDE([Serious Collisions], [Total Collisions], 0)
```
```dax
Slight Collision % = DIVIDE([Slight Collisions], [Total Collisions], 0)
```
```dax
Avg Casualties per Collision = DIVIDE([Total Casualties], [Total Collisions], 0)
```

### Machine Learning Metrics:
```dax
Total Test Predictions = COUNTROWS('FactPredictions')
```
```dax
Correct Predictions = CALCULATE([Total Test Predictions], 'FactPredictions'[prediction_correct] = 1)
```
```dax
Prediction Accuracy = DIVIDE([Correct Predictions], [Total Test Predictions], 0)
```
```dax
Actual Fatal Collisions = CALCULATE([Total Test Predictions], 'FactPredictions'[actual_severity] = 1)
```
```dax
Fatal True Positives = CALCULATE([Total Test Predictions], 'FactPredictions'[actual_severity] = 1, 'FactPredictions'[predicted_severity] = 1)
```
```dax
Fatal Detection Rate (Recall) = DIVIDE([Fatal True Positives], [Actual Fatal Collisions], 0)
```
```dax
Fatal Precision = DIVIDE([Fatal True Positives], CALCULATE([Total Test Predictions], 'FactPredictions'[predicted_severity] = 1), 0)
```

---

## PART 5: Step-by-Step Canvas Construction for All 4 Pages

Set Page Canvas size to **16:9** (1280 x 720 px or 1920 x 1080 px).

---

### PAGE 1: ROAD ACCIDENT SEVERITY ANALYTICS (Executive Overview)

**Page Name**: Rename tab to `Executive Overview`.

#### 1. Header Banner:
- Add a **Text Box** across the top ($X=20, Y=15, W=1240, H=45$):
  - Text: **UK ROAD ACCIDENT SEVERITY ANALYTICS (STATS19 2025)**
  - Font: `Segoe UI Semibold`, Size `18pt`, Color `#F8FAFC`.
  - Subtitle: *"Big Data Analytics & Machine Learning Decision Support System | Sample N = 100,000 Valid Collisions"* (Size `10pt`, Color `#94A3B8`).

#### 2. Top KPI Cards Row ($Y=70, H=85$):
Create 5 separate **Card** visuals side-by-side:
- **Card 1**: Field = `[Total Collisions]` (Label: "TOTAL ACCIDENTS", Displays: **100,000**, Accent: Cyan `#38BDF8`).
- **Card 2**: Field = `[Fatal Collisions]` (Label: "FATAL ACCIDENTS (1.43%)", Displays: **1,431**, Accent: Red `#EF4444`).
- **Card 3**: Field = `[Serious Collisions]` (Label: "SERIOUS ACCIDENTS (24.81%)", Displays: **24,812**, Accent: Orange `#F59E0B`).
- **Card 4**: Field = `[Slight Collisions]` (Label: "SLIGHT ACCIDENTS (73.76%)", Displays: **73,757**, Accent: Green `#10B981`).
- **Card 5**: Field = `[Total Casualties]` (Label: "TOTAL CASUALTIES", Displays: **125,983**, Accent: Purple `#A855F7`).

#### 3. Slicers Panel (Left Column, $X=20, W=190$):
Add 4 **Slicer** visuals vertically:
- Slicer 1: `DimSeverity[SeverityName]` (or `FactCollisions[severity_name]`) — Tile or Vertical List.
- Slicer 2: `FactCollisions[month_name]` — Dropdown.
- Slicer 3: `FactCollisions[day_name]` — Dropdown.
- Slicer 4: `FactCollisions[urban_rural_name]` — Tile (`Urban`, `Rural`).

#### 4. Visual 1 — Donut Chart: Collision Severity Distribution
- Position: $X=220, Y=170, W=320, H=250$.
- Visual: **Donut Chart**.
- Legend: `FactCollisions[severity_name]`.
- Values: `[Total Collisions]`.
- Colors: Set Fatal to `#EF4444`, Serious to `#F59E0B`, Slight to `#10B981`.
- Detail Labels: **Category, Percent of total**.

#### 5. Visual 2 — Clustered Column Chart: Collisions by Weekday
- Position: $X=555, Y=170, W=400, H=250$.
- Visual: **Clustered Column Chart**.
- X-Axis: `FactCollisions[day_name]`.
- Y-Axis: `[Total Collisions]`.
- Data Color: `#06B6D4` (Highlights Friday peak ~16.5k).

#### 6. Visual 3 — Line Chart: Monthly Collision Trends
- Position: $X=970, Y=170, W=290, H=250$.
- Visual: **Line Chart**.
- X-Axis: `FactCollisions[month_name]`.
- Y-Axis: `[Total Collisions]`.
- Line stroke: `#38BDF8`, Width `3pt`, Markers ON.

#### 7. Visual 4 — Bar Chart: Collisions by Time of Day
- Position: $X=220, Y=435, W=450, H=265$.
- Visual: **Clustered Bar Chart**.
- Y-Axis: `FactCollisions[time_of_day]` (`Afternoon`, `Evening`, `Morning`, `Night`).
- X-Axis: `[Total Collisions]`.
- Data labels: ON.

#### 8. Visual 5 — Stacked Column Chart: Severity Distribution by Urban/Rural
- Position: $X=685, Y=435, W=575, H=265$.
- Visual: **100% Stacked Column Chart**.
- X-Axis: `FactCollisions[urban_rural_name]`.
- Legend: `FactCollisions[severity_name]`.
- Y-Axis: `[Total Collisions]`.
- *Key Takeaway*: Clearly shows that Rural roads have more than double the proportion of Fatal crashes compared to Urban roads.

---

### PAGE 2: ROAD CONDITIONS & ENVIRONMENTAL RISK ANALYSIS

**Page Name**: Rename tab to `Environmental Risk`.

#### 1. Header Banner & Slicers Bar:
- Top Title: **ROAD CONDITIONS & ENVIRONMENTAL HAZARD ANALYSIS**
- Top Slicers Row:
  - Slicer 1: `FactCollisions[weather_group]` (`Fine`, `Raining`, `Snow/Ice/Fog`, `Other/Unknown`).
  - Slicer 2: `FactCollisions[road_surface_group]` (`Dry`, `Wet/Flood`, `Snow/Ice`).
  - Slicer 3: `FactCollisions[light_group]` (`Daylight`, `Darkness - Lights Lit`, `Darkness - Unlit/No Lights`).
  - Slicer 4: `FactCollisions[speed_zone]` (`<=20 mph`, `30 mph`, `40-50 mph`, `60-70 mph`).

#### 2. Visual 1 — Clustered Bar Chart: Collisions by Weather Group
- Visual: **Clustered Bar Chart**.
- Y-Axis: `FactCollisions[weather_group]`.
- X-Axis: `[Total Collisions]`.
- Color: `#38BDF8`.

#### 3. Visual 2 — 100% Stacked Bar Chart: Severity Proportion by Weather
- Visual: **100% Stacked Bar Chart**.
- Y-Axis: `FactCollisions[weather_group]`.
- Legend: `FactCollisions[severity_name]`.
- Values: `[Total Collisions]`.
- Colors: Fatal (`#EF4444`), Serious (`#F59E0B`), Slight (`#10B981`).

#### 4. Visual 3 — Clustered Column Chart: Road Surface Conditions
- Visual: **Clustered Column Chart**.
- X-Axis: `FactCollisions[road_surface_group]`.
- Y-Axis: `[Total Collisions]`.
- Tooltip: `[Fatal Collision %]`, `[Total Casualties]`.

#### 5. Visual 4 — 100% Stacked Bar Chart: Visibility & Lighting Risk
- Visual: **100% Stacked Bar Chart**.
- Y-Axis: `FactCollisions[light_group]`.
- Legend: `FactCollisions[severity_name]`.
- Values: `[Total Collisions]`.
- *Academic Insight*: Unlit darkness shows severe fatality rate surge.

#### 6. Visual 5 — Clustered Column Chart: Severity Rate by Speed Limit
- Visual: **Line and Clustered Column Chart**.
- X-Axis: `FactCollisions[speed_limit]`.
- Column Y-Axis: `[Total Collisions]`.
- Line Y-Axis: `[Severe Collision Percentage]`.
- *Takeaway*: Highlights that while 30 mph has highest volume, 60 mph has highest severity rate.

#### 7. Visual 6 — Treemap: Collisions by Road Type
- Visual: **Treemap**.
- Category: `FactCollisions[road_type_name]`.
- Values: `[Total Collisions]`.
- Tooltip: `[Total Casualties]`.

---

### PAGE 3: UK ROAD ACCIDENT HOTSPOT ANALYSIS (Geographical Mapping)

**Page Name**: Rename tab to `Geospatial Intelligence`.

#### 1. Header Banner & Filter Row:
- Title: **GEOSPATIAL COLLISION HOTSPOT INTELLIGENCE (GREAT BRITAIN)**
- Slicers:
  - Severity: `DimSeverity[SeverityName]` (Red Fatal / Orange Serious / Green Slight).
  - Area: `FactCollisions[urban_rural_name]`.
  - Speed Limit: `FactCollisions[speed_limit]`.

#### 2. Visual 1 — Main Interactive Map (Center Canvas, $W=820, H=590$):
- Visual: **Map** (or **Azure Map**).
- Latitude: `FactCollisions[latitude]` (ensure Data Category = Latitude).
- Longitude: `FactCollisions[longitude]` (ensure Data Category = Longitude).
- Size: `FactCollisions[number_of_casualties]`.
- Legend: `FactCollisions[severity_name]`.
- Colors: Fatal = `#EF4444`, Serious = `#F59E0B`, Slight = `#10B981`.
- Map Controls: Set Map Style to **Dark** or **Night** mode to match the theme.
- Map Settings: Turn **Zoom buttons ON**, **Map Clustering ON** (clusters dense points in London, Birmingham, Manchester, and Leeds into readable count clusters).

#### 3. Visual 2 — Top Police Force / Regions Bar Chart ($X=860, Y=80, W=400, H=290$):
- Visual: **Clustered Bar Chart**.
- Y-Axis: `FactCollisions[police_force]`.
- X-Axis: `[Total Collisions]`.
- Filter: Top 10 by `[Total Collisions]`.

#### 4. Visual 3 — Decomposition Tree ($X=860, Y=385, W=400, H=290$):
- Visual: **Decomposition Tree**.
- Analyze: `[Total Collisions]`.
- Explain By: `urban_rural_name` $\rightarrow$ `road_type_name` $\rightarrow$ `speed_zone` $\rightarrow$ `severity_name`.
- *Allows examiners to click and drill through from nationwide totals down to specific hazard segments!*

---

### PAGE 4: ROAD ACCIDENT SEVERITY PREDICTION PERFORMANCE (ML Center)

> **IMPORTANT**: This is the flagship technical page demonstrating your Apache Spark MLlib modeling accomplishments.

**Page Name**: Rename tab to `ML Model Performance`.

#### 1. Header Banner:
- Text: **APACHE SPARK (PYSPARK) MACHINE LEARNING PREDICTION PERFORMANCE**
- Subtitle: *"Multiclass Classification on 20,099 Untouched STATS19 Test Records | Balanced Inverse-Frequency Class Weighting"*

#### 2. Top Model KPI Cards Row ($Y=70, H=85$):
- **Card 1**: Field = `[Total Test Predictions]` (Displays: **20,099**).
- **Card 2**: Field = `[Prediction Accuracy]` (Displays: **47.43%**, Sublabel: "OVERALL ACCURACY").
- **Card 3**: Field = `[Fatal Detection Rate (Recall)]` (Displays: **61.11%**, Accent: `#EF4444`, Sublabel: "FATAL RECALL (PRIMARY SAFETY KPI)").
- **Card 4**: Field = `[Fatal Precision]` (Displays: **3.67%**, Sublabel: "FATAL PRECISION").
- **Card 5**: Field = `[Macro F1]` (Displays: **0.3383**, Sublabel: "MACRO F1-SCORE").

#### 3. Visual 1 — Model Comparison Clustered Bar Chart ($X=20, Y=170, W=420, H=260$):
- Data Source: Table **`DimModelPerformance`**.
- Visual: **Clustered Bar Chart**.
- Y-Axis: `DimModelPerformance[model_name]` (`Logistic Regression`, `Decision Tree`, `Random Forest`).
- X-Axis: Add `accuracy`, `macro_f1`, `fatal_recall`.
- *Key Takeaway Visualized*:
  - Random Forest achieves higher accuracy (54.50%).
  - But Logistic Regression achieves the highest Fatal Recall (61.11%).

#### 4. Visual 2 — 3x3 Confusion Matrix Visual ($X=460, Y=170, W=420, H=260$):
- Data Source: Table **`FactPredictions`**.
- Visual: **Matrix**.
- Rows: `FactPredictions[actual_severity_name]` (`Fatal`, `Serious`, `Slight`).
- Columns: `FactPredictions[predicted_severity_name]` (`Fatal`, `Serious`, `Slight`).
- Values: `COUNT('FactPredictions'[collision_index])`.
- In Matrix Formatting: Turn ON **Conditional Formatting** > **Background Color** (Light blue to dark navy gradient based on cell count).
- *Displays exact ground truth*:
  - Fatal row: 187 True Fatal, 63 Serious, 56 Slight.
  - Serious row: 1,493 Fatal, 1,624 Serious, 1,726 Slight.
  - Slight row: 3,417 Fatal, 3,812 Serious, 7,721 Slight.

#### 5. Visual 3 — Probability Distribution Scatter / Area Chart ($X=900, Y=170, W=360, H=260$):
- Data Source: **`FactPredictions`**.
- Visual: **Line Chart** or **Scatter Chart**.
- X-Axis: `FactPredictions[hour]`.
- Y-Axis:
  - `[Avg Predicted Fatal Probability]` (Color `#EF4444`)
  - `[Avg Predicted Serious Probability]` (Color `#F59E0B`)
  - `[Avg Predicted Slight Probability]` (Color `#10B981`)
- *Insight*: Shows how the model increases Fatal and Serious probabilities during evening and late night hours.

#### 6. Visual 4 — Granular Accident Prediction Audit Table ($X=20, Y=445, W=860, H=255$):
- Visual: **Table**.
- Columns:
  1. `FactPredictions[collision_index]`
  2. `FactPredictions[actual_severity_name]`
  3. `FactPredictions[predicted_severity_name]`
  4. `FactPredictions[prediction_correct]` (Formatted with Green check / Red cross)
  5. `FactPredictions[prob_fatal]`
  6. `FactPredictions[prob_serious]`
  7. `FactPredictions[prob_slight]`
  8. `FactPredictions[speed_limit]`
  9. `FactPredictions[weather_group]`
  10. `FactPredictions[road_type_name]`

#### 7. Visual 5 — Academic Model Interpretation Card ($X=900, Y=445, W=360, H=255$):
Add a styled **Text Box** or **Card**:
```text
ACADEMIC EVALUATION & TRADE-OFF CRITIQUE:

1. The Class Imbalance Challenge:
Fatal accidents represent only 1.43% of collisions. A naïve model predicting "Slight" achieves ~73.8% accuracy but detects 0% of fatal crashes.

2. Cost-Sensitive Class Weighting:
Applying inverse-frequency weighting in PySpark penalizes fatal errors 23.7x more heavily. This raises Fatal Detection Rate to 61.11% (187 of 306 fatal crashes caught).

3. Model Trade-off:
- Logistic Regression: Highest Fatal Safety Recall (61.11%).
- Random Forest: Highest Accuracy (54.50%) & Macro F1 (0.3667).
In road safety planning, false alarms are vastly preferable to undetected fatalities.
```

---

## PART 6: Academic Viva-Voce Defense Talking Points

When presenting your dashboard to college professors and university evaluators, emphasize these 4 points:

1. **End-to-End Enterprise Big Data Architecture**:
   *"We ingested over 100,000 official UK STATS19 collision records, cleaned them using Pandas, engineered 14 domain-specific risk features, stored them in columnar Parquet, and built a distributed machine learning pipeline using Apache Spark (PySpark MLlib)."*

2. **Strict Prevention of Data Leakage**:
   *"Unlike naive projects that cheat by using `number_of_casualties` to predict severity, our model strictly excludes post-accident outcomes. We only use pre-accident indicators like speed limit, weather, lighting, road type, and diurnal time."*

3. **Handling Severe Class Imbalance (73.8% Slight vs 1.4% Fatal)**:
   *"Standard accuracy is a deceptive metric. We formulated an inverse-class-weighting strategy in Spark MLlib, penalizing fatal misclassifications 23.7x higher. This pushed Fatal Accident Recall to 61.11%, making the model operationally valuable for emergency logistics."*

4. **Business Intelligence Integration**:
   *"Our Power BI dashboard connects directly to the Parquet and PySpark outputs using a Star Schema, providing executive KPIs, geospatial clustering, environmental risk matrices, and interactive model diagnostics."*

