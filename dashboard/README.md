# Excel Analytical Dashboard: Road Accident Severity Prediction

## 1. Executive Overview

This directory documents the **Phase 3 Visualization and Business Intelligence Deliverable** for the academic capstone project: **Road Accident Severity Prediction Using Apache Spark and Machine Learning with Excel Dashboard**.

The primary visualization technology is **Microsoft Excel**, replacing the originally planned Power BI implementation. The workbook incorporates executive KPI cards, temporal distributions, infrastructure and environmental hazard analyses, geographic breakdowns, machine learning model comparisons, confusion matrices, and itemized prediction diagnostics.

- **Authoritative Workbook File**: `dataset/excel/Road_Accident_Severity_Professional_Dashboard.xlsx`
- **File Size**: ~17.5 MB
- **Total Worksheets**: 8 (6 user-facing dashboards, 2 calculation/sample support sheets)
- **Total Embedded Charts**: 41 native Excel visualization charts
- **Total Processed Records Embedded**: 100,000 STATS19 collision records

---

## 2. Workbook Location and Access Instructions

### File Path
```
Road_Accident_Severity_Prediction/
└── dataset/
    └── excel/
        └── Road_Accident_Severity_Professional_Dashboard.xlsx
```

### How to Open
1. **Direct Launch**: Double-click the file in File Explorer or open Microsoft Excel and browse to `dataset/excel/Road_Accident_Severity_Professional_Dashboard.xlsx`.
2. **Batch Shortcut**: Execute `open_dashboard.bat` from the project root directory.
3. **Compatible Software**:
   - Microsoft Excel 2016, 2019, 2021, or Microsoft 365 (recommended for best chart rendering).
   - Excel for the Web / Excel Online.
   - LibreOffice Calc (viewable; complex chart formatting is optimized for native Microsoft Excel).

---

## 3. Worksheet Structure & Architecture

The workbook contains eight structured worksheets connected via top-level tab navigation links:

| Sheet # | Worksheet Name | Type | Purpose & Scope |
|---|---|---|---|
| 1 | **Executive Dashboard** | Presentation | High-level collision KPIs, severity distributions, temporal trends, and 7 dropdown filter criteria. |
| 2 | **Road & Environment** | Presentation | Collision risk analysis across road types, speed limits, weather, road surface, and lighting conditions. |
| 3 | **Geographical Analysis** | Presentation | Collision frequency and severity rates across UK police forces, urban vs. rural areas, and coordinate density. |
| 4 | **ML Model Performance** | Presentation | Benchmarking of Logistic Regression, Decision Tree, and Random Forest; confusion matrices; per-class metrics. |
| 5 | **Prediction Analysis** | Presentation | Test partition prediction audit (20,099 rows); actual vs. predicted severity; predicted probability simplexes. |
| 6 | **Accident Data** | Data Repository | Full 100,000 cleaned and feature-engineered collision records with 32 attributes. |
| 7 | **Calc Engine** | Calculation Support | Formula-driven aggregation matrix and lookup tables that feed every chart and KPI card across the workbook. |
| 8 | **Map Sample** | Visualization Support | Stratified coordinate sample (7,431 records) providing coordinate pairs for the latitude/longitude scatter chart. |

---

## 4. Key Performance Indicator (KPI) Definitions

The dashboard calculates and displays standard traffic safety and machine learning evaluation metrics:

### Traffic Safety KPIs (Based on 100,000 Stratified Collisions)
- **Total Collisions**: Exactly 100,000 validated UK collision records.
- **Fatal Collisions**: 1,431 collisions (1.43% of total).
- **Serious Collisions**: 24,812 collisions (24.81% of total).
- **Slight Collisions**: 73,757 collisions (73.76% of total).
- **Total Casualties**: 125,983 persons injured or killed.
- **Total Vehicles Involved**: 181,245 vehicles.
- **Average Casualties per Collision**: 1.26 persons.
- **Average Vehicles per Collision**: 1.81 vehicles.
- **Weekday Collisions**: 75,524 (75.52%).
- **Weekend Collisions**: 24,476 (24.48%).
- **Urban Collisions**: 66,127 (66.13%).
- **Rural Collisions**: 33,871 (33.87%).

### Machine Learning KPIs (Based on 20,099 Held-Out Test Records)
- **Total Test Predictions**: 20,099 disjoint test records.
- **Correct Predictions**: 9,532 records.
- **Incorrect Predictions**: 10,567 records.
- **Overall Model Accuracy**: 47.43% (Class-Weighted Logistic Regression).
- **Actual Fatal Collisions in Test**: 306 collisions.
- **Predicted Fatal Collisions**: 5,097 collisions.
- **Fatal True Positives (TP)**: 187 collisions.
- **Fatal False Positives (FP)**: 4,910 collisions.
- **Fatal False Negatives (FN)**: 119 collisions.
- **Fatal True Negatives (TN)**: 14,883 collisions.
- **Fatal Detection Rate (Recall)**: 61.11% (187 / 306).
- **Fatal Precision**: 3.67% (187 / 5,097).
- **Fatal F1-Score**: 0.0692.
- **Serious Class Recall**: 33.53%.
- **Slight Class Recall**: 51.65%.
- **Average Predicted Probabilities**: Fatal: 27.03%, Serious: 34.52%, Slight: 38.45%.

---

## 5. Chart Catalog

Across the presentation worksheets, the workbook provides 41 charts:
1. **Severity Distribution Pie / Doughnut**: Visualizing the 73.8% Slight, 24.8% Serious, and 1.4% Fatal proportion.
2. **Monthly Collision Trend Line**: 12-month timeline showing collision seasonality across 2025.
3. **Day-of-Week Column Chart**: Friday peak vs. weekend distributions.
4. **24-Hour Time-of-Day Bar Chart**: Morning rush (08:00) and evening rush (17:00) peaks.
5. **Urban vs. Rural Severity Clustered Column Chart**: Comparing crash frequencies and fatal ratios by geographic classification.
6. **Road Type Risk Breakdown**: Single carriageway vs. dual carriageway and roundabout distributions.
7. **Weather & Surface Interaction Chart**: Collision counts under rain, wet roads, frost, and high winds.
8. **Lighting Conditions vs. Severity**: Night darkness with/without street lighting vs. daylight.
9. **Top Police Forces Horizontal Bar Chart**: Metropolitan Police, Greater Manchester, West Midlands, etc.
10. **Geographical Latitude/Longitude Scatter Plot**: Visual representation of collision distribution across Great Britain.
11. **Model Accuracy & F1 Comparison Charts**: Grouped bar charts comparing Logistic Regression, Decision Tree, and Random Forest.
12. **Confusion Matrix Heatmaps**: Cell-shaded matrices illustrating prediction distributions across classes.
13. **Actual vs. Predicted Severity Mosaic / Stacked Bar**: Showing test partition alignment.
14. **Prediction Probability Distribution Histograms**: Displaying soft classification confidence scores.

---

## 6. Source CSV Datasets

The workbook was built upon and verified against three pipeline-exported CSV artifacts located in `dataset/processed/`:

1. `excel_dashboard_dataset.csv` (18.1 MB, 100,000 rows, 32 columns):
   Contains the clean 100k collision records with all engineered categorical labels, temporal fields, and environmental groupings.
2. `excel_dashboard_predictions.csv` (3.6 MB, 20,099 rows, 26 columns):
   Contains held-out test predictions, actual severity, predicted severity, prediction correctness flags, and individual class probabilities (`prob_fatal`, `prob_serious`, `prob_slight`).
3. `model_performance_metrics.csv` (633 bytes, 3 rows, 15 columns):
   Contains comparative evaluation metrics (accuracy, weighted precision/recall/F1, macro F1, fatal recall, precision, F1, training time) for all three evaluated models.

---

## 7. Supporting Calculation Sheets

### Calc Engine (`Sheet 7`)
- Acts as the central calculation backend for the entire workbook.
- Reads 7 filter criteria from the `Executive Dashboard` dropdown controls (Month, Weather, Road Type, Road Surface, Lighting, Area, Severity).
- Maintains unique lookup tables for all categorical dimensions.
- Evaluates native Excel array formulas, `COUNTIFS`, `SUMIFS`, and dynamic matrix aggregations that dynamically update summary tables.
- Feeds all KPI metric cards and chart data series across the presentation sheets.
- **Rule**: Designed as a protected calculation sheet; manual editing is not recommended.

### Map Sample (`Sheet 8`)
- Contains a stratified geographic sample of 7,431 collision coordinates:
  - 100% of all Fatal incidents (1,431 records).
  - 2,500 randomly sampled Serious incidents (seed 42).
  - 3,500 randomly sampled Slight incidents (seed 42).
- Column J applies the active filter criteria from the Executive Dashboard. Filtered-out records return `#N/A`, which native Excel scatter plots ignore, providing dynamic map filtering.

---

## 8. Dashboard Limitations & Technical Constraints

1. **Analytical Reporting Tool**: The dashboard is a diagnostic and decision-support analytical workbook. It is not an active real-time traffic signal or live collision dispatch system.
2. **Formula-Driven Architecture**: The workbook utilizes native formula-driven tables rather than an external OLAP pivot cache or Power BI Tabular model.
3. **Scatter Plot Sampling**: Because Excel scatter plots degrade in rendering performance with 100,000 overlapping markers, geographic plotting utilizes a representative 7,431-point stratified sample that preserves all fatal collisions.
4. **Imbalance Reality**: Machine learning predictions reflect a class-weighted safety-oriented model where fatal recall (61.11%) was prioritized over precision (3.67%). Predicted fatal cases represent high-risk scenarios rather than deterministic outcomes.

---

## 9. Regenerating Source Data

To re-execute the upstream data engineering and machine learning pipelines:

```powershell
# Step 1: Execute Phase 1 Preprocessing and Feature Engineering
python main.py

# Step 2: Execute Phase 2 Apache Spark Machine Learning Pipeline
python spark_main.py

# Step 3: Verify Exported Data Benchmarks
python src/dashboard_export_validation.py

# Step 4: Run Test Suite
python tests/test_pipeline.py
```

