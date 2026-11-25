# Power BI Data Model & Star Schema Architecture

**Project**: Road Accident Severity Prediction Using Apache Spark (PySpark) and Power BI  
**Target Platform**: Microsoft Power BI Desktop  
**Design Standard**: Enterprise Star Schema with Conformed Dimensions & Zero Circular References  

---

## 1. Conceptual Architecture & ERD Diagram

The data model uses a clean **Star Schema** architecture featuring two fact tables (`FactCollisions` for macro accident analysis and `FactPredictions` for PySpark machine learning diagnostics), a standalone benchmark comparison table (`DimModelPerformance`), and shared conformed dimension tables (`DimSeverity`, `DimDate`, `DimWeather`).

```mermaid
erDiagram
    DimSeverity ||--o{ FactCollisions : "1-to-Many (collision_severity)"
    DimSeverity ||--o{ FactPredictions : "1-to-Many (actual_severity)"
    DimDate ||--o{ FactCollisions : "1-to-Many (date)"
    DimDate ||--o{ FactPredictions : "1-to-Many (date)"
    DimWeather ||--o{ FactCollisions : "1-to-Many (weather_group)"
    DimWeather ||--o{ FactPredictions : "1-to-Many (weather_group)"
    FactCollisions ||--o| FactPredictions : "1-to-Zero/One (collision_index)"
    
    FactCollisions {
        string collision_index PK
        string date FK
        int collision_severity FK
        string severity_name
        string time
        int hour
        string time_of_day
        string day_name
        string month_name
        string weather_group FK
        string road_surface_group
        string light_group
        int speed_limit
        string speed_zone
        string road_type_name
        string urban_rural_name
        float latitude
        float longitude
        int number_of_vehicles
        int number_of_casualties
    }

    FactPredictions {
        string collision_index PK_FK
        string date FK
        int actual_severity FK
        string actual_severity_name
        int predicted_severity
        string predicted_severity_name
        int prediction_correct
        float prob_fatal
        float prob_serious
        float prob_slight
        int speed_limit
        string road_type_name
        string urban_rural_name
        string weather_group FK
        float latitude
        float longitude
        int number_of_casualties
    }

    DimSeverity {
        int SeverityCode PK
        string SeverityName
        string SeverityColorHex
        int SortOrder
    }

    DimDate {
        string Date PK
        int Year
        int Month
        string MonthName
        string DayName
        int WeekendIndicator
        string WeekdayWeekend
    }

    DimWeather {
        string WeatherGroup PK
        string WeatherRiskLevel
        string WeatherColorHex
    }

    DimModelPerformance {
        string model_name PK
        string model_type
        float accuracy
        float weighted_f1
        float macro_f1
        float fatal_recall
        float train_time_sec
    }
```

---

## 2. Table Schemas and Ingest Specifications

### Table 1: `FactCollisions` (Primary Fact Table)
- **Source File**: `dataset/processed/powerbi_dataset.csv`
- **Grain**: One row per recorded road traffic collision in the UK.
- **Row Count**: **100,000 rows** across **32 columns**.
- **Role**: Powers macroscopic accident analytics on Pages 1, 2, and 3.
- **Key Columns & Types**:
  - `collision_index`: Text (Primary Key for record drilldown).
  - `date`: Text / Date (Format: `DD/MM/YYYY`).
  - `time`: Text (Format: `HH:MM`).
  - `collision_severity`: Whole Number (`1` = Fatal, `2` = Serious, `3` = Slight).
  - `severity_name`: Text (`Fatal`, `Serious`, `Slight`).
  - `number_of_vehicles`: Whole Number (Vehicles involved).
  - `number_of_casualties`: Whole Number (Human casualties — Key KPI).
  - `latitude`: Decimal Number (WGS84 coordinate).
  - `longitude`: Decimal Number (WGS84 coordinate).
  - `hour`: Whole Number (0–23).
  - `time_of_day`: Text (`Morning`, `Afternoon`, `Evening`, `Night`).
  - `day_name`: Text (`Monday` to `Sunday`).
  - `month_name`: Text (`January` to `December`).
  - `weekend_indicator`: Whole Number (`1` = Weekend, `0` = Weekday).
  - `weekday_weekend`: Text (`Weekday`, `Weekend`).
  - `speed_limit`: Whole Number (`20`, `30`, `40`, `50`, `60`, `70`).
  - `speed_zone`: Text (`<=20 mph`, `30 mph`, `40-50 mph`, `60-70 mph`).
  - `road_type_name`: Text (`Single carriageway`, `Dual carriageway`, `Roundabout`, etc.).
  - `urban_rural_name`: Text (`Urban`, `Rural`).
  - `weather_group`: Text (`Fine`, `Raining`, `Snow/Ice/Fog`, `Other/Unknown`).
  - `road_surface_group`: Text (`Dry`, `Wet/Flood`, `Snow/Ice`, `Other/Unknown`).
  - `light_group`: Text (`Daylight`, `Darkness - Lights Lit`, `Darkness - Unlit/No Lights`, `Other/Unknown`).

---

### Table 2: `FactPredictions` (ML Diagnostics Fact Table)
- **Source File**: `dataset/processed/powerbi_predictions.csv`
- **Grain**: One row per model-evaluated test set collision.
- **Row Count**: **20,099 rows** across **26 columns** (Unseen test set).
- **Role**: Powers Machine Learning Diagnostic Center on Page 4.
- **Key Columns & Types**:
  - `collision_index`: Text (Foreign Key matching `FactCollisions`).
  - `actual_severity`: Whole Number (`1`, `2`, `3`).
  - `actual_severity_name`: Text (`Fatal`, `Serious`, `Slight`).
  - `predicted_severity`: Whole Number (`1`, `2`, `3`).
  - `predicted_severity_name`: Text (`Fatal`, `Serious`, `Slight`).
  - `prediction_correct`: Whole Number (`1` = Match, `0` = Misclassification).
  - `prob_fatal`: Decimal Number (Predicted probability of Fatal class, range: `0.0` - `1.0`).
  - `prob_serious`: Decimal Number (Predicted probability of Serious class).
  - `prob_slight`: Decimal Number (Predicted probability of Slight class).
  - `number_of_casualties`: Whole Number (Retained solely for KPI cards and filter slicers).
  - `latitude`, `longitude`, `speed_limit`, `weather_group`, `road_surface_group`, `light_group`.

---

### Table 3: `DimModelPerformance` (Benchmark Dimension Table)
- **Source File**: `dataset/processed/powerbi_model_performance.csv`
- **Grain**: One row per trained PySpark MLlib classification algorithm.
- **Row Count**: **3 rows** across **15 columns**.
- **Role**: Powers the Model Comparison visual on Page 4.
- **Columns & Types**:
  - `model_name`: Text (`Logistic Regression`, `Decision Tree`, `Random Forest`).
  - `model_type`: Text (Model description and tuning parameters).
  - `accuracy`: Decimal Number (Percentage: `47.43`, `46.02`, `54.50`).
  - `weighted_f1`: Decimal Number (`0.5464`, `0.5346`, `0.5985`).
  - `macro_f1`: Decimal Number (`0.3383`, `0.3257`, `0.3667`).
  - `fatal_recall`: Decimal Number (`61.11`, `59.48`, `54.90`).
  - `serious_recall`: Decimal Number (`33.53`, `30.00`, `31.72`).
  - `slight_recall`: Decimal Number (`51.65`, `50.93`, `61.87`).
  - `train_time_sec`: Decimal Number (`9.54`, `10.04`, `57.59`).

---

### Table 4: `DimSeverity` (DAX Calculated Dimension)
Can be created directly in Power BI Desktop using the following DAX formula:

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

---

### Table 5: `DimWeather` (DAX Calculated Dimension)
```dax
DimWeather = 
DATATABLE(
    "WeatherGroup", STRING,
    "WeatherRiskLevel", STRING,
    "WeatherColorHex", STRING,
    {
        {"Fine", "Standard Exposure", "#38BDF8"},
        {"Raining", "Moderate Hazard", "#0284C7"},
        {"Snow/Ice/Fog", "High Hazard", "#F59E0B"},
        {"Other/Unknown", "Variable", "#64748B"}
    }
)
```

---

## 3. Relationships & Cardinality Configuration

In Power BI Desktop **Model View**, set up relationships as follows:

| From (Dimension) | To (Fact) | Cardinality | Cross Filter Direction | Active | Notes |
|---|---|:---:|:---:|:---:|---|
| `DimSeverity[SeverityCode]` | `FactCollisions[collision_severity]` | 1-to-Many (`1:*`) | Single | **Yes** | Filters all accident visuals |
| `DimSeverity[SeverityCode]` | `FactPredictions[actual_severity]` | 1-to-Many (`1:*`) | Single | **Yes** | Filters test predictions by actual severity |
| `DimSeverity[SeverityCode]` | `FactPredictions[predicted_severity]` | 1-to-Many (`1:*`) | Single | *No* | Inactive (use `USERELATIONSHIP` if needed) |
| `DimWeather[WeatherGroup]` | `FactCollisions[weather_group]` | 1-to-Many (`1:*`) | Single | **Yes** | Synchronizes weather slicers |
| `DimWeather[WeatherGroup]` | `FactPredictions[weather_group]` | 1-to-Many (`1:*`) | Single | **Yes** | Synchronizes weather slicers |
| `FactCollisions[collision_index]` | `FactPredictions[collision_index]` | 1-to-One / Many | Single (`FactCollisions` $\rightarrow$ `FactPredictions`) | **Yes** | Collision ID used solely for drilldown |

> **Best Practice Rule**: Never use `collision_index` for analytical summaries or aggregations. Use the conformed dimensions (`DimSeverity`, `DimWeather`) to cross-filter both fact tables synchronously without ambiguity.

