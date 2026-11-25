# End-to-End Project Execution Workflow
## Road Accident Severity Prediction Using Apache Spark and Machine Learning with Excel Dashboard

---

## 1. Workflow Diagram

```
+-------------------------------------------------------------------------------+
|                      STEP 1: PREREQUISITES & ENVIRONMENT SETUP                |
|  * Python 3.11.9 (64-bit) & Virtual Environment (venv)                        |
|  * Oracle Java 8 (jdk1.8.0_481 64-bit)                                        |
|  * Hadoop 3.2.4 Winutils (winutils.exe in C:\hadoop-3.2.4\bin)                |
|  * Apache Spark 3.5.9 (C:\Spark)                                              |
|  * Microsoft Excel 2016+ / Microsoft 365                                      |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                   STEP 2: DATA PREPROCESSING & HYGIENE                        |
|  Command: python main.py                                                      |
|  * Ingests: dataset/raw/dft-road-casualty-statistics-collision-2025.csv      |
|  * Eliminates 18 target leakage features & 2 invalid coordinate rows          |
|  * Performs 14 feature engineering transformations                            |
|  * Exports: accident_cleaned.csv, accident_final_100k.csv,                    |
|             accident_final_100k.parquet, excel_dashboard_dataset.csv          |
|  * Generates: reports/preprocessing_report.txt, reports/data_quality_report.txt|
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|             STEP 3: APACHE SPARK PROCESSING & MLlib TRAINING                  |
|  Command: python spark_main.py                                                |
|  * Ingests: dataset/processed/accident_final_100k.parquet                     |
|  * Fits 23-stage StringIndexer + OneHotEncoder + VectorAssembler pipeline    |
|  * Computes inverse class weights for Fatal / Serious / Slight                |
|  * Trains: Logistic Regression, Decision Tree, Random Forest                  |
|  * Evaluates: 20,099 held-out test partition                                  |
|  * Exports: excel_dashboard_predictions.csv, model_performance_metrics.csv    |
|  * Serializes: models/spark_feature_pipeline,                                 |
|                models/spark_logistic_regression_model, evaluation_summary.json|
|  * Generates: reports/spark_processing_report.txt,                            |
|               reports/model_evaluation_report.txt                             |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|             STEP 4: BENCHMARK VALIDATION & VERIFICATION TEST SUITE            |
|  Commands:                                                                    |
|    python src/dashboard_export_validation.py                                  |
|    python tests/test_pipeline.py                                              |
|  * Validates row counts, schemas, coordinate bounding boxes, probability sum  |
|  * Verifies all test assertions pass (8/8 unit tests)                         |
|  * Generates: reports/dashboard_benchmark_report.txt                          |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                 STEP 5: EXCEL DASHBOARD VISUALIZATION & AUDIT                 |
|  Action: Execute open_dashboard.bat OR open:                                  |
|          dataset/excel/Road_Accident_Severity_Professional_Dashboard.xlsx     |
|  * Explore 8 interactive worksheets across 41 embedded native Excel charts    |
|  * Test 7 dropdown filters on the Executive Dashboard                         |
|  * Review ML model comparisons and itemized test predictions                  |
+-------------------------------------------------------------------------------+
```

---

## 2. Step-by-Step Execution Guide

All commands should be executed from the project root directory:
`D:\ApacheSpark Project\Road_Accident_Severity_Prediction`

### Step 2.1: Verify Python Environment & Dependencies
Activate the virtual environment:
```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```
Verify that required Python packages are installed:
```powershell
pip install -r requirements.txt
```

### Step 2.2: Verify Environmental Variables (Java, Hadoop, Spark)
Verify that Java and Hadoop are reachable:
```powershell
# Check Java Version (Must be 64-bit Java 8)
java -version

# Check Hadoop Winutils
C:\hadoop-3.2.4\bin\winutils.exe ls \
```
*Note*: `src/spark_session.py` and `config/spark_config.py` automatically inject default paths (`C:\Java\jdk1.8.0_481`, `C:\hadoop-3.2.4`, `C:\Spark`) if system environment variables are not globally configured.

### Step 2.3: Execute Phase 1 Preprocessing Pipeline
Run the data cleaning, leakage removal, feature engineering, and stratified sampling script:
```powershell
python main.py
```
**Expected Console Output**:
- Original Records: `101,525` rows across 44 columns
- Cleaned Records: `101,523` rows across 25 columns
- Final Selected Records: `100,000` rows across 39 columns
- Target Distribution: Fatal = 1,431 (1.43%), Serious = 24,812 (24.81%), Slight = 73,757 (73.76%)
- Exported Files: `accident_cleaned.csv`, `accident_final_100k.csv`, `accident_final_100k.parquet`, `excel_dashboard_dataset.csv`
- Reports Generated: `preprocessing_report.txt`, `data_quality_report.txt`

### Step 2.4: Execute Phase 2 Apache Spark ML Pipeline
Run the Spark MLlib feature engineering, class weighting, model training, and prediction export:
```powershell
python spark_main.py
```
**Expected Console Output**:
- Apache Spark Session initialized in local mode (`local[*]`).
- Parquet ingested: `100,000` rows, 39 columns.
- Training Partition (80%): `79,901` rows.
- Testing Partition (20%): `20,099` rows.
- Models trained & evaluated:
  - Logistic Regression: Accuracy = 47.43%, Fatal Recall = 61.11%
  - Decision Tree: Accuracy = 46.02%, Fatal Recall = 59.48%
  - Random Forest: Accuracy = 54.50%, Fatal Recall = 54.90%
- Selected Best Safety Model: `LOGISTIC REGRESSION`
- Exported Files: `excel_dashboard_predictions.csv`, `model_performance_metrics.csv`
- Serialized Artifacts: `models/spark_feature_pipeline`, `models/spark_logistic_regression_model`, `models/evaluation_summary.json`
- Reports Generated: `spark_processing_report.txt`, `model_evaluation_report.txt`

### Step 2.5: Validate Pipeline Outputs & Run Automated Test Suite
Run the dashboard benchmark validator:
```powershell
python src/dashboard_export_validation.py
```
Run the automated unit test suite:
```powershell
python tests/test_pipeline.py
# Or double-click run_tests.bat
```
**Expected Test Results**:
```
Ran 8 tests in ~3.0s
OK
```

### Step 2.6: Open and Explore the Excel Dashboard
Launch the final analytical deliverable:
```powershell
.\open_dashboard.bat
```
Alternatively, open Microsoft Excel directly and navigate to:
`dataset\excel\Road_Accident_Severity_Professional_Dashboard.xlsx`

Review the eight worksheets:
1. `Executive Dashboard`: View high-level KPIs, severity distribution, monthly trend, and test the 7 filter dropdowns.
2. `Road & Environment`: Review collision risk by road geometry, speed limit, weather, and lighting.
3. `Geographical Analysis`: Review top police forces, urban vs. rural fatality disparity, and Great Britain coordinate scatter.
4. `ML Model Performance`: Review model accuracy, weighted/macro F1, safety recall, and confusion matrices.
5. `Prediction Analysis`: Inspect 20,099 individual predictions and calibrated class probability scores.
6. `Accident Data`: Review the complete 100,000-record repository.
7. `Calc Engine`: Inspect formula-driven aggregation tables.
8. `Map Sample`: Inspect the 7,431-point stratified coordinate sample and dynamic filter logic.

---

## 3. Common Troubleshooting Scenarios

| Issue / Symptom | Root Cause | Solution |
|---|---|---|
| `java.lang.UnsatisfiedLinkError` or `HADOOP_HOME not set` on Windows | Missing `winutils.exe` in `C:\hadoop-3.2.4\bin` | Ensure `winutils.exe` exists in `C:\hadoop-3.2.4\bin`. Both `config/spark_config.py` and `src/spark_session.py` automatically inject this path into system PATH. |
| `PySpark Java Gateway exit code 1` | Incompatible Java version | Apache Spark 3.5.9 requires 64-bit Java 8 or Java 11. Verify `JAVA_HOME=C:\Java\jdk1.8.0_481`. |
| `File used by another process` when saving `.xlsx` | Excel workbook is open in Microsoft Excel | Close the workbook in Microsoft Excel before running Python scripts that write to it, or use read-only sharing streams. |
| `ModuleNotFoundError: No module named 'pyspark'` | PySpark not on system Python path | Use the project virtual environment `venv/Scripts/python.exe`, which has access to `C:\Spark\python`. |
| Scatter plot displays empty points | Filter dropdowns set to an impossible combination | Reset Executive Dashboard dropdowns to `"All"` to clear active filters. |

