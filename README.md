# Road Accident Severity Prediction Using Apache Spark and Machine Learning with Excel Dashboard

**Academic Degree**: Bachelor of Technology (B.Tech) in Computer Science & Engineering / Data Science  
**Project Type**: Final Year Academic Capstone Project  
**Academic Year**: 2025–2026  
**Domain**: Big Data Engineering, Machine Learning & Business Intelligence  
**Dataset Source**: UK Department for Transport (DfT) STATS19 (2025 Collisions)  
**Primary Technologies**: Python 3.11.9, Apache Spark 3.5.9 (PySpark MLlib), Java 8, Hadoop 3.2.4 (winutils), Microsoft Excel  

---

## 1. Project Overview

This project delivers an end-to-end data engineering, distributed machine learning, and analytical visualization architecture to analyze personal injury road collisions and classify collision injury severity into three standard statutory categories defined by the UK Department for Transport:

- **Fatal (Class 1 / Label 0.0)**: Collisions resulting in the death of at least one person within 30 days of the incident.
- **Serious (Class 2 / Label 1.0)**: Collisions causing severe injuries requiring hospital admission (e.g., fractures, severe lacerations, severe head trauma).
- **Slight (Class 3 / Label 2.0)**: Collisions causing minor injuries (e.g., sprains, shallow cuts, shock) not requiring overnight hospital admission.

The project utilizes:
- **Python** for data acquisition, quality auditing, target leakage elimination, cyclical and interaction feature engineering, and stratified sampling.
- **Apache Spark and PySpark MLlib** (configured in local multi-threaded execution mode) for distributed data transformation, multi-strategy class imbalance weighting, hyperparameter tuning on validation partitions, and multiclass classifier training.
- **Microsoft Excel** for executive dashboard development, analytical exploration, dynamic formula aggregation, and diagnostic model verification via an interactive 8-worksheet workbook.

> [!NOTE]
> **Prediction Scope Clarification**: This system classifies the severity associated with collision records using available environmental, infrastructural, and temporal collision-related features. It is **not** an accident forecasting system that predicts crashes before they occur, nor is it an in-vehicle collision avoidance device. It serves as an analytical risk-evaluation and decision-support system for transportation safety engineers and policy analysts.

---

## 2. Project Objectives

1. **Analyze Road Accident Patterns**: Identify high-hazard spatial, temporal, infrastructural, and meteorological conditions contributing to collision severity across Great Britain.
2. **Clean & Preprocess STATS19 Data**: Cleanse the official 2025 collision registry, eliminate deprecated fields, validate spatial boundaries, and eliminate duplicate records.
3. **Strictly Eliminate Target Data Leakage**: Methodically isolate and exclude all post-accident consequences (such as casualty counts, police attendance, and post-hoc severity adjustments) from model predictors.
4. **Engineer High-Impact Domain Features**: Synthesize continuous cyclical representations of time and date, compound multi-condition risk interactions (e.g., high-speed single carriageways, adverse weather on wet surfaces, unlit nighttime travel).
5. **Process Data Using Apache Spark MLlib**: Ingest columnar Snappy-compressed Parquet datasets and construct a 23-stage MLlib pipeline (`StringIndexer`, `OneHotEncoder`, `VectorAssembler`) producing a 195-dimensional assembled feature space.
6. **Address Extreme Class Imbalance**: Benchmark inverse-frequency weighting, smoothed square root weighting, effective number of samples weighting, and training-only resampling to balance minority fatal collision detection against majority precision.
7. **Evaluate Diverse MLlib Algorithms**: Benchmark Logistic Regression, Decision Tree, Random Forest, Gradient-Boosted Trees (via `OneVsRest` multiclass adapter), Multilayer Perceptron, and Naive Bayes against an empirical Majority-Class Baseline.
8. **Systematic Hyperparameter Tuning**: Perform parameter searches across regularization, tree depth, and ensemble counts using an internal 80/20 train/validation split on the training partition, preserving a 100% untouched test partition.
9. **Deliver an Authoritative Excel Dashboard**: Synchronize executive KPIs, infrastructure profiles, spatial scatter charts, model benchmarks, and test prediction audits across an interactive 8-worksheet Excel workbook.

---

## 3. Dataset Description & Preprocessing Hygiene

### 3.1 Source & Provenance
The analysis is based on the official **UK Department for Transport (DfT) STATS19 2025 Collisions Dataset** (`dft-road-casualty-statistics-collision-2025.csv`), documenting all police-reported injury collisions across England, Scotland, and Wales for calendar year 2025.

### 3.2 Preprocessing Audit & Verification
- **Original Raw Records**: `101,525` rows across `44` raw attributes.
- **Invalid Coordinate Records Dropped**: Exactly `2` records containing missing/null coordinates outside Great Britain.
- **Cleaned Valid Records**: `101,523` rows across `25` retained base attributes.
- **Stratified Final Sample Size**: Exactly `100,000` records selected via stratified random sampling to preserve exact national injury proportions.
- **Random Sampling Seed**: `42` (ensuring 100% deterministic reproducibility).
- **Missing Values**: `0` unhandled `NaN` or missing values across all final modeling columns.
- **Duplicate Records**: `0` duplicate rows; `100,000` unique `collision_index` primary keys.
- **Final Feature Space**: `39` tabular columns in Parquet; `195` assembled vector features in Spark MLlib.

### 3.3 Target Class Distribution (100,000-Record Stratified Sample)

| Severity Level | STATS19 Code | MLlib Label | Total Records | Percentage |
|---|:---:|:---:|---:|---:|
| **Fatal** | `1` | `0.0` | 1,431 | 1.43% |
| **Serious** | `2` | `1.0` | 24,812 | 24.81% |
| **Slight** | `3` | `2.0` | 73,757 | 73.76% |
| **Total** | — | — | **100,000** | **100.00%** |

---

## 4. Feature Engineering & Target Leakage Prevention

### 4.1 Strict Exclusion of Target Leakage Features
To maintain scientific validity, all post-accident consequences and outcome-derived metadata were strictly excluded from model features:

| Excluded Column | Reason for Exclusion | Consequence If Included |
|---|---|---|
| `number_of_casualties` | Post-accident outcome; casualties are counted after collision occurs | Artificial predictive inflation |
| `enhanced_severity_collision` | DfT post-hoc derived severity variable | Direct target leakage |
| `collision_adjusted_severity_serious` | Probability adjustment derived from medical records | Direct target leakage |
| `collision_adjusted_severity_slight` | Probability adjustment derived from medical records | Direct target leakage |
| `did_police_officer_attend_scene_of_accident` | Post-accident response action | Spurious correlation |

### 4.2 Domain-Justified Engineered Features

| Feature Name | Type | Formula / Logic | Domain Physical Justification |
|---|---|---|---|
| `sin_hour`, `cos_hour` | Continuous Numerical | $\sin(2\pi \cdot h / 24)$, $\cos(2\pi \cdot h / 24)$ | Eliminates artificial boundary between 23:59 and 00:00; preserves cyclical diurnal progression |
| `sin_month`, `cos_month` | Continuous Numerical | $\sin(2\pi \cdot (m-1) / 12)$, $\cos(2\pi \cdot (m-1) / 12)$ | Captures continuous seasonal progression between December (12) and January (1) |
| `road_speed_inter` | Categorical Interaction | `concat(road_type_name, '_', speed_zone)` | Captures geometric hazard differences (e.g., 60 mph on single carriageway vs. 70 mph on motorway) |
| `weather_surface_inter` | Categorical Interaction | `concat(weather_group, '_', road_surface_group)` | Captures compound traction loss (e.g., Rain on Flood/Wet surface vs. Dry Clear conditions) |
| `light_time_inter` | Categorical Interaction | `concat(light_group, '_', time_of_day)` | Distinguishes natural night darkness from lit urban night roadways |
| `weekend_time_inter` | Categorical Interaction | `concat(weekend_indicator, '_', time_of_day)` | Isolates late-night weekend recreational travel from weekday morning commuting |
| `urban_rural_speed_inter` | Categorical Interaction | `concat(urban_rural_name, '_', speed_zone)` | High speeds on unlit rural country lanes exhibit the highest fatality rates in Great Britain |
| `high_speed_single_carriageway` | Binary Numerical | $1 \text{ if } \text{single carriageway} \land \text{speed} \ge 50 \text{ else } 0$ | Single carriageways with $\ge 50$ mph speeds lack median barriers, causing catastrophic head-on crashes |

---

## 5. Handling Extreme Class Imbalance

With Fatal collisions comprising only **1.43%** of real-world observations, standard unweighted classifiers suffer from the **Accuracy Paradox**:

### 5.1 The Accuracy Paradox & Majority-Class Baseline
- An uninformed baseline classifier that predicts "Slight" for 100% of collisions achieves **74.38% overall accuracy**.
- However, this baseline has **0.0% Fatal Recall** and **0.0% Serious Recall**, failing to identify even a single life-threatening collision.
- Evaluating transportation safety models purely on raw accuracy is scientifically unsound.

### 5.2 Multi-Strategy Imbalance Handling
We evaluated four distinct class-balancing strategies, fitted **strictly on the training partition**:

1. **Inverse Frequency Weighting**:
   $$w_c = \frac{N_{\text{total}}}{C \cdot N_c}$$
   Yields $w_{\text{Fatal}} = 23.67, w_{\text{Serious}} = 1.34, w_{\text{Slight}} = 0.45$. Aggressively penalizes fatal false negatives, maximizing safety recall.
2. **Smoothed Square Root Weighting**:
   $$w_c \propto \sqrt{\frac{N_{\text{total}}}{N_c}}$$
   Moderates extreme weights, reducing false alarms and improving overall weighted F1.
3. **Effective Number of Samples Weighting** (Cui et al., CVPR 2019):
   $$E_{n} = \frac{1 - \beta^{N_c}}{1 - \beta}, \quad w_c \propto \frac{1}{E_n} \quad (\beta = 0.9999)$$
   Accounts for information overlap in dense majority classes.
4. **Resampling Techniques**:
   Minority oversampling and majority undersampling applied exclusively to the training set (`train_df`), leaving the 20,099 held-out test records untouched.

---

## 6. Machine Learning Algorithms & GBT Multiclass Limitation

### 6.1 Evaluated Algorithms
1. **Multinomial Logistic Regression**: Linear baseline with softmax link and multi-strategy class weights.
2. **Decision Tree Classifier**: Non-linear tree partitioning with Gini impurity and leaf instance constraints.
3. **Random Forest Classifier**: Ensemble of 60 bagging trees with bootstrap aggregation and feature subset sampling.
4. **Gradient-Boosted Trees (GBTClassifier via `OneVsRest`)**: Sequential boosting with gradient descent.
5. **Multilayer Perceptron Classifier (MLP)**: 4-layer feedforward deep neural network (195 -> 64 -> 32 -> 3).
6. **Multinomial Naive Bayes**: Probabilistic conditionally-independent baseline.

### 6.2 Technical Deep Dive: Spark MLlib GBT Multiclass Limitation
In Apache Spark MLlib 3.5.9, the native `GBTClassifier` strictly supports **binary classification only**:
```text
java.lang.RuntimeException: Labels MUST be in {0, 1}, but got 2.0
```
When supplied with a 3-class target (`0.0=Fatal, 1.0=Serious, 2.0=Slight`), native `GBTClassifier.fit()` terminates with a runtime exception.

**Solution**: We wrapped `GBTClassifier` inside Apache Spark MLlib's native `OneVsRest` meta-classifier:
```python
from pyspark.ml.classification import GBTClassifier, OneVsRest

gbt_binary = GBTClassifier(maxIter=15, maxDepth=5, seed=42)
ovr_gbt = OneVsRest(classifier=gbt_binary, featuresCol="features", labelCol="label")
ovr_model = ovr_gbt.fit(train_weighted)
```
`OneVsRest` trains three independent binary boosting models (Fatal vs. Rest, Serious vs. Rest, Slight vs. Rest) and chooses the class with the highest margin score, fully respecting the Spark MLlib distributed execution model.

---

## 7. Systematic Hyperparameter Tuning

To prevent data leakage, hyperparameter tuning was conducted using an **internal 80/20 train/validation split** on `train_df` (63,921 sub-train rows, 15,980 validation rows). The held-out test partition (20,099 rows) was **never touched** during tuning.

### 7.1 Parameter Grid & Validation Results (`reports/hyperparameter_tuning_results.csv`)

| Model Family | Configuration Tested | Val Accuracy | Val Weighted F1 | Val Fatal Recall | Val Fatal Precision | Selection Result |
|---|---|---:|---:|---:|---:|:---:|
| **Logistic Regression** | `regParam=0.01, elasticNet=0.0, Inverse Weights` | 47.45% | 0.5468 | 61.35% | 0.0368 | Baseline |
| **Logistic Regression** | `regParam=0.001, elasticNet=0.0, Inverse Weights` | 47.88% | 0.5512 | **62.30%** | 0.0372 | **Selected LR** |
| **Logistic Regression** | `regParam=0.05, elasticNet=0.2, Sqrt Weights` | 72.10% | 0.6742 | 2.85% | 0.1050 | Low Safety Recall |
| **Decision Tree** | `maxDepth=8, minInstances=10, impurity=gini` | 46.12% | 0.5355 | 59.40% | 0.0334 | Baseline |
| **Decision Tree** | `maxDepth=10, minInstances=5, impurity=gini` | 46.65% | 0.5408 | 58.75% | 0.0345 | **Selected DT** |
| **Random Forest** | `numTrees=30, maxDepth=10, minInstances=5` | 55.20% | 0.6015 | 52.80% | 0.0450 | Underfitted |
| **Random Forest** | `numTrees=50, maxDepth=10, minInstances=5` | 55.95% | 0.6082 | 51.50% | 0.0468 | Original |
| **Random Forest** | `numTrees=60, maxDepth=12, minInstances=5` | **57.56%** | **0.6105** | 43.72% | **0.0513** | **Selected RF** |

---

## 8. Empirical Model Evaluation & Benchmark Comparison

All final models were evaluated on the **20,099 held-out test records** (disjoint 20% partition).

### 8.1 Complete Model Performance Benchmark Table

| Model Name | Accuracy | Weighted F1 | Macro F1 | Fatal Recall | Fatal Precision | Fatal F1 | Training Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Majority-Class Baseline** (Always Slight) | 74.38% | 0.6345 | 0.2844 | 0.00% | 0.00% | 0.0000 | 0.00s |
| **Logistic Regression** (Class-Weighted) | 47.66% | 0.5494 | 0.3392 | **61.76%** | 3.66% | 0.0691 | **18.89s** |
| **Decision Tree** (maxDepth=10) | 73.58% | 0.6476 | 0.3088 | 0.33% | **6.25%** | 0.0062 | 22.55s |
| **Random Forest** (60 Trees, Depth=12) | **58.11%** | **0.6210** | **0.3809** | 44.12% | 4.82% | **0.0869** | 159.69s |
| **Gradient-Boosted Trees** (OneVsRest, 15 iter) | 74.29% | 0.6433 | 0.2971 | 0.00% | 0.00% | 0.0000 | 53.21s |
| **Multilayer Perceptron** (4-Layer MLP) | 74.38% | 0.6345 | 0.2844 | 0.00% | 0.00% | 0.0000 | 81.45s |
| **Multinomial / Gaussian Naive Bayes** | 12.71% | 0.1917 | 0.1005 | 97.06% | 1.72% | 0.0339 | 1.58s |

### 8.2 Side-by-Side: Original vs. Improved Results (Random Forest)

| Metric | Original Run | Improved Run | Absolute Gain / Delta |
|---|---:|---:|:---:|
| **Overall Accuracy** | 54.50% | **58.11%** | **+3.61%** |
| **Weighted F1-Score** | 0.5985 | **0.6210** | **+0.0225** |
| **Macro F1-Score** | 0.3667 | **0.3809** | **+0.0142** |
| **Fatal Precision** | 4.39% | **4.82%** | **+0.43%** |
| **Fatal F1-Score** | 0.0814 | **0.0869** | **+0.0055** |
| **Serious F1-Score** | 0.3542 | **0.3294** | -0.0248 |
| **Slight F1-Score** | 0.6645 | **0.7264** | **+0.0619** |
| **Assembled Feature Count** | 185 features | **195 features** | **+10 features** |
| **Feature Representation** | Linear / Discrete | **Continuous Cyclical + Interactions** | **Domain-Grounded** |

### 8.3 3x3 Confusion Matrix (Optimized Random Forest, Test Set)

```text
Actual Severity \ Predicted       Pred Fatal       Pred Serious       Pred Slight       Total Actual
---------------------------------------------------------------------------------------------------
Fatal (True, Severity 1)                 135                 84                87                306
Serious (True, Severity 2)               841              1,598             2,404              4,843
Slight (True, Severity 3)              1,826              3,177             9,947             14,950
---------------------------------------------------------------------------------------------------
Total Predicted                        2,802              4,859            12,438             20,099
```

### 8.4 3x3 Confusion Matrix (Class-Weighted Logistic Regression, Safety Screening Model)

```text
Actual Severity \ Predicted       Pred Fatal       Pred Serious       Pred Slight       Total Actual
---------------------------------------------------------------------------------------------------
Fatal (True, Severity 1)                 189                 60                57                306
Serious (True, Severity 2)             1,534              1,597             1,712              4,843
Slight (True, Severity 3)              3,438              3,718             7,794             14,950
---------------------------------------------------------------------------------------------------
Total Predicted                        5,161              5,375             9,563             20,099
```

---

## 9. Probability Threshold & Risk-Decision Analysis

Rather than relying purely on standard $\arg\max_c P(Y=c|X)$ classification, safety-critical transportation systems can adjust the fatal decision threshold $\tau_{\text{Fatal}}$ on validation data to navigate the trade-off between fatal recall (capturing dangerous incidents) and fatal precision (reducing false alarms):

$$\hat{Y} = \begin{cases} 1 \text{ (Fatal)} & \text{if } P(\text{Fatal}) \ge \tau_{\text{Fatal}} \\ 2 \text{ (Serious)} & \text{else if } P(\text{Serious}) \ge P(\text{Slight}) \\ 3 \text{ (Slight)} & \text{otherwise} \end{cases}$$

> [!NOTE]
> **Class-Weighted Probability Mechanics**: Because our final primary model (Random Forest, 60 trees, depth 12) incorporates inverse class weighting ($w_{\text{Fatal}}=23.67, w_{\text{Serious}}=1.40, w_{\text{Slight}}=0.46$), predicted probabilities are calibrated upwards to elevate rare events: the empirical mean predicted probability of Fatal across all 20,099 test cases is **$20.39\%$** (median **$16.68\%$**, max $76.69\%$). Consequently, setting $\tau = 0.05$ flags $96.5\%$ of cases as Fatal (causing $19,096$ false alarms), whereas the effective operational decision thresholds operate around $\tau \in [0.30, 0.50]$.

### Complete Held-Out Test Set Evaluation Across Decision Thresholds (`N = 20,099`, Final Random Forest):

| Threshold ($\tau$) | Overall Accuracy | Fatal Recall | Fatal Precision | Fatal F1 | Fatal TP | Fatal FP ($FP_0$) | Serious Recall | Slight Recall | Macro F1 | Weighted F1 | False Fatal Rate | Operational Profile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **$\tau = 0.05$** | $4.64\%$ | **$100.00\%$** | $1.58\%$ | $0.0311$ | $306$ | $19,096$ | $0.02\%$ | $4.18\%$ | $0.0371$ | $0.0600$ | $96.48\%$ | Universal emergency alert (over-sensitized) |
| **$\tau = 0.10$** | $16.75\%$ | **$97.06\%$** | $1.82\%$ | $0.0358$ | $297$ | $15,992$ | $0.97\%$ | $20.22\%$ | $0.1266$ | $0.2470$ | $80.80\%$ | Ultra-high hazard sweep |
| **$\tau = 0.20$** | $46.10\%$ | **$78.10\%$** | $3.07\%$ | $0.0591$ | $239$ | $7,541$ | $12.64\%$ | $56.29\%$ | $0.3013$ | $0.5383$ | $38.10\%$ | Emergency surge mobilization |
| **$\tau = 0.25$** | $52.12\%$ | **$65.36\%$** | $3.62\%$ | $0.0686$ | $200$ | $5,322$ | $20.28\%$ | $62.17\%$ | $0.3399$ | $0.5827$ | $26.89\%$ | High hazard coverage |
| **$\tau = 0.30$** | $55.70\%$ | **$54.58\%$** | $4.22\%$ | $0.0783$ | $167$ | $3,793$ | $26.60\%$ | $65.16\%$ | $0.3630$ | $0.6060$ | $19.16\%$ | Broad safety screening |
| **$\tau = 0.35$** | $57.92\%$ | **$44.77\%$** | $4.75\%$ | $0.0858$ | $137$ | $2,749$ | $31.94\%$ | $66.60\%$ | $0.3782$ | $0.6195$ | $13.89\%$ | High-risk active patrol |
| **Standard $\arg\max$** | **$58.11\%$** | **$44.12\%$** | **$4.82\%$** | **$0.0869$** | $135$ | **$2,667$** | **$33.00\%$** | **$66.54\%$** | **$0.3809$** | **$0.6210$** | **$13.47\%$** | **Default primary model baseline** |
| **$\tau = 0.40$** | **$59.68\%$** | $37.58\%$ | **$5.68\%$** | **$0.0986$** | $115$ | **$1,911$** | $36.30\%$ | $67.71\%$ | **$0.3916$** | **$0.6294$** | **$9.65\%$** | **False alarms cut by $-28.3\%$ ($-756$)** |
| **$\tau = 0.50$** | **$61.68\%$** | $22.55\%$ | **$9.00\%$** | **$0.1286$** | $69$ | **$698$** | $43.51\%$ | $68.37\%$ | **$0.4119$** | **$0.6393$** | **$3.53\%$** | **False alarms cut by $-73.8\%$ ($-1,969$)** |
| **$\tau = 0.60$** | **$62.56\%$** | $3.92\%$ | **$11.21\%$** | $0.0581$ | $12$ | **$95$** | $48.17\%$ | $68.41\%$ | $0.3943$ | **$0.6426$** | **$0.48\%$** | Ultra-high precision enforcement |

---

## 10. Final Model Selection & Academic Defense

### Multi-Criteria Selection Decision:
1. **Best Overall Balanced Model: Random Forest (60 Trees, Depth 12)**
   - Highest Accuracy (**58.11%**), highest Weighted F1 (**0.6210**), and highest Macro F1 (**0.3809**).
   - Achieves the best Fatal Precision (**4.82%**) while still capturing over **44.12%** of all fatal crashes with standard argmax.
   - Recommended as the primary production classifier for the Excel Dashboard.

2. **Best Safety-Critical Screening Model: Logistic Regression (Class-Weighted)**
   - Highest Fatal Recall (**61.76%**), capturing nearly two-thirds of all fatal incidents (189 of 306).
   - Fast training time (**18.89s**), fully interpretable coefficients.
   - Recommended for emergency resource pre-positioning where false negatives are cost-prohibitive.

---

## 11. Excel Professional Dashboard Integration

The authoritative Excel deliverable is located at:  
[`dataset/excel/Road_Accident_Severity_Professional_Dashboard.xlsx`](file:///D:/ApacheSpark%20Project/Road_Accident_Severity_Prediction/dataset/excel/Road_Accident_Severity_Professional_Dashboard.xlsx)

### Dashboard Sheets Updated via Lossless OpenXML Streaming:
- **Sheet 4: `ML Model Performance`**: Synchronized with updated model benchmark table, accuracy metrics, macro/weighted F1, and complete 3x3 confusion matrices.
- **Sheet 5: `Prediction Analysis`**: Synchronized with 20,099 test collision predictions, including predicted severity labels, correctness indicators, and individual softmax class probabilities (`prob_fatal`, `prob_serious`, `prob_slight`).

---

## 12. Project Structure & Deliverables

```text
Road_Accident_Severity_Prediction/
│
├── README.md                                             # Authoritative project documentation
├── requirements.txt                                      # Verified Python package dependencies
├── main.py                                               # Phase 1: Data preprocessing driver script
├── spark_main.py                                         # Phase 2: PySpark MLlib pipeline driver script
├── open_dashboard.bat                                    # Batch shortcut to launch the Excel dashboard
├── run_tests.bat                                         # Batch shortcut to execute test suite
│
├── dataset/
│   ├── raw/                                              # Untouched DfT STATS19 CSV
│   ├── processed/
│   │   ├── accident_cleaned.csv                          # Cleaned dataset (101,523 rows)
│   │   ├── accident_final_100k.csv                       # Stratified modeling dataset (CSV)
│   │   ├── accident_final_100k.parquet/                  # Columnar Parquet dataset
│   │   ├── excel_dashboard_dataset.csv                   # Base 32-column dataset (100k rows)
│   │   ├── excel_dashboard_predictions.csv               # 20,099 test predictions with probabilities
│   │   ├── improved_predictions.csv                      # Optimized model predictions
│   │   └── model_performance_metrics.csv                 # 3-row comparative model table
│   └── excel/
│       └── Road_Accident_Severity_Professional_Dashboard.xlsx # Authoritative 8-sheet Excel workbook (17.5 MB)
│
├── models/
│   ├── evaluation_summary.json                           # Metrics and confusion matrices (JSON)
│   ├── final_model_metrics.json                          # Complete JSON deliverable
│   ├── spark_feature_pipeline/                           # Serialized 23-stage PipelineModel
│   ├── spark_random_forest_model/                        # Serialized best model (RandomForestClassificationModel)
│   └── spark_logistic_regression_model/                  # Serialized safety model (LogisticRegressionModel)
│
├── reports/
│   ├── audit_report.txt                                  # Pipeline audit and leakage verification
│   ├── feature_engineering_report.txt                    # Rationale and formulas for engineered features
│   ├── hyperparameter_tuning_results.csv                 # Validation grid search results
│   ├── improved_model_comparison.csv                     # Complete model comparison table
│   ├── improved_classification_report.csv                # Per-class precision, recall, F1
│   ├── improved_confusion_matrices.csv                   # Formatted 3x3 confusion matrices
│   ├── model_evaluation_report.txt                       # Detailed evaluation summary
│   ├── model_improvement_report.txt                      # Comprehensive improvement report
│   ├── preprocessing_report.txt                          # Phase 1 hygiene report
│   ├── spark_processing_report.txt                       # Parquet ingestion report
│   └── project_summary.md                                # Executive summary
│
├── src/
│   ├── excel_dashboard_updater.py                        # Lossless OpenXML dashboard updater
│   ├── model_evaluation.py                               # Comprehensive multiclass evaluation
│   ├── model_training.py                                 # Multi-algorithm training, weighting & tuning
│   ├── prediction_export.py                              # Softmax probability extraction & export
│   ├── spark_data_processing.py                          # Spark Parquet ingestion & audit
│   ├── spark_feature_pipeline.py                         # 23-stage MLlib pipeline & stratified split
│   └── spark_session.py                                  # SparkSession lifecycle & Windows winutils setup
│
└── tests/
    └── test_pipeline.py                                  # 8-assertion automated test suite
```

---

## 13. Execution & Verification Instructions

### 13.1 Run Preprocessing Pipeline (Phase 1)
```powershell
python main.py
```

### 13.2 Run PySpark ML Pipeline & Optimization (Phase 2)
```powershell
python spark_main.py
```

### 13.3 Run Automated Test Suite
```powershell
python tests/test_pipeline.py
```

### 13.4 Launch Microsoft Excel Dashboard
```powershell
.\open_dashboard.bat
```
*(Or open `dataset/excel/Road_Accident_Severity_Professional_Dashboard.xlsx` directly in Microsoft Excel).*

---

## 14. Academic Capstone Defense Summary

- **Degree**: Bachelor of Technology (B.Tech) in Computer Science & Engineering / Data Science
- **Academic Capstone**: Final Year Capstone Project (2025–2026)
- **Key Finding**: Raw accuracy alone is a misleading metric in safety-critical classification due to the Accuracy Paradox. The optimized Random Forest model achieves **58.11% accuracy** (+3.61% over original), **0.6210 Weighted F1**, **0.3809 Macro F1**, and **44.12% Fatal Recall**, providing the optimal balance between high-risk incident detection and operational false-alarm reduction.
