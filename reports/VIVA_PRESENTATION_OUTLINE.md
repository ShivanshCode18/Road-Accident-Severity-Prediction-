# Academic Capstone Viva Defense Presentation Deck & Speaker Notes
## Road Accident Severity Prediction Using Apache Spark (PySpark) and Excel Dashboard

**Project**: B.Tech Final Year Capstone Project  
**Target Duration**: 15 Minutes (10-12 mins presentation, 3-5 mins Q&A)  
**Presenter(s)**: Data Science & Big Data Engineering Team  
**Evaluation Panel**: Department Academic Review Committee / External Examiner  

---

## Slide 1: Title Slide & Project Scope
- **Slide Title**: Road Accident Severity Prediction Using Apache Spark (PySpark) and Excel Dashboard
- **Subtitle**: An End-to-End Big Data Engineering, Machine Learning & Business Intelligence Solution
- **Visuals**: Logos/Icons of Apache Spark, PySpark, Python, and Microsoft Excel. High-level three-tier architecture diagram.
- **Key Talking Points (1 min)**:
  - Good morning, respected committee members and professors.
  - Today, we present our capstone project focused on predicting road traffic collision severity across the United Kingdom using official Department for Transport (STATS19) data.
  - The project combines **Big Data Data Engineering**, **Distributed Machine Learning with Apache Spark (PySpark MLlib)**, and an **Authoritative Microsoft Excel Professional Dashboard**.
  - Our primary objective is not merely achieving high theoretical accuracy, but solving the life-critical challenge of **detecting fatal collisions in highly imbalanced real-world data**.

---

## Slide 2: The Problem: Road Trauma & The Accuracy Paradox
- **Slide Title**: Problem Definition & The Reality of Class Imbalance
- **Visuals**:
  - Pie chart showing extreme class imbalance: Slight (73.76%), Serious (24.81%), Fatal (1.43%).
  - Comparison table: "Standard Classifier" (74% accuracy, 0% fatal recall) vs. "Our Safety-Oriented Model" (47.4% accuracy, 61.11% fatal recall).
- **Speaker Notes (1 min)**:
  - Traffic safety datasets present an extreme challenge: **class imbalance**. Less than 1.5% of crashes are fatal.
  - A naive classifier that predicts "Slight" for every single crash achieves **73.8% accuracy**, but fails to catch even a single fatal crash. In public safety, this is catastrophic.
  - Furthermore, many existing academic papers suffer from **target leakage**—including post-crash consequences like the number of casualties into model features.
  - Our research explicitly prevents leakage and formulates an **imbalance-aware, recall-maximizing framework**.

---

## Slide 3: Dataset Ingestion & Leakage Prevention
- **Slide Title**: Data Pipeline: Clean Ingestion & Strict Leakage Elimination
- **Visuals**: Flow diagram showing raw 101,525 rows $\rightarrow$ 18 columns permanently excluded $\rightarrow$ 100,000 stratified records exported to Parquet.
- **Speaker Notes (1 min)**:
  - We sourced the raw UK STATS19 2025 collisions dataset comprising 101,525 records and 44 columns.
  - We dropped 2 records with missing geographic coordinates, ensuring 100% valid UK boundaries.
  - **Zero Duplicates**: Audited unique collision identifiers; zero duplicate records existed.
  - **Leakage Prevention**: We strictly stripped out 18 fields, including post-accident outcomes like `number_of_casualties`, `enhanced_severity_collision`, and `collision_adjusted_severity`.
  - We engineered 14 spatial, temporal, and environmental risk features and exported a clean, reproducible 100,000-record Parquet data lake.

---

## Slide 4: System Architecture & Local Spark Optimization
- **Slide Title**: Architecture: Distributed PySpark Processing
- **Visuals**: Architectural block diagram showing Ingestion $\rightarrow$ SparkSession (`local[*]`) $\rightarrow$ MLlib Pipeline $\rightarrow$ Model Registry $\rightarrow$ Excel Dashboard.
- **Speaker Notes (1 min)**:
  - We initialized Apache Spark 3.5.9 on top of Java 8 and Hadoop 3.2.4 using PySpark.
  - To simulate production cluster workflows on a workstation without claiming fake cluster infrastructure, we configured Spark locally using `local[*]` with a 4GB driver, 2GB executor, and 8 shuffle partitions.
  - The feature engineering is completely decoupled from model fitting using Spark's Directed Acyclic Graph (DAG) optimization.

---

## Slide 5: PySpark MLlib 23-Stage Pipeline & Stratified Split
- **Slide Title**: Feature Engineering & ML Pipeline Architecture
- **Visuals**: Diagram of the 23-stage pipeline:
  - 11 `StringIndexers` (`handleInvalid="keep"`)
  - 11 `OneHotEncoders` (`dropLast=False`)
  - 1 `VectorAssembler` (combining 11 OHE vectors + 7 numerical features)
- **Speaker Notes (1 min)**:
  - In machine learning pipelines, fitting transformations on test data creates data leakage.
  - We split our 100,000-record dataset into **80% Train (79,901 rows)** and **20% Test (20,099 rows)** using stratified class sampling.
  - We validated complete record isolation using a `left_anti` join on `collision_index`, proving zero test records appeared in the training set.
  - The 23-stage MLlib pipeline was fitted **strictly on the training partition**.

---

## Slide 6: Tackling Imbalance: Inverse-Frequency Class Weighting
- **Slide Title**: Cost-Sensitive Learning: Mathematical Formulation
- **Visuals**:
  - Class weighting formula: $w_c = \frac{N}{K \cdot N_c}$
  - Bar chart showing computed weights: $w_{\text{Fatal}} = 23.6744$, $w_{\text{Serious}} = 1.3338$, $w_{\text{Slight}} = 0.4529$.
- **Speaker Notes (1.5 mins)**:
  - Instead of synthetic oversampling (SMOTE) which can synthesize physically impossible weather/spatial combinations, we utilized **Cost-Sensitive Learning**.
  - We penalized errors on the minority fatal class by computing inverse class frequencies across the 79,901 training records.
  - With a weight of **23.67** on Fatal crashes versus **0.45** on Slight crashes, any misclassified fatal incident incurs over 52 times the gradient penalty of a misclassified slight incident.
  - This mathematically forces the decision boundary toward high fatal recall.

---

## Slide 7: Model Benchmarking & Performance Comparison
- **Slide Title**: Experimental Model Evaluation (Untouched Test Partition N = 20,099)
- **Visuals**:
  - Side-by-side metric comparison table:
    | Model | Accuracy | Weighted F1 | Macro F1 | Fatal Recall | Training Time |
    |---|:---:|:---:|:---:|:---:|:---:|
    | **Logistic Regression (Weighted)** | **47.43%** | **0.5464** | **0.3383** | **61.11%** | **9.54 s** |
    | Decision Tree ($depth=8$) | 46.02% | 0.5346 | 0.3257 | 59.48% | 10.04 s |
    | Random Forest ($50\text{ trees}$) | 54.50% | 0.5985 | 0.3667 | 54.90% | 57.59 s |
- **Speaker Notes (1.5 mins)**:
  - We trained and benchmarked three distinct multiclass architectures.
  - **Random Forest** achieved the highest raw accuracy (54.50%) and Weighted F1 (0.5985).
  - However, **Multinomial Logistic Regression with inverse class weighting achieved the highest Fatal Recall: 61.11%**, detecting 187 out of 306 fatal crashes in the test set.
  - In transportation risk assessment, missing a fatal crash carries immense consequences. Therefore, Logistic Regression was selected as our primary production model.

---

## Slide 8: In-Depth Diagnostic: Confusion Matrix & Trade-offs
- **Slide Title**: Diagnostic Evaluation: Primary Model Confusion Matrix
- **Visuals**:
  - 3x3 Confusion Matrix:
    - True Fatal: 187 detected (TP), 119 missed (FN)
    - True Serious: 1,624 detected
    - True Slight: 7,721 detected
  - Precision-Recall Trade-off curves.
- **Speaker Notes (1 min)**:
  - Looking at our primary model's confusion matrix on the 20,099 test cases:
  - It successfully identified 187 fatal crashes out of 306.
  - The trade-off is lower fatal precision (3.67%), resulting in 4,910 false alarms.
  - For emergency management agencies, preparing trauma centers for a crash that turns out to be serious is acceptable; missing a fatal incident due to an under-resourced response is catastrophic.

---

## Slide 9: Excel Dashboard Architecture: Formula Engine & Dynamic Filtering
- **Slide Title**: Business Intelligence Layer: Microsoft Excel Professional Dashboard
- **Visuals**:
  - High-level workbook schematic showing the 8 worksheets, 41 embedded native Excel charts, the underlying Calc Engine, and the Map Sample.
  - Snapshot of formula-driven `COUNTIFS`/`SUMIFS` aggregation matrix.
- **Speaker Notes (1 min)**:
  - To bring these technical results to decision-makers and academic evaluators, we developed an authoritative **Microsoft Excel Professional Dashboard** (`Road_Accident_Severity_Professional_Dashboard.xlsx`), replacing earlier Power BI prototypes.
  - The workbook features 8 structured worksheets and 41 native Excel visualization charts.
  - The backend is powered by a formula-driven **Calc Engine** sheet responding dynamically to 7 dropdown filters on the Executive Dashboard.
  - Spatial mapping is powered by a stratified 7,431-point **Map Sample** with conditional `#N/A` masking for responsive coordinate scatter charting.
  - All benchmarks were verified against our automated Python validation script (`src/dashboard_export_validation.py`).

---

## Slide 10: Dashboard Walkthrough: Executive & Geographical Worksheets
- **Slide Title**: Dashboard Sheets 1–3: Executive KPIs, Environmental Hazards & Geographic Risk
- **Visuals**: Layouts of Sheet 1 (Executive Dashboard), Sheet 2 (Road & Environment), and Sheet 3 (Geographical Analysis).
- **Speaker Notes (1.5 mins)**:
  - **Sheet 1 (Executive Dashboard)**: Displays headline KPIs (100,000 Collisions, 125,983 Casualties, 1.43% Fatal, 24.81% Serious), severity donuts, monthly progression, and 24-hour diurnal curves (rush hour peaks at 08:00 and 17:00).
  - **Sheet 2 (Road & Environment)**: Pinpoints infrastructure risks across speed limits, single carriageways (~74% of severe crashes), adverse weather, and unlit roads at night.
  - **Sheet 3 (Geographical Analysis)**: Shows UK Police Force incident rankings and the **Urban vs. Rural paradox**:
    - Rural roads account for only 33.9% of collisions, but cause the majority of all fatalities due to higher traveling speeds and longer trauma transit times.

---

## Slide 11: Dashboard Walkthrough: ML Evaluation & Prediction Audit
- **Slide Title**: Dashboard Sheets 4 & 5: Model Comparisons & Test Prediction Audit
- **Visuals**: Layouts of Sheet 4 (ML Model Performance) and Sheet 5 (Prediction Analysis).
- **Speaker Notes (1.5 mins)**:
  - **Sheet 4 (ML Model Performance)**: Directly visualizes PySpark benchmark metrics across all three classifiers, interactive 3x3 confusion matrices, and safety recall comparisons.
  - **Sheet 5 (Prediction Analysis)**: Enables itemized audit of all 20,099 held-out test predictions, displaying actual severity, predicted severity, correctness flags, and individual class probabilities (`prob_fatal`, `prob_serious`, `prob_slight`).

---

## Slide 12: Key Analytical Findings & Transport Policy Recommendations
- **Slide Title**: Actionable Insights & Public Safety Recommendations
- **Visuals**: Summary infographic of top 3 hazard drivers and policy recommendations.
- **Speaker Notes (1 min)**:
  - **1. Rural Road Speed Intervention**: 60-70 mph rural single carriageways demand immediate automated speed enforcement and rumble strips.
  - **2. Unlit Nighttime Corridors**: Collisions occurring on unlit roads at night exhibit a 3.1% fatality rate compared to 0.9% in daylight. Highway authorities should prioritize dynamic LED smart-lighting.
  - **3. Rush Hour Dynamic Deployment**: Emergency medical services (EMS) should pre-stage trauma units near high-volume motorway junctions during the 16:00–18:00 evening rush.

---

## Slide 13: Technical Highlights & Viva Defense Strategies
- **Slide Title**: Defense Summary: Top Technical Accomplishments
- **Visuals**: Checkmark badges:
  - [x] Zero Target Leakage Invariant
  - [x] 100% Stratified Record Disjointness
  - [x] Spark MLlib 23-Stage Reusable Pipeline
  - [x] 61.11% Fatal Recall via Class Weighting
  - [x] Fully Scripted Automated Test Suite (8/8 Passed)
- **Speaker Notes (1 min)**:
  - In conclusion, this project provides a rigorous, scalable, and fully tested pipeline from raw telemetry to executive visualization.
  - All code has been structured into clean, modular Python modules, validated by automated unit tests, and backed by comprehensive documentation.
  - Thank you. We now welcome questions from the examination committee.

---

## Committee Q&A Defense Guide: Top 7 Anticipated Questions

### Question 1: "Why is your model accuracy only 47.4%? Couldn't you get 80% or 90%?"
- **Answer**: 
  > "In imbalanced public safety data, raw accuracy is deeply misleading. Because 73.8% of crashes are Slight, a dummy model predicting 'Slight' for every record achieves 73.8% accuracy with **0% Fatal Recall**—meaning it fails to detect a single death. To protect human life, we applied inverse class weights ($w_{\text{Fatal}}=23.67$). This forced the model to prioritize fatal events, achieving a **61.11% Fatal Recall**. In transportation safety, detecting 61% of fatal crashes is vastly superior to a high-accuracy model that detects zero."

### Question 2: "Why did you exclude `number_of_casualties` from your model features?"
- **Answer**: 
  > "Target leakage. The number of casualties is an outcome of the collision, not a causal prior. When a 999/911 call is placed, authorities know the location, weather, and speed limit, but rarely the precise casualty count. Training on post-outcome variables creates artificial inflated accuracy that collapses in real-world deployment. We strictly retained casualty count for Power BI historical reporting, but excluded it from MLlib predictors."

### Question 3: "Why did you choose Apache Spark MLlib instead of scikit-learn or XGBoost?"
- **Answer**: 
  > "Scikit-learn operates strictly in single-node RAM and cannot scale to multi-gigabyte or streaming sensor data. Spark MLlib encapsulates feature transformers and estimators into a unified Directed Acyclic Graph (DAG). While our project executed on `local[*]` for workstation reproducibility, the exact same code, schema, and serialized ML pipeline can be submitted directly to enterprise multi-node Hadoop YARN or Databricks clusters without code refactoring."

### Question 4: "Why use Logistic Regression instead of Random Forest as your final model?"
- **Answer**: 
  > "Random Forest attained higher overall accuracy (54.50%), but its Fatal Recall dropped to 54.90%. Multinomial Logistic Regression with inverse class weights achieved **61.11% Fatal Recall**. In emergency decision support, a false alarm (over-preparing hospital trauma centers) is manageable; a false negative (failing to recognize a fatal crash) is deadly."

### Question 5: "How does Power BI interact with Apache Spark in your solution?"
- **Answer**: 
  > "Power BI ingests the post-processed datasets, model performance matrices, and test predictions generated by PySpark. We structured the data in Power BI as a clean Star Schema, connecting fact tables to conformed dimension tables (`DimSeverity`, `DimDate`, `DimWeather`) with 1-to-many relationships, eliminating circular dependencies and enabling high-performance DAX evaluations."

### Question 6: "How did you ensure training and testing data did not overlap?"
- **Answer**: 
  > "We used stratified class fractions to partition the 100,000 records into 80% train (79,901) and 20% test (20,099). We then executed Spark's `left_anti` join on the unique `collision_index`, which mathematically verified that exactly zero test records were present in the training set."

### Question 7: "What is your solution's biggest real-world limitation?"
- **Answer**: 
  > "The primary limitation is the absence of high-frequency vehicle CAN-bus telemetry (e.g., speed at impact, braking force, anti-lock brake activation) and driver physiological metrics (e.g., alcohol intoxication levels, eye-tracking fatigue). Integrating vehicle-to-everything (V2X) streaming telemetry into Spark Structured Streaming is our recommended future enhancement."

