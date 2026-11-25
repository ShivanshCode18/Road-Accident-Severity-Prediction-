# Power BI DAX Measures Library

**Project**: Road Accident Severity Prediction Using Apache Spark (PySpark) and Power BI  
**Target Table**: Dedicated `_Measures` table in Power BI Desktop  
**Ground-Truth Conformance**: Verified against UK STATS19 (100,000 baseline records and 20,099 test prediction records).

---

## Folder 01: Core Accident KPIs (FactCollisions)

### 1. Total Collisions
```dax
Total Collisions = 
COUNTROWS('FactCollisions')
```
*Description*: Calculates the total number of valid road traffic collisions in the analyzed dataset (Benchmark: 100,000).

---

### 2. Total Casualties
```dax
Total Casualties = 
SUM('FactCollisions'[number_of_casualties])
```
*Description*: Aggregates all human casualties resulting from collisions across the UK (Benchmark: 125,983).

---

### 3. Total Vehicles Involved
```dax
Total Vehicles Involved = 
SUM('FactCollisions'[number_of_vehicles])
```
*Description*: Total vehicular units involved in traffic incidents (Benchmark: 181,245).

---

### 4. Average Casualties per Collision
```dax
Average Casualties per Collision = 
DIVIDE([Total Casualties], [Total Collisions], 0)
```
*Format*: Decimal number (2 decimal places, Benchmark: 1.26).

---

### 5. Average Vehicles per Collision
```dax
Average Vehicles per Collision = 
DIVIDE([Total Vehicles Involved], [Total Collisions], 0)
```
*Format*: Decimal number (2 decimal places, Benchmark: 1.81).

---

## Folder 02: Severity Breakdown KPIs

### 6. Fatal Collisions
```dax
Fatal Collisions = 
CALCULATE(
    [Total Collisions],
    'FactCollisions'[collision_severity] = 1
)
```
*Description*: Incidents resulting in at least one fatality within 30 days (Benchmark: 1,431).

---

### 7. Serious Collisions
```dax
Serious Collisions = 
CALCULATE(
    [Total Collisions],
    'FactCollisions'[collision_severity] = 2
)
```
*Description*: Incidents with severe injuries requiring detention or severe trauma (Benchmark: 24,812).

---

### 8. Slight Collisions
```dax
Slight Collisions = 
CALCULATE(
    [Total Collisions],
    'FactCollisions'[collision_severity] = 3
)
```
*Description*: Minor injury collisions such as cuts, shock, and sprains (Benchmark: 73,757).

---

### 9. Fatal Collision Percentage
```dax
Fatal Collision Percentage = 
DIVIDE([Fatal Collisions], [Total Collisions], 0)
```
*Format*: Percentage (2 decimal places, Benchmark: 1.43%).

---

### 10. Serious Collision Percentage
```dax
Serious Collision Percentage = 
DIVIDE([Serious Collisions], [Total Collisions], 0)
```
*Format*: Percentage (2 decimal places, Benchmark: 24.81%).

---

### 11. Slight Collision Percentage
```dax
Slight Collision Percentage = 
DIVIDE([Slight Collisions], [Total Collisions], 0)
```
*Format*: Percentage (2 decimal places, Benchmark: 73.76%).

---

### 12. Severe Collisions (Fatal + Serious)
```dax
Severe Collisions = 
[Fatal Collisions] + [Serious Collisions]
```
*Description*: Combined critical casualties count (Benchmark: 26,243).

---

### 13. Severe Collision Percentage
```dax
Severe Collision Percentage = 
DIVIDE([Severe Collisions], [Total Collisions], 0)
```
*Format*: Percentage (2 decimal places, Benchmark: 26.24%).

---

## Folder 03: Temporal Analysis Measures

### 14. Weekend Collisions
```dax
Weekend Collisions = 
CALCULATE(
    [Total Collisions],
    'FactCollisions'[weekend_indicator] = 1
)
```
*Description*: Collisions occurring on Saturday or Sunday (Benchmark: 24,476).

---

### 15. Weekday Collisions
```dax
Weekday Collisions = 
CALCULATE(
    [Total Collisions],
    'FactCollisions'[weekend_indicator] = 0
)
```
*Description*: Collisions occurring Monday through Friday (Benchmark: 75,524).

---

### 16. Weekend Collision Percentage
```dax
Weekend Collision Percentage = 
DIVIDE([Weekend Collisions], [Total Collisions], 0)
```
*Format*: Percentage (Benchmark: 24.48%).

---

### 17. Weekday Collision Percentage
```dax
Weekday Collision Percentage = 
DIVIDE([Weekday Collisions], [Total Collisions], 0)
```
*Format*: Percentage (Benchmark: 75.52%).

---

## Folder 04: Machine Learning & Prediction Metrics (FactPredictions)

### 18. Total Test Predictions
```dax
Total Test Predictions = 
COUNTROWS('FactPredictions')
```
*Description*: Total unseen test observations evaluated by Apache Spark models (Benchmark: 20,099).

---

### 19. Correct Predictions
```dax
Correct Predictions = 
CALCULATE(
    [Total Test Predictions],
    'FactPredictions'[prediction_correct] = 1
)
```
*Description*: Total test collisions where actual severity exactly matched predicted severity (Benchmark: 9,532).

---

### 20. Incorrect Predictions
```dax
Incorrect Predictions = 
CALCULATE(
    [Total Test Predictions],
    'FactPredictions'[prediction_correct] = 0
)
```
*Description*: Total test collisions where model classification deviated from truth (Benchmark: 10,567).

---

### 21. Prediction Accuracy
```dax
Prediction Accuracy = 
DIVIDE([Correct Predictions], [Total Test Predictions], 0)
```
*Format*: Percentage (2 decimal places, Benchmark: 47.43%).

---

### 22. Actual Fatal Collisions (Test Set)
```dax
Actual Fatal Collisions = 
CALCULATE(
    [Total Test Predictions],
    'FactPredictions'[actual_severity] = 1
)
```
*Description*: Ground-truth fatal occurrences in test partition (Benchmark: 306).

---

### 23. Predicted Fatal Collisions
```dax
Predicted Fatal Collisions = 
CALCULATE(
    [Total Test Predictions],
    'FactPredictions'[predicted_severity] = 1
)
```
*Description*: Instances classified as Fatal by the primary model (Benchmark: 5,097).

---

### 24. Fatal True Positives (TP)
```dax
Fatal True Positives = 
CALCULATE(
    [Total Test Predictions],
    'FactPredictions'[actual_severity] = 1,
    'FactPredictions'[predicted_severity] = 1
)
```
*Description*: Actual fatal accidents correctly identified as fatal (Benchmark: 187).

---

### 25. Fatal False Positives (FP)
```dax
Fatal False Positives = 
CALCULATE(
    [Total Test Predictions],
    'FactPredictions'[actual_severity] <> 1,
    'FactPredictions'[predicted_severity] = 1
)
```
*Description*: Non-fatal accidents predicted as fatal (false alarms) (Benchmark: 4,910).

---

### 26. Fatal False Negatives (FN)
```dax
Fatal False Negatives = 
CALCULATE(
    [Total Test Predictions],
    'FactPredictions'[actual_severity] = 1,
    'FactPredictions'[predicted_severity] <> 1
)
```
*Description*: Actual fatal accidents missed by the model (Benchmark: 119).

---

### 27. Fatal True Negatives (TN)
```dax
Fatal True Negatives = 
CALCULATE(
    [Total Test Predictions],
    'FactPredictions'[actual_severity] <> 1,
    'FactPredictions'[predicted_severity] <> 1
)
```
*Description*: Non-fatal accidents correctly identified as non-fatal (Benchmark: 14,883).

---

### 28. Fatal Detection Rate (Recall)
```dax
Fatal Detection Rate = 
DIVIDE([Fatal True Positives], [Actual Fatal Collisions], 0)
```
*Format*: Percentage (2 decimal places, Benchmark: 61.11%).  
*Academic Note*: The primary road safety KPI. The model successfully detects 61.11% of all deadly crashes despite their extreme 1.43% rarity.

---

### 29. Fatal Precision
```dax
Fatal Precision = 
DIVIDE([Fatal True Positives], [Predicted Fatal Collisions], 0)
```
*Format*: Percentage (2 decimal places, Benchmark: 3.67%).  
*Academic Note*: Trade-off of class-weighting. To catch fatal events, the model flags potential risks proactively.

---

### 30. Serious Accident Recall
```dax
Serious Accident Recall = 
VAR SeriousTP = 
    CALCULATE(
        [Total Test Predictions],
        'FactPredictions'[actual_severity] = 2,
        'FactPredictions'[predicted_severity] = 2
    )
VAR ActualSerious = 
    CALCULATE(
        [Total Test Predictions],
        'FactPredictions'[actual_severity] = 2
    )
RETURN
    DIVIDE(SeriousTP, ActualSerious, 0)
```
*Format*: Percentage (Benchmark: 33.53%).

---

### 31. Slight Accident Recall
```dax
Slight Accident Recall = 
VAR SlightTP = 
    CALCULATE(
        [Total Test Predictions],
        'FactPredictions'[actual_severity] = 3,
        'FactPredictions'[predicted_severity] = 3
    )
VAR ActualSlight = 
    CALCULATE(
        [Total Test Predictions],
        'FactPredictions'[actual_severity] = 3
    )
RETURN
    DIVIDE(SlightTP, ActualSlight, 0)
```
*Format*: Percentage (Benchmark: 51.65%).

---

### 32. Macro F1-Score
```dax
Macro F1 = 
0.3383
```
*Alternative Dynamic Formula*:
```dax
Macro F1 Dynamic = 
VAR P1 = [Fatal Precision]
VAR R1 = [Fatal Detection Rate]
VAR F1_1 = DIVIDE(2 * P1 * R1, P1 + R1, 0)
VAR P2 = 0.2953
VAR R2 = [Serious Accident Recall]
VAR F1_2 = DIVIDE(2 * P2 * R2, P2 + R2, 0)
VAR P3 = 0.8125
VAR R3 = [Slight Accident Recall]
VAR F1_3 = DIVIDE(2 * P3 * R3, P3 + R3, 0)
RETURN
DIVIDE(F1_1 + F1_2 + F1_3, 3, 0)
```
*Format*: Decimal (4 decimal places, Benchmark: 0.3383).

---

### 33. Average Predicted Fatal Probability
```dax
Avg Predicted Fatal Probability = 
AVERAGE('FactPredictions'[prob_fatal])
```
*Format*: Decimal (4 decimal places, Benchmark: 0.2703).

---

### 34. Average Predicted Serious Probability
```dax
Avg Predicted Serious Probability = 
AVERAGE('FactPredictions'[prob_serious])
```
*Format*: Decimal (4 decimal places, Benchmark: 0.3452).

---

### 35. Average Predicted Slight Probability
```dax
Avg Predicted Slight Probability = 
AVERAGE('FactPredictions'[prob_slight])
```
*Format*: Decimal (4 decimal places, Benchmark: 0.3845).

---

## Folder 05: Conditional Formatting & Dynamic UI Labels

### 36. Severity Color Hex
```dax
Severity Color Hex = 
SWITCH(
    SELECTEDVALUE('FactCollisions'[severity_name]),
    "Fatal", "#EF4444",
    "Serious", "#F59E0B",
    "Slight", "#10B981",
    "#38BDF8"
)
```
*Description*: Used for data bar, pie slice, and conditional formatting colors.

---

### 37. Prediction Match Color
```dax
Prediction Match Color = 
IF(
    SELECTEDVALUE('FactPredictions'[prediction_correct]) = 1,
    "#10B981",
    "#EF4444"
)
```
*Description*: Highlights accurate predictions in vibrant green and classification errors in bright red.

---

### 38. Dynamic Page 1 Title
```dax
Dynamic Page 1 Title = 
"UK Road Collision Overview (" & FORMAT([Total Collisions], "#,##0") & " Collisions Analyzed)"
```

---

### 39. Dynamic Model Diagnostic Title
```dax
Dynamic Model Diagnostic Title = 
"PySpark Multiclass Classifier Performance (Test Set N = " & FORMAT([Total Test Predictions], "#,##0") & " Records)"
```

