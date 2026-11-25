"""End-to-End Pipeline & Artifact Verification Test Suite.

Academic B.Tech Project: Road Accident Severity Prediction with PySpark & Excel Dashboard
"""

import os
import sys
import json
import unittest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class TestDataArtifacts(unittest.TestCase):
    """Test data artifacts for presence, row counts, schema conformance, and dashboard deliverables."""

    def test_01_processed_files_exist(self):
        """Verify all processed data files and final Excel dashboard exist."""
        required_files = [
            os.path.join(PROJECT_ROOT, "dataset", "processed", "accident_cleaned.csv"),
            os.path.join(PROJECT_ROOT, "dataset", "processed", "accident_final_100k.csv"),
            os.path.join(PROJECT_ROOT, "dataset", "processed", "accident_final_100k.parquet"),
            os.path.join(PROJECT_ROOT, "dataset", "processed", "excel_dashboard_dataset.csv"),
            os.path.join(PROJECT_ROOT, "dataset", "processed", "excel_dashboard_predictions.csv"),
            os.path.join(PROJECT_ROOT, "dataset", "processed", "model_performance_metrics.csv"),
            os.path.join(PROJECT_ROOT, "dataset", "excel", "Road_Accident_Severity_Professional_Dashboard.xlsx"),
        ]
        for f in required_files:
            self.assertTrue(os.path.exists(f), f"Missing data artifact: {f}")
            self.assertGreater(os.path.getsize(f), 0, f"File is empty: {f}")

    def test_02_reports_exist(self):
        """Verify all phase audit reports exist."""
        required_reports = [
            os.path.join(PROJECT_ROOT, "reports", "preprocessing_report.txt"),
            os.path.join(PROJECT_ROOT, "reports", "data_quality_report.txt"),
            os.path.join(PROJECT_ROOT, "reports", "spark_processing_report.txt"),
            os.path.join(PROJECT_ROOT, "reports", "model_evaluation_report.txt"),
            os.path.join(PROJECT_ROOT, "reports", "project_summary.md"),
        ]
        for r in required_reports:
            self.assertTrue(os.path.exists(r), f"Missing report: {r}")
            self.assertGreater(os.path.getsize(r), 100, f"Report too short: {r}")

    def test_03_dashboard_documentation_exists(self):
        """Verify all Excel dashboard documentation and dictionary files exist."""
        dash_files = [
            os.path.join(PROJECT_ROOT, "dashboard", "README.md"),
            os.path.join(PROJECT_ROOT, "dashboard", "excel_dashboard_documentation.md"),
            os.path.join(PROJECT_ROOT, "dashboard", "dashboard_data_dictionary.md"),
            os.path.join(PROJECT_ROOT, "dataset", "excel", "Road_Accident_Severity_Professional_Dashboard.xlsx"),
        ]
        for p in dash_files:
            self.assertTrue(os.path.exists(p), f"Missing dashboard file: {p}")
            self.assertGreater(os.path.getsize(p), 100, f"File too small: {p}")

    def test_04_fact_collisions_row_count_and_schema(self):
        """Verify excel_dashboard_dataset.csv has exactly 100,000 rows and 0 null coordinates."""
        csv_path = os.path.join(PROJECT_ROOT, "dataset", "processed", "excel_dashboard_dataset.csv")
        df = pd.read_csv(csv_path, dtype={"collision_index": str})
        self.assertEqual(len(df), 100000, "Collisions dataset must have exactly 100,000 rows")
        self.assertEqual(df["collision_index"].nunique(), 100000, "collision_index must be 100% unique")
        self.assertEqual(df["latitude"].isnull().sum(), 0, "Latitude must have 0 nulls")
        self.assertEqual(df["longitude"].isnull().sum(), 0, "Longitude must have 0 nulls")

        # Target column validation
        self.assertIn("collision_severity", df.columns)
        self.assertIn("severity_name", df.columns)
        severity_counts = df["collision_severity"].value_counts().to_dict()
        self.assertEqual(severity_counts.get(1, 0), 1431, "Fatal count mismatch")
        self.assertEqual(severity_counts.get(2, 0), 24812, "Serious count mismatch")
        self.assertEqual(severity_counts.get(3, 0), 73757, "Slight count mismatch")

    def test_05_fact_predictions_metrics(self):
        """Verify excel_dashboard_predictions.csv has 20,099 rows and valid probabilities."""
        csv_path = os.path.join(PROJECT_ROOT, "dataset", "processed", "excel_dashboard_predictions.csv")
        df = pd.read_csv(csv_path, dtype={"collision_index": str})
        self.assertEqual(len(df), 20099, "Predictions dataset must have exactly 20,099 test rows")

        # Probability simplex check: sum of probabilities must approximate 1.0
        prob_sum = df["prob_fatal"] + df["prob_serious"] + df["prob_slight"]
        np.testing.assert_allclose(prob_sum.values, 1.0, atol=1e-4, err_msg="Probabilities must sum to 1.0")

        # Metric assertions for model predictions
        correct = (df["actual_severity"] == df["predicted_severity"]).sum()
        accuracy = correct / len(df)
        self.assertTrue(0.40 <= accuracy <= 0.75, f"Model accuracy out of valid bounds: {accuracy}")

        # Fatal recall: TP / (TP + FN)
        actual_fatal = (df["actual_severity"] == 1).sum()
        self.assertEqual(actual_fatal, 306, "Actual fatal count in test set mismatch")
        fatal_tp = ((df["actual_severity"] == 1) & (df["predicted_severity"] == 1)).sum()
        fatal_recall = fatal_tp / actual_fatal
        self.assertTrue(0.40 <= fatal_recall <= 0.75, f"Fatal recall out of valid bounds: {fatal_recall}")

    def test_06_model_performance_summary(self):
        """Verify model_performance_metrics.csv contains 3 models with complete metrics."""
        csv_path = os.path.join(PROJECT_ROOT, "dataset", "processed", "model_performance_metrics.csv")
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 3, "Model performance table must contain exactly 3 models")
        expected_models = {"Logistic Regression", "Decision Tree", "Random Forest"}
        self.assertEqual(set(df["model_name"]), expected_models)

        # Check Random Forest has highest accuracy
        rf_acc = df.loc[df["model_name"] == "Random Forest", "accuracy"].values[0]
        lr_acc = df.loc[df["model_name"] == "Logistic Regression", "accuracy"].values[0]
        self.assertGreater(rf_acc, lr_acc)

        # Check Logistic Regression has highest Fatal Recall
        rf_rec = df.loc[df["model_name"] == "Random Forest", "fatal_recall"].values[0]
        lr_rec = df.loc[df["model_name"] == "Logistic Regression", "fatal_recall"].values[0]
        self.assertGreater(lr_rec, rf_rec)

    def test_07_serialized_models(self):
        """Verify Spark MLlib model persistence directories and metadata."""
        pipe_dir = os.path.join(PROJECT_ROOT, "models", "spark_feature_pipeline")
        model_dir = os.path.join(PROJECT_ROOT, "models", "spark_logistic_regression_model")
        summary_file = os.path.join(PROJECT_ROOT, "models", "evaluation_summary.json")

        self.assertTrue(os.path.exists(pipe_dir), "Pipeline model directory missing")
        self.assertTrue(os.path.exists(model_dir), "Logistic Regression model directory missing")
        self.assertTrue(os.path.exists(summary_file), "Evaluation summary JSON missing")

        with open(summary_file, "r") as f:
            summary = json.load(f)
        models_dict = summary.get("models", summary)
        self.assertIn("Logistic Regression", models_dict)
        self.assertIn("Decision Tree", models_dict)
        self.assertIn("Random Forest", models_dict)

    def test_08_no_target_leakage_in_features(self):
        """Confirm post-outcome features were strictly excluded from model feature list."""
        from src.spark_feature_pipeline import EXCLUDED_COLUMNS, CATEGORICAL_PREDICTORS, NUMERICAL_PREDICTORS
        from src.config import PERMANENTLY_DROPPED_COLUMNS

        leakage_cols = [
            "number_of_casualties",
            "enhanced_severity_collision",
            "collision_adjusted_severity_serious",
            "collision_adjusted_severity_slight",
            "did_police_officer_attend_scene_of_accident",
        ]
        all_excluded = set(EXCLUDED_COLUMNS).union(set(PERMANENTLY_DROPPED_COLUMNS))
        for col in leakage_cols:
            self.assertIn(col, all_excluded, f"Potential leakage column {col} was not in excluded sets!")

        # Also confirm none of these leakage columns are in predictor lists
        all_predictors = set(CATEGORICAL_PREDICTORS).union(set(NUMERICAL_PREDICTORS))
        for col in leakage_cols:
            self.assertNotIn(col, all_predictors, f"Target leakage {col} found in ML model predictors!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
