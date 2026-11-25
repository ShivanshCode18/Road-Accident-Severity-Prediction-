"""Apache Spark and PySpark MLlib Configuration Module.

Centralizes local execution parameters, Java/Hadoop environment settings,
memory allocations, and directory path definitions for the Road Accident
Severity Prediction project.

Note:
    This project is configured for single-node local execution (Spark local[*]),
    suitable for academic evaluation, workstation development, and reproducibility.
    It does not require or assume a distributed Spark cluster.
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Base Project Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "dataset"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXCEL_DATA_DIR = DATA_DIR / "excel"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DOCS_DIR = PROJECT_ROOT / "docs"
WAREHOUSE_DIR = PROJECT_ROOT / "spark-warehouse"

# Spark Application & Resource Configuration (Local Mode)
SPARK_APP_NAME = "Road Accident Severity Prediction"
SPARK_MASTER = "local[*]"
SPARK_DRIVER_MEMORY = "4g"
SPARK_EXECUTOR_MEMORY = "2g"
SPARK_SHUFFLE_PARTITIONS = "8"
SPARK_ARROW_ENABLED = "true"
SPARK_LOG_LEVEL = "WARN"

# Processed Dataset Files
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "accident_cleaned.csv"
FINAL_100K_CSV_PATH = PROCESSED_DATA_DIR / "accident_final_100k.csv"
FINAL_100K_PARQUET_PATH = PROCESSED_DATA_DIR / "accident_final_100k.parquet"
EXCEL_DASHBOARD_DATA_PATH = PROCESSED_DATA_DIR / "excel_dashboard_dataset.csv"
EXCEL_DASHBOARD_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "excel_dashboard_predictions.csv"
MODEL_PERFORMANCE_METRICS_PATH = PROCESSED_DATA_DIR / "model_performance_metrics.csv"
EXCEL_WORKBOOK_PATH = EXCEL_DATA_DIR / "Road_Accident_Severity_Professional_Dashboard.xlsx"


def configure_spark_environment():
    """Verify and initialize JAVA_HOME, HADOOP_HOME, and PySpark Python binaries."""
    # Java configuration (defaults to known Windows install if not in environment)
    if not os.environ.get("JAVA_HOME"):
        default_java = r"C:\Java\jdk1.8.0_481"
        if os.path.exists(default_java):
            os.environ["JAVA_HOME"] = default_java
            logger.info("Set JAVA_HOME to: %s", default_java)

    # Hadoop / winutils configuration (Windows local mode requirement)
    if not os.environ.get("HADOOP_HOME"):
        default_hadoop = r"C:\hadoop-3.2.4"
        if os.path.exists(default_hadoop):
            os.environ["HADOOP_HOME"] = default_hadoop
            logger.info("Set HADOOP_HOME to: %s", default_hadoop)

    # Spark Home
    if not os.environ.get("SPARK_HOME"):
        default_spark = r"C:\Spark"
        if os.path.exists(default_spark):
            os.environ["SPARK_HOME"] = default_spark
            logger.info("Set SPARK_HOME to: %s", default_spark)

    # Ensure Hadoop and Java bin directories are on the system PATH
    current_path = os.environ.get("PATH", "")
    hadoop_bin = os.path.join(os.environ.get("HADOOP_HOME", ""), "bin")
    java_bin = os.path.join(os.environ.get("JAVA_HOME", ""), "bin")

    path_additions = []
    if os.path.exists(hadoop_bin) and hadoop_bin not in current_path:
        path_additions.append(hadoop_bin)
    if os.path.exists(java_bin) and java_bin not in current_path:
        path_additions.append(java_bin)

    if path_additions:
        os.environ["PATH"] = ";".join(path_additions) + ";" + current_path

    # Ensure worker and driver processes bind to the active Python interpreter
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def get_spark_configuration_dictionary() -> dict:
    """Return dictionary of recommended Spark configuration key-values."""
    return {
        "spark.master": SPARK_MASTER,
        "spark.driver.host": "127.0.0.1",
        "spark.driver.bindAddress": "127.0.0.1",
        "spark.driver.memory": SPARK_DRIVER_MEMORY,
        "spark.executor.memory": SPARK_EXECUTOR_MEMORY,
        "spark.sql.shuffle.partitions": SPARK_SHUFFLE_PARTITIONS,
        "spark.sql.warehouse.dir": str(WAREHOUSE_DIR.as_uri()),
        "spark.sql.execution.arrow.pyspark.enabled": SPARK_ARROW_ENABLED,
        "spark.ui.enabled": "false",
        "spark.ui.showConsoleProgress": "false",
    }

