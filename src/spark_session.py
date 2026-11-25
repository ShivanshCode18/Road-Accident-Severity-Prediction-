"""SparkSession Initialization and Lifecycle Management Module.

Configures a local Apache Spark environment suitable for Windows,
handling Java, Hadoop (winutils), and PySpark memory settings.
"""

import os
import sys
import logging
from pathlib import Path
from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

# Base project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WAREHOUSE_DIR = PROJECT_ROOT / "spark-warehouse"


def configure_spark_environment():
    """Ensure JAVA_HOME, HADOOP_HOME, and PySpark Python binaries are properly set."""
    # Java configuration (defaults to known Windows install if not in environment)
    if not os.environ.get("JAVA_HOME"):
        default_java = r"C:\Java\jdk1.8.0_481"
        if os.path.exists(default_java):
            os.environ["JAVA_HOME"] = default_java
            logger.info("Set JAVA_HOME to: %s", default_java)

    # Hadoop / winutils configuration
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

    # Add Hadoop bin and Java bin to PATH to prevent native IO issues on Windows
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

    # Ensure driver and workers use the active virtual environment Python interpreter
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable


def get_spark_session(app_name: str = "Road Accident Severity Prediction") -> SparkSession:
    """Create or retrieve a local SparkSession configured for Windows development.
    
    Args:
        app_name: Title of the Spark application.
        
    Returns:
        SparkSession: Configured SparkSession.
    """
    configure_spark_environment()
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing SparkSession: '%s' (local execution mode)...", app_name)
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.sql.warehouse.dir", str(WAREHOUSE_DIR.as_uri())) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.ui.enabled", "false") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()
        
    spark.sparkContext.setLogLevel("WARN")
    logger.info("SparkSession initialized successfully. Spark version: %s", spark.version)
    return spark


def display_dataframe_summary(df, name: str = "DataFrame", num_rows: int = 5):
    """Utility to print schema, record count, and sample rows from a Spark DataFrame."""
    count = df.count()
    cols = len(df.columns)
    print("=" * 80)
    print(f"SPARK DATA SUMMARY: {name.upper()}")
    print("=" * 80)
    print(f"Total Record Count : {count:,}")
    print(f"Total Column Count : {cols}")
    print("-" * 80)
    print("Schema:")
    df.printSchema()
    print("-" * 80)
    print(f"Sample First {num_rows} Records:")
    df.show(num_rows, truncate=False)
    print("=" * 80)


def stop_spark_session(spark: SparkSession):
    """Safely terminate the SparkSession."""
    if spark is not None:
        logger.info("Stopping SparkSession...")
        spark.stop()
        logger.info("SparkSession stopped.")

