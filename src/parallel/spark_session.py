"""

Spark session management for the CSCE 5300 Parallel K-Means project.

This module centralizes Spark configuration so that development,

benchmarking, testing, and demonstration scripts use a consistent

execution environment.

The local development configuration is intentionally conservative

because development is performed on an Apple M2 MacBook Air with

8 GB of unified memory.

A different Spark master can be supplied later for distributed

cluster experiments without changing the clustering algorithm.

"""

from __future__ import annotations

from pyspark.sql import SparkSession

DEFAULT_APP_NAME = "CSCE5300-Parallel-KMeans"

DEFAULT_MASTER = "local[2]"

def create_spark_session(

    app_name: str = DEFAULT_APP_NAME,

    master: str = DEFAULT_MASTER,

    log_level: str = "WARN",

) -> SparkSession:

    """

    Create and configure a Spark session.

    Parameters

    ----------

    app_name:

        Human-readable name displayed by Spark.

    master:

        Spark execution target. During local development this will

        normally be local[1], local[2], or local[4].

    log_level:

        Spark logging level.

    Returns

    -------

    SparkSession

        Configured Spark session.

    """

    spark = (

        SparkSession.builder

        .appName(app_name)

        .master(master)

        .config("spark.driver.memory", "2g")

        .config("spark.executor.memory", "2g")

        .config("spark.sql.shuffle.partitions", "4")

        .getOrCreate()

    )

    spark.sparkContext.setLogLevel(log_level)

    return spark

def stop_spark_session(spark: SparkSession) -> None:

    """

    Stop a Spark session cleanly.

    Parameters

    ----------

    spark:

        Active Spark session.

    """

    spark.stop()
