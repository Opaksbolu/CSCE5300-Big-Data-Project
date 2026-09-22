"""
Shared pytest fixtures for the parallel K-Means test suite.

Spark integration tests use a common session so that individual test
modules do not need to duplicate Spark setup and teardown logic.
"""

from __future__ import annotations

import pytest

from src.parallel.spark_session import (
    create_spark_session,
    stop_spark_session,
)


@pytest.fixture(scope="session")
def spark():
    """
    Create one Spark session shared by the parallel test suite.

    The test environment uses two local worker threads to exercise
    Spark's parallel execution model without placing unnecessary
    pressure on the development machine.
    """

    session = create_spark_session(
        app_name="CSCE5300-Parallel-KMeans-Tests",
        master="local[2]",
        log_level="ERROR",
    )

    yield session

    stop_spark_session(session)
