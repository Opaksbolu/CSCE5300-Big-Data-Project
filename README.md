# CSCE 5300 Big Data Project
## Reproduction and Experimental Evaluation of Parallel K-Means for Big Data Clustering

This repository contains the implementation and experimental framework for a CSCE 5300 group project focused on reproducing and evaluating a parallel K-Means clustering method for large-scale data processing.

The project is based on the 2024 IEEE conference paper:

> H. Liu, Y. Bai, Z. Chen, and Z. Zhang,
> “Big Data Clustering Method Based on Parallel K-Means,”
> 2024 IEEE 4th International Conference on Power, Electronics and Computer Applications (ICPECA), Shenyang, China, 2024.

The objective is not simply to execute an existing clustering library. Instead, the project independently implements the major stages of the parallel clustering architecture described in the paper so that its behavior, performance, scalability, and clustering quality can be studied experimentally.

---

## 1. Project Objective

Traditional K-Means clustering repeatedly assigns data points to cluster centers and recalculates those centers until convergence. As datasets become larger, these repeated distance calculations and center updates can become computationally expensive.

The selected paper addresses this problem using Apache Spark and a parallel initialization strategy.

The project investigates the following research question:

> How does parallel K-Means compare with conventional K-Means in execution time, scalability, convergence behavior, and clustering quality as dataset size increases?

The implementation is designed to support controlled experiments involving different dataset sizes, Spark execution configurations, numbers of partitions, repeated experimental trials, clustering runtime, convergence behavior, Sum of Squared Errors (SSE), cluster sizes, and scalability measurements.

---

## 2. Reference Paper

**Title:** Big Data Clustering Method Based on Parallel K-Means
**Authors:** Haibo Liu, Yongbin Bai, Zhenhao Chen, and Zhenfeng Zhang
**Conference:** 2024 IEEE 4th International Conference on Power, Electronics and Computer Applications (ICPECA)
**Year:** 2024
**DOI:** 10.1109/ICPECA60615.2024.10470970

The paper proposes an improved Spark-based K-Means workflow in which data is partitioned across worker nodes, local clustering is performed within partitions, the resulting local centers are aggregated, and those centers are used to initialize a distributed global K-Means process.

---

## 3. Implemented Architecture

```text
Input Feature Vectors
        |
        v
Spark RDD
        |
        v
RDD Partitions
        |
        v
Local K-Means on Each Partition
        |
        v
Partition-Level Candidate Centers
        |
        v
Candidate-Center Aggregation
        |
        v
Global Initialization Centers
        |
        v
Broadcast Centers to Spark Workers
        |
        v
Parallel Point Assignment
        |
        v
reduceByKey Cluster Aggregation
        |
        v
Global Center Update
        |
        v
Convergence Check / Repeat
        |
        v
Final Centers, SSE, Cluster Counts,
Iteration History, and Timing Data
```

The implementation separates the mathematical clustering logic from Spark orchestration so that components can be tested independently.

---

## 4. Current Implementation

### Local K-Means

`src/parallel/local_kmeans.py`

Provides the deterministic mathematical clustering core used for partition-level clustering. Current features include K-Means++ initialization, deterministic random seeds, squared Euclidean distance, nearest-center assignment, center recalculation, empty-cluster handling, convergence detection, SSE calculation, dimensionality validation, and finite-value validation.

The local implementation intentionally does not depend on Spark, NumPy, or scikit-learn.

### Spark Session Configuration

`src/parallel/spark_session.py`

Provides reusable Spark session creation and shutdown utilities. The current local development configuration includes a configurable Spark master, 2 GB driver memory, 2 GB executor memory, four Spark SQL shuffle partitions, and a configurable Spark log level.

Local experiments can use execution configurations such as `local[1]`, `local[2]`, and `local[4]`.

### Partition-Level Clustering

`src/parallel/partition_clustering.py`

Executes local K-Means independently inside Spark RDD partitions. Each non-empty partition runs deterministic local K-Means and returns its local cluster centers and clustering metadata. These centers become candidates for global initialization.

### Candidate-Center Aggregation

`src/parallel/center_aggregation.py`

Combines the centers produced by individual Spark partitions and clusters the candidate centers again to reduce them to exactly `k` global initialization centers.

### Distributed Global K-Means Iteration

`src/parallel/global_iteration.py`

Implements one distributed global K-Means iteration. Current centers are broadcast to Spark workers, points are assigned to their nearest center, partial statistics are aggregated with `reduceByKey`, and new centers are calculated on the driver. Empty clusters retain their previous center.

### Global Convergence Loop

`src/parallel/global_kmeans.py`

Repeatedly executes distributed global iterations until the convergence tolerance is satisfied or the maximum iteration count is reached. The result includes final centers, iteration count, convergence status, final SSE, cluster counts, and iteration history.

### Complete Parallel K-Means Pipeline

`src/parallel/parallel_kmeans.py`

Connects partition clustering, candidate-center aggregation, and distributed global K-Means into the complete workflow. The pipeline exposes both final results and intermediate metadata for experimental analysis.

---

## 5. Runtime Instrumentation

The pipeline records runtime measurements using Python's monotonic high-resolution performance counter.

Current measurements include:

- partition clustering time,
- candidate-center aggregation time,
- total initialization time,
- global clustering time, and
- total pipeline runtime.

The timing relationships are covered by automated tests.

### Important Timing Note

The current `global_clustering_seconds` measurement wraps the complete `fit_global_kmeans()` call, including final evaluation. It therefore should not yet be treated as directly equivalent to the clustering-time definition used in the reference paper. Before direct paper-to-project runtime comparisons, iterative clustering time and final evaluation time will be separated.

---

## 6. Project Structure

```text
CSCE5300-Big-Data-Project/
|
|-- README.md
|-- .gitignore
|
|-- src/
|   |-- __init__.py
|   `-- parallel/
|       |-- __init__.py
|       |-- center_aggregation.py
|       |-- global_iteration.py
|       |-- global_kmeans.py
|       |-- local_kmeans.py
|       |-- parallel_kmeans.py
|       |-- partition_clustering.py
|       |-- partitioning.py
|       `-- spark_session.py
|
|-- scripts/
|   |-- __init__.py
|   |-- demo_parallel_kmeans.py
|   `-- demo_partitioning.py
|
|-- tests/
|   |-- __init__.py
|   `-- parallel/
|       |-- __init__.py
|       |-- conftest.py
|       |-- test_center_aggregation.py
|       |-- test_global_iteration.py
|       |-- test_global_kmeans.py
|       |-- test_local_kmeans.py
|       |-- test_parallel_kmeans.py
|       |-- test_partition_clustering.py
|       |-- test_spark_initialization_pipeline.py
|       `-- test_spark_partition_integration.py
|
`-- results/
    |-- figures/
    |   `-- .gitkeep
    `-- raw/
        `-- .gitkeep
```

---

## 7. Development Environment

The current development environment uses:

```text
Apache Spark: 4.2.0
PySpark:      4.2.0
Python:       3.13
Java:         OpenJDK 21
pytest:       9.1.1
```

The project is currently developed and validated using Spark local mode. Local Spark execution provides parallel worker threads on one physical computer and should not be interpreted as equivalent to the three-node distributed cluster used in the reference paper. This difference will be documented as an experimental limitation.

---

## 8. Environment Setup

Create and activate the Python virtual environment:

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
```

The current local Spark installation is located at `/opt/spark`.

Configure the Spark environment:

```bash
export SPARK_HOME=/opt/spark
export PYTHONPATH="$SPARK_HOME/python:$(find "$SPARK_HOME/python/lib" -maxdepth 1 -name 'py4j*.zip' -print -quit)"
```

Verify PySpark:

```bash
python -c "import pyspark; print(pyspark.__version__)"
```

Expected version for the current development environment:

```text
4.2.0
```

---

## 9. Running the Tests

Activate the virtual environment and configure Spark before running the test suite:

```bash
source .venv/bin/activate
export SPARK_HOME=/opt/spark
export PYTHONPATH="$SPARK_HOME/python:$(find "$SPARK_HOME/python/lib" -maxdepth 1 -name 'py4j*.zip' -print -quit)"
```

Run the complete test suite:

```bash
python -m pytest -v --tb=short
```

At the current stable milestone, the project contains **58 passing tests** covering local K-Means mathematics, validation behavior, partition-level clustering, candidate-center aggregation, Spark integration, distributed global iterations, convergence, complete pipeline execution, reproducibility, and runtime instrumentation.

---

## 10. Running the Demonstrations

### Spark Partitioning Demonstration

```bash
python -m scripts.demo_partitioning
```

This demonstration verifies record distribution across Spark partitions and validates partition-level processing.

### Complete Parallel K-Means Demonstration

```bash
python -m scripts.demo_parallel_kmeans
```

This demonstration runs the complete Parallel K-Means pipeline on a small synthetic dataset with clearly separated clusters. It is intended for functional validation and should not be treated as a large-scale performance benchmark.

---

## 11. Experimental Methodology

The experimental framework will evaluate the implementation using controlled configurations. Planned variables include dataset size, feature count, number of clusters (`k`), Spark master configuration, RDD partition count, random seed, maximum iterations, convergence tolerance, and trial number.

Measurements will include partition clustering time, center aggregation time, initialization time, global clustering time, total runtime, iterations completed, convergence status, SSE, and cluster counts.

Initial local scalability experiments are planned for `local[1]`, `local[2]`, and `local[4]`. Dataset sizes will increase progressively so that correctness, memory behavior, and runtime characteristics can be validated at each scale.

---

## 12. Experimental Results

Generated raw experiment data will be stored under:

```text
results/raw/
```

Generated figures will be stored under:

```text
results/figures/
```

Raw generated CSV and JSON experiment files are excluded from Git by default to prevent benchmark output from unnecessarily increasing repository size. Final report-ready results and figures may be selectively preserved when the experimental phase is complete.

---

## 13. Reproducibility

Reproducibility is a major design goal. The implementation uses explicit random seeds during local clustering and candidate-center aggregation.

Controlled experiments will record configuration metadata alongside each result, including dataset, record count, feature count, `k`, Spark master, partition count, random seed, trial number, iteration limits, tolerance, runtime, iterations completed, convergence status, and SSE.

This allows experimental results to be traced back to the configuration that produced them.

---

## 14. Testing Strategy

Development follows a staged workflow. Each major algorithmic component is implemented independently, tested with focused unit tests, validated with Spark integration tests where appropriate, checked against the complete regression suite, and committed only after validation.

This approach reduces the risk of introducing errors while expanding the distributed implementation.

---

## 15. Current Project Status

Completed milestones:

```text
[Completed] Spark development environment
[Completed] Spark partitioning validation
[Completed] Deterministic local K-Means
[Completed] Partition-level Spark clustering
[Completed] Candidate-center aggregation
[Completed] Distributed global K-Means iteration
[Completed] Global convergence loop
[Completed] Complete Parallel K-Means pipeline
[Completed] End-to-end Spark demonstration
[Completed] Runtime instrumentation
[Completed] Runtime validation
[Completed] 58-test regression suite
```

Current development milestone:

```text
[In Progress] Experimental benchmark foundation
```

Planned work:

```text
[Planned] Deterministic synthetic benchmark datasets
[Planned] Experiment metadata schema
[Planned] Controlled benchmark runner
[Planned] RDD persistence strategy
[Planned] Separate iteration and evaluation timing
[Planned] local[1] / local[2] / local[4] experiments
[Planned] Partition-count experiments
[Planned] Repeated benchmark trials
[Planned] Progressive dataset scaling
[Planned] Conventional K-Means comparison
[Planned] KDD Cup 99 experiment
[Planned] Purity evaluation
[Planned] SSE comparison
[Planned] Runtime and speedup analysis
[Planned] Memory investigation
[Planned] Comparison with reference-paper results
[Planned] Report-ready figures and tables
```

---

## 16. Current Experimental Limitations

### Local Spark Execution

The current environment uses Spark local mode on one physical machine. Although `local[1]`, `local[2]`, and `local[4]` allow different levels of parallel execution, these configurations are not equivalent to the three-machine Spark cluster used by the reference paper.

### Timing Definition

The current global clustering timer includes final cluster evaluation. This will be separated before direct runtime comparisons with the paper.

### Convergence Definition

The current implementation uses maximum squared center movement as its convergence criterion. The reference paper describes convergence using a change in clustering cost. This difference must either be aligned experimentally or clearly documented when comparing results.

### Dataset Persistence

The current pipeline does not explicitly persist the input RDD. Because iterative clustering accesses the same dataset multiple times, the persistence strategy will be controlled during benchmarking so that runtime comparisons remain meaningful and reproducible.

---

## 17. Team Workflow

The project is divided into four primary areas:

**Research and Algorithm Analysis** — understanding the reference paper, documenting the algorithm, and connecting the implementation to the original methodology.

**Data and Preprocessing** — datasets, cleaning, feature preparation, standardization, and reproducible data pipelines.

**Spark and Parallel Systems** — Spark configuration, partitioning, parallel clustering, distributed aggregation, performance instrumentation, scalability, and execution architecture.

**Experiments and Evaluation** — experimental comparisons, clustering metrics, visualization, statistical summaries, and report-ready results.

The components are designed so that algorithm, data, systems, and evaluation work can be developed independently while sharing a common experimental pipeline.

---

## 18. Academic Purpose

This repository was created for the CSCE 5300 Big Data course project.

The project focuses on independently reproducing, testing, and experimentally evaluating ideas from the selected research paper rather than presenting the original authors' implementation as our own.

All final experimental claims will be based on results produced by this repository and will be distinguished from results reported in the reference paper.
