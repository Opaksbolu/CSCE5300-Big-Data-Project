# Experimental Methodology

## 1. Research Objective and Experimental Design

This project investigates the performance and clustering behavior of a Spark-based Parallel K-Means implementation inspired by the research paper *Big Data Clustering Method Based on Parallel K-Means* by Liu et al. (2024).

The primary objective is to evaluate whether parallelizing K-Means operations improves execution time as dataset size increases while maintaining comparable clustering quality.

To investigate this question, we implemented a Parallel K-Means pipeline using Apache Spark. The implementation performs local clustering within individual Spark partitions, combines the resulting candidate centers to initialize the global model, and then performs distributed global clustering.

We also implemented a conventional, single-process K-Means baseline to examine differences in initialization behavior, clustering quality, and runtime. Because the baseline and parallel implementations use different initialization procedures, we interpret their runtime results cautiously rather than treating them as directly equivalent workloads.

Our experimental evaluation focuses on three main areas:

1. **Execution time:** Measuring the time required to complete the clustering pipeline under different Spark local execution configurations.
2. **Parallel scalability:** Evaluating how runtime changes when Spark is configured with one, two, or four local worker threads.
3. **Clustering quality:** Comparing the sum of squared errors (SSE) produced by the algorithms and verifying whether parallel execution preserves clustering quality.

The principal scalability experiments use synthetic datasets containing 10,000 and 50,000 records. Each dataset has 20 numerical features and five underlying clusters.

All scalability experiments are performed on a single Apple Silicon computer using Apache Spark in local mode. Therefore, the results measure single-machine parallel execution rather than performance across multiple physical cluster nodes.

For reproducibility, the experiments use controlled random seeds, fixed clustering parameters, repeated measurements, and an initial unmeasured warm-up run for each Spark-master process before its benchmark matrix.

The evaluation also considers limitations associated with initialization strategies, memory requirements, the number of experimental repetitions, and the use of synthetic data.

## 2. Parallel K-Means Algorithm and Implementation

### 2.1 Overview

Our implementation follows a partition-based initialization and distributed refinement strategy inspired by Liu et al. (2024). Instead of calling Spark MLlib's K-Means estimator, the project implements the clustering operations explicitly in Python and PySpark. The process has three principal stages: local clustering within Spark partitions, aggregation of candidate centers, and global refinement across the distributed dataset.

### 2.2 Nine-Step Algorithm

1. **Prepare the input.** Generate a numerical dataset with a specified number of observations, features, clusters, and random seed. Construct a Spark RDD with a fixed partition count.
2. **Partition the observations.** Distribute the RDD across four Spark partitions for the reported scalability experiments.
3. **Cluster within each partition.** Use `mapPartitionsWithIndex` to run local K-Means independently on each nonempty partition. Each local run uses seeded K-Means++ initialization and squared Euclidean distance.
4. **Collect candidate centers.** Return the local centers from the partition workers to the driver. With four nonempty partitions and five local clusters each, the expected candidate pool contains 20 centers.
5. **Aggregate candidate centers.** Run local K-Means on the collected centers using five deterministic initialization restarts. Select the candidate-center solution with the lowest aggregation SSE to initialize the global model. These candidate centers are **not weighted by their originating partition's population**.
6. **Broadcast global centers.** Distribute the current global centers to Spark workers using a broadcast variable for each global refinement iteration.
7. **Assign points and aggregate statistics.** Assign each observation to its nearest center, then use `reduceByKey` to combine coordinate sums, point counts, and SSE contributions by cluster.
8. **Update and check convergence.** Compute updated global centers from the aggregated sums and counts. Repeat distributed assignment and updating until the absolute change in consecutive-iteration SSE is at most `1e-6`, or 100 global iterations have been reached.
9. **Evaluate final centers.** Run a final distributed evaluation using the returned centers to calculate final SSE and cluster counts. Record convergence information, iteration counts, and phase runtimes.

This is a nine-step description of **our implementation**, not a claim that every implementation detail exactly reproduces the paper.

### 2.3 Objective Function and Center Updates

K-Means minimizes the within-cluster sum of squared Euclidean distances:

\[
\mathrm{SSE} = \sum_{i=1}^{n} \left\|x_i - \mu_{c(i)}\right\|_2^2,
\]

where `x_i` is observation `i`, `c(i)` is its assigned cluster, and `μ` denotes a cluster center. For a nonempty cluster `j`, its updated center is the arithmetic mean of assigned observations:

\[
\mu_j = \frac{1}{|C_j|}\sum_{x_i \in C_j} x_i.
\]

The distributed implementation calculates these means from aggregated coordinate sums and counts rather than collecting all assigned observations at the driver.

### 2.4 Local and Global Stopping Criteria

Local K-Means uses a maximum **squared center shift** threshold of `1e-6`. Global K-Means uses an **absolute change in consecutive-iteration SSE** threshold of `1e-6`. Both stages allow at most 100 iterations. Although the tolerances share a numerical value, they are applied to different quantities and should not be interpreted as identical convergence tests.

### 2.5 Implementation Mapping

| Stage | Primary source file |
| --- | --- |
| Local K-Means, distance calculations, initialization | `src/parallel/local_kmeans.py` |
| Partition-level execution | `src/parallel/partition_clustering.py` |
| Candidate-center aggregation | `src/parallel/center_aggregation.py` |
| Distributed assignment and reduction | `src/parallel/global_iteration.py` |
| Global iteration loop and evaluation | `src/parallel/global_kmeans.py` |
| End-to-end algorithm orchestration | `src/parallel/parallel_kmeans.py` |
| Dataset generation and benchmark configuration | `src/experiments/synthetic_data.py`, `src/experiments/experiment_runner.py` |
| RDD persistence and benchmark lifecycle | `src/experiments/benchmark_runner.py` |

### 2.6 Implementation Constraints

The partition-level routine materializes each nonempty partition in its worker process before local clustering. This simplifies local K-Means but creates a memory constraint for large partitions. The aggregation stage also collects candidate centers to the driver, and the candidate centers are treated equally rather than being weighted by the number of original observations they represent. These choices are documented limitations, especially when extending the approach to imbalanced or substantially larger datasets.

The Spark operations used here—partition transformations, broadcast variables, keyed reductions, and RDD persistence—are documented in the [Apache Spark RDD Programming Guide](https://spark.apache.org/docs/latest/rdd-programming-guide.html).
