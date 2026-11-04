from src.experiments.clustering_quality_report import (
    load_quality_comparison_csv,
    summarize_quality_comparisons,
)

rows = load_quality_comparison_csv("results/verified_quality/clustering_quality_comparison.csv")

for s in summarize_quality_comparisons(rows):
    print(f"=== {s.dataset_name} ===")
    print(f"  Spark {s.spark_base_master}:      {s.spark_base_seconds:.4f}s")
    if s.speedup:
        print(f"  Spark {s.spark_comparison_master}:      {s.spark_comparison_master_seconds:.4f}s  (speedup {s.speedup:.2f}x)")
    print(f"  Traditional:          {s.traditional_seconds:.4f}s")
    print()
    print(f"  {'Metric':<10} {'Spark':>10} {'Traditional':>12} {'Gap':>10}")
    print(f"  {'purity':<10} {s.spark_purity:>10.4f} {s.traditional_purity:>12.4f} {s.purity_gap:>10.4f}")
    print(f"  {'ARI':<10} {s.spark_adjusted_rand_index:>10.4f} {s.traditional_adjusted_rand_index:>12.4f} {s.adjusted_rand_index_gap:>10.4f}")
    print(f"  {'NMI':<10} {s.spark_normalized_mutual_information:>10.4f} {s.traditional_normalized_mutual_information:>12.4f} {s.normalized_mutual_information_gap:>10.4f}")
    print()