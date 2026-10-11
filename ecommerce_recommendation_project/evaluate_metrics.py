"""
Evaluation & Quantitative Benchmarking Suite for E-Commerce Recommender
BE Sem-7 Big Data Analytics (BDA) Project
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from recommendation import build_recommender, recommend


def evaluate_system(data_path="data/products.csv", top_k=5, weights=(0.1, 0.3, 0.5, 0.7, 0.9)):
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"File not found: {data_file}")

    products, similarity = build_recommender(data_file)
    n_products = len(products)
    all_product_names = products["product_name"].tolist()

    print("=" * 70)
    print("  BIG DATA ANALYTICS (BDA) - RECOMMENDER SYSTEM BENCHMARK SUITE")
    print("=" * 70)
    print(f"Catalog Size: {n_products} items")
    print(f"Categories: {products['category'].nunique()} distinct classes: {list(products['category'].unique())}")
    print(f"Top-K Cutoff: K = {top_k}")
    print("-" * 70)

    # 1. Latency & Execution Benchmark
    t0 = time.perf_counter()
    for name in all_product_names:
        _ = recommend(name, products, similarity, top_n=top_k, similarity_weight=0.7)
    total_latency_ms = (time.perf_counter() - t0) * 1000
    avg_latency_ms = total_latency_ms / n_products

    print(f"Latency Benchmark (Single-Node In-Memory):")
    print(f"  Total evaluation time ({n_products} queries): {total_latency_ms:.2f} ms")
    print(f"  Mean latency per item query: {avg_latency_ms:.3f} ms")
    print("-" * 70)

    # 2. Sensitivity Analysis across alpha (similarity_weight)
    results_table = []

    for alpha in weights:
        all_recommended_ids = set()
        category_hit_count = 0
        total_evaluations = 0
        diversity_scores = []

        for name in all_product_names:
            source_row = products[products["product_name"] == name].iloc[0]
            source_cat = source_row["category"]
            source_idx = source_row.name

            recs = recommend(name, products, similarity, top_n=top_k, similarity_weight=alpha)
            rec_ids = set(recs["product_id"])
            all_recommended_ids.update(rec_ids)

            # Category relevance: proportion of top-K belonging to same category
            same_cat_hits = (recs["category"] == source_cat).sum()
            category_hit_count += same_cat_hits
            total_evaluations += top_k

            # Intra-list Diversity (1 - pairwise cosine similarity among recommendations)
            rec_indices = [products.index[products["product_id"] == pid][0] for pid in recs["product_id"]]
            pairwise_sims = []
            for i in range(len(rec_indices)):
                for j in range(i + 1, len(rec_indices)):
                    pairwise_sims.append(similarity[rec_indices[i], rec_indices[j]])
            if pairwise_sims:
                diversity_scores.append(1.0 - float(np.mean(pairwise_sims)))

        # Catalog Coverage
        catalog_coverage_pct = (len(all_recommended_ids) / n_products) * 100.0
        # Precision@K by category match
        cat_precision = (category_hit_count / total_evaluations) * 100.0
        # Mean Intra-List Diversity
        mean_diversity = float(np.mean(diversity_scores)) if diversity_scores else 0.0

        results_table.append({
            "Similarity Weight (alpha)": f"{alpha:.1f}",
            "Category Precision@K (%)": f"{cat_precision:.1f}%",
            "Catalog Coverage (%)": f"{catalog_coverage_pct:.1f}%",
            "Intra-List Diversity": f"{mean_diversity:.3f}",
            "Unique Items Rec'd": len(all_recommended_ids)
        })

    df_results = pd.DataFrame(results_table)
    print("\nWeight Sensitivity Analysis (Content Similarity vs Popularity Blend):")
    print(df_results.to_string(index=False))
    print("=" * 70)

    return df_results


if __name__ == "__main__":
    evaluate_system()
