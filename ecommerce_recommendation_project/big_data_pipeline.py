"""
Big Data Scalability & Distributed Architecture Pipeline Simulation
BE Sem-7 Big Data Analytics (BDA) Project
Simulates big data clickstream interactions, matrix sparsity, SVD/ALS matrix factorization,
and demonstrates distributed scaling vs single-node constraints.
"""

import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def generate_interaction_log(n_users=500, n_products=20, n_interactions=3000, seed=42):
    """Generates synthetic high-volume user-item click/purchase logs for big data analytics."""
    np.random.seed(seed)
    user_ids = [f"U{i:04d}" for i in range(1, n_users + 1)]
    product_ids = [f"P{i:03d}" for i in range(1, n_products + 1)]

    # Simulate Power-Law / Zipfian popularity distribution common in e-commerce
    weights = 1.0 / np.arange(1, n_products + 1) ** 0.8
    prob_products = weights / weights.sum()

    users_sampled = np.random.choice(user_ids, size=n_interactions, replace=True)
    products_sampled = np.random.choice(product_ids, size=n_interactions, p=prob_products, replace=True)
    ratings = np.random.choice([1, 2, 3, 4, 5], size=n_interactions, p=[0.05, 0.10, 0.20, 0.35, 0.30])

    df_interactions = pd.DataFrame({
        "user_id": users_sampled,
        "product_id": products_sampled,
        "interaction_rating": ratings
    }).drop_duplicates(subset=["user_id", "product_id"])

    return df_interactions


def run_collaborative_filtering_svd(df_interactions, k_factors=5):
    """
    Simulates matrix factorization (collaborative filtering ALS / SVD baseline).
    Decomposes User-Item interaction matrix R ~ U * Sigma * V^T.
    """
    users = df_interactions["user_id"].unique()
    products = df_interactions["product_id"].unique()

    user_map = {u: i for i, u in enumerate(users)}
    prod_map = {p: i for i, p in enumerate(products)}

    row_ind = [user_map[u] for u in df_interactions["user_id"]]
    col_ind = [prod_map[p] for p in df_interactions["product_id"]]
    data = df_interactions["interaction_rating"].values

    # Build sparse CSR matrix
    R_sparse = csr_matrix((data, (row_ind, col_ind)), shape=(len(users), len(products)), dtype=float)
    total_elements = R_sparse.shape[0] * R_sparse.shape[1]
    nonzero_elements = R_sparse.nnz
    sparsity = (1.0 - (nonzero_elements / total_elements)) * 100.0

    # Truncated SVD (Matrix Factorization)
    k = min(k_factors, min(R_sparse.shape) - 1)
    u, s, vt = svds(R_sparse, k=k)
    s_diag = np.diag(s)
    predicted_ratings = np.dot(np.dot(u, s_diag), vt)

    return {
        "n_users": len(users),
        "n_products": len(products),
        "total_interactions": nonzero_elements,
        "sparsity_pct": sparsity,
        "predicted_matrix_shape": predicted_ratings.shape,
        "latent_factors": k
    }


def analyze_big_data_scalability():
    """
    Computes theoretical vs empirical scaling memory footprints
    comparing Single-Node In-Memory Dense matrix vs Distributed Big Data (PySpark / HDFS).
    """
    scales = [100, 10_000, 100_000, 1_000_000, 10_000_000]
    scaling_analysis = []

    for n in scales:
        # N x N pairwise float32 (4 bytes per cell)
        dense_bytes = (n * n * 4)
        dense_gb = dense_bytes / (1024 ** 3)
        # Sparse with 0.1% non-zeros
        sparse_mb = (n * 50 * 12) / (1024 ** 2)  # top-50 candidate index
        feasible_single_node = "Yes" if dense_gb < 16.0 else "OUT OF MEMORY (OOM)"
        distributed_engine = "Single Pandas/NumPy" if dense_gb < 16.0 else "Apache Spark RDD / GraphX"

        scaling_analysis.append({
            "Catalog Size (N)": f"{n:,}",
            "Dense Similarity Matrix (GB)": f"{dense_gb:.2f} GB" if dense_gb < 1000 else f"{dense_gb/1024:.1f} TB",
            "Single-Node RAM Feasibility": feasible_single_node,
            "Recommended Big Data Engine": distributed_engine
        })

    return pd.DataFrame(scaling_analysis)


def main():
    print("=" * 75)
    print("  BIG DATA ANALYTICS: INTERACTION PIPELINE & SCALABILITY ANALYSIS")
    print("=" * 75)

    # 1. Interaction log generation
    print("\n[Step 1] Generating High-Volume E-Commerce Interaction Stream...")
    interactions = generate_interaction_log(n_users=1000, n_products=20, n_interactions=5000)
    print(f"Generated {len(interactions):,} distinct user-item implicit/explicit interactions.")

    # 2. Matrix Factorization & Sparsity
    print("\n[Step 2] User-Item Matrix Decomposition (Collaborative Filtering SVD/ALS):")
    stats = run_collaborative_filtering_svd(interactions, k_factors=5)
    print(f"  User Count (|U|): {stats['n_users']}")
    print(f"  Product Count (|I|): {stats['n_products']}")
    print(f"  Active Interacted Cells: {stats['total_interactions']:,}")
    print(f"  Matrix Sparsity: {stats['sparsity_pct']:.2f}% (High Sparsity confirms Big Data characteristics)")
    print(f"  Latent Dimensionality (K): {stats['latent_factors']}")

    # 3. Big Data Scaling Analysis
    print("\n[Step 3] Scalability Bottleneck Analysis (Single-Node vs Apache Spark Distributed):")
    df_scaling = analyze_big_data_scalability()
    print(df_scaling.to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    main()
