from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def min_max_scale(values):
    """Scale numeric values to [0, 1], handling constant columns safely."""
    values = np.asarray(values, dtype=float)
    minimum, maximum = values.min(), values.max()
    if maximum == minimum:
        return np.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def build_recommender(csv_path):
    products = pd.read_csv(csv_path)

    required = {
        "product_id", "product_name", "category", "brand",
        "description", "rating", "purchases"
    }
    missing = required - set(products.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    products = products.dropna(subset=["product_id", "product_name"]).copy()
    products["description"] = products["description"].fillna("")
    products["category"] = products["category"].fillna("")
    products["brand"] = products["brand"].fillna("")
    products["rating"] = pd.to_numeric(products["rating"], errors="coerce").fillna(0)
    products["purchases"] = pd.to_numeric(products["purchases"], errors="coerce").fillna(0)

    # Combine fields so the text representation captures more than product names.
    products["content"] = (
        products["product_name"] + " " +
        products["category"] + " " +
        products["brand"] + " " +
        products["description"]
    )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(products["content"])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Log scaling prevents very popular products from dominating too strongly.
    popularity = np.log1p(products["purchases"].clip(lower=0))
    popularity_score = min_max_scale(popularity)
    rating_score = min_max_scale(products["rating"])

    # A transparent baseline quality/popularity score.
    products["popularity_score"] = 0.7 * popularity_score + 0.3 * rating_score
    return products.reset_index(drop=True), similarity_matrix


def recommend(product_name, products, similarity_matrix, top_n=5, similarity_weight=0.7):
    matches = products.index[
        products["product_name"].str.casefold() == product_name.casefold()
    ].tolist()
    if not matches:
        available = ", ".join(products["product_name"].head(10).tolist())
        raise ValueError(
            f"Product '{product_name}' not found. Examples in dataset: {available}"
        )

    source_idx = matches[0]
    similarity_scores = similarity_matrix[source_idx]
    candidates = products.index[products.index != source_idx].to_numpy()

    # Blend content similarity with product popularity/ratings.
    final_scores = (
        similarity_weight * similarity_scores[candidates] +
        (1 - similarity_weight) * products.loc[candidates, "popularity_score"].to_numpy()
    )
    ranked = candidates[np.argsort(final_scores)[::-1]][:top_n]

    result = products.loc[ranked, [
        "product_id", "product_name", "category", "brand", "price",
        "rating", "purchases"
    ]].copy()
    result["content_similarity"] = similarity_scores[ranked]
    result["recommendation_score"] = final_scores[
        np.argsort(final_scores)[::-1][:top_n]
    ]
    return result.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid e-commerce product recommender (content + popularity)."
    )
    parser.add_argument("--data", default="data/products.csv", help="Path to products CSV")
    parser.add_argument("--product", default="Wireless Headphones", help="Product to find similar items for")
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--similarity-weight", type=float, default=0.7,
                        help="Weight for content similarity, between 0 and 1")
    args = parser.parse_args()

    if args.top < 1:
        parser.error("--top must be at least 1")
    if not 0 <= args.similarity_weight <= 1:
        parser.error("--similarity-weight must be between 0 and 1")

    data_path = Path(args.data)
    if not data_path.exists():
        parser.error(f"Dataset not found: {data_path}. Run from the project root or pass --data.")

    products, similarity = build_recommender(data_path)
    try:
        result = recommend(args.product, products, similarity, args.top, args.similarity_weight)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"\nProducts similar to: {args.product}\n")
    print(result.round({"content_similarity": 3, "recommendation_score": 3}).to_string(index=False))
    print("\nNote: this is a demonstration using a small illustrative dataset, not real customer data.")


if __name__ == "__main__":
    main()
