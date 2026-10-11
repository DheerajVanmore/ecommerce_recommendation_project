# Engineering a Hybrid E-Commerce Recommendation Engine with Python

**Authors:** Dheeraj Vanmore and Sahil Gawade
**Department of Computer Engineering, Vidyalankar Institute of Technology, Mumbai**  
**Course:** Big Data Analytics (BE Sem-7)

Online stores contain large product catalogs and generate a steady stream of product views, purchases, ratings, and searches. Helping customers discover relevant items is therefore both a recommendation problem and a data-processing challenge. In this article, we build a hybrid product recommendation prototype that combines text similarity with popularity signals, explain the mathematics behind it, and discuss how the design could evolve for larger datasets.

> **Project scope:** The implementation described here is a local prototype using a small illustrative product dataset. The distributed Apache Spark architecture is a proposed scaling direction, not a component implemented or benchmarked by this prototype.

## 1. The problem: relevance versus popularity

A store that ranks products only by sales can repeatedly show the same bestsellers. That may be useful for discovering popular items, but it can hide relevant alternatives and niche products. On the other hand, a system that considers only product descriptions may recommend items that sound similar but have little evidence of customer interest.

Recommendation systems commonly face three related challenges:

- **Cold start:** New products have little or no interaction history, making behavior-based recommendations difficult.
- **Popularity bias:** Frequently purchased products can dominate rankings, reducing exposure for less popular items.
- **Scalability:** Comparing every product with every other product becomes expensive as a catalog grows.

This project explores a simple hybrid approach: recommend items similar in their metadata while also considering purchase counts and ratings.

## 2. System overview

The pipeline has five stages:

1. **Load data:** Read a CSV product catalog.
2. **Preprocess fields:** Handle missing descriptions and invalid numeric values.
3. **Extract text features:** Combine each product's name, category, brand, and description.
4. **Calculate scores:** Use TF-IDF and cosine similarity for content relevance, then combine it with a popularity score.
5. **Rank candidates:** Exclude the query product and return the highest-scoring alternatives.

```text
Product CSV
    |
    v
Data cleaning and text preparation
    |
    +-------------------------------+
    |                               |
    v                               v
TF-IDF feature vectors       Purchases and ratings
    |                               |
    v                               v
Cosine similarity            Normalized popularity score
    |                               |
    +---------------+---------------+
                    |
                    v
             Hybrid ranking
                    |
                    v
          Top-N recommendations
```

## 3. How the recommendation score works

### 3.1 Representing product text with TF-IDF

The system combines four fields into a text document for each product:

`product name + category + brand + description`

For example, a description containing terms such as “wireless,” “Bluetooth,” “noise cancellation,” and “battery life” may be similar to descriptions of other audio products.

Term Frequency–Inverse Document Frequency (TF-IDF) converts text into numerical vectors. Term frequency reflects how often a term occurs in a document, while inverse document frequency gives less weight to terms that occur in many documents. The implementation uses unigrams and bigrams, allowing it to represent both individual words and phrases such as “noise cancellation.”

TF-IDF is a lexical technique: it can identify overlapping terms, but it does not truly understand meaning. Two products described with different words may be related even if their TF-IDF similarity is low.

### 3.2 Comparing products with cosine similarity

After converting product descriptions into TF-IDF vectors, the system calculates cosine similarity between them:

\[
\operatorname{sim}(A,B)=\frac{A\cdot B}{\|A\|_2\|B\|_2}
\]

For non-negative TF-IDF vectors, cosine similarity is typically between 0 and 1. A larger value indicates greater overlap in the represented text features. The prototype uses this score to find products whose metadata resembles the selected product.

### 3.3 Building a popularity score

The prototype combines purchase counts and ratings. Purchase counts are first transformed using a logarithm:

\[
P_i'=\ln(1+\text{purchases}_i)
\]

Logarithmic scaling reduces the effect of very large counts. The transformed purchase values and ratings are then min-max normalized to the range \([0,1]\).

The popularity score is:

\[
P_i=0.70\,\widehat{P_i'}+0.30\,\widehat{R_i}
\]

Here, the hat indicates min-max normalization, and \(R_i\) is the product rating. The 70/30 weighting is a configurable demonstration choice, not a value learned from real customer behavior.

### 3.4 Combining relevance and popularity

For query product \(q\) and candidate product \(c\), the hybrid score is:

\[
H(q,c)=\alpha\,S(q,c)+(1-\alpha)\,P_c
\]

where \(S(q,c)\) is content similarity, \(P_c\) is the candidate's popularity score, and \(\alpha\) controls the balance. The prototype defaults to \(\alpha=0.70\), giving content similarity a 70% weight and popularity a 30% weight.

- When \(\alpha=1\), ranking uses content similarity only.
- When \(\alpha=0\), ranking uses the popularity score only.
- Intermediate values combine both signals.

The selected product is excluded from the result list so that it is not recommended to itself.

## 4. Implementation with Python

The project uses Python, pandas, NumPy, and scikit-learn. The accompanying repository includes the script, a sample CSV dataset, a dependency file, and setup instructions.

Install the required packages from the project directory:

```bash
python -m pip install -r requirements.txt
```

The core feature-extraction logic is:

```python
products["content"] = (
    products["product_name"].fillna("") + " " +
    products["category"].fillna("") + " " +
    products["brand"].fillna("") + " " +
    products["description"].fillna("")
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)
tfidf_matrix = vectorizer.fit_transform(products["content"])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

The complete program also cleans numeric fields, computes the popularity score, combines scores, and ranks candidate products. Run it from the project root:

```bash
python recommendation.py
```

To request recommendations for a different product:

```bash
python recommendation.py --product "Gaming Headset" --top 5
```

To increase the influence of content similarity:

```bash
python recommendation.py --product "Wireless Headphones" --similarity-weight 0.8
```

The similarity weight must be between 0 and 1. The included catalog is intentionally small so that the project can run on a standard laptop without a GPU, cloud service, or distributed cluster.

## 5. Results and interpretation

When executed against the included sample catalog, the program returns a ranked table containing product names, categories, ratings, purchase counts, content similarity, and the final recommendation score.

The ranking is determined by both the selected product's text similarity and each candidate's popularity score. Changing the similarity weight can change the order of recommendations. This makes the trade-off visible and easy to experiment with.

The sample data is illustrative rather than real commercial data. The prototype demonstrates that the pipeline runs and produces recommendations, but it does not establish that the system improves sales, customer satisfaction, or recommendation accuracy. No claims about real-world performance should be inferred from this small example.

A meaningful evaluation would use a documented real-world dataset and a clear test protocol. Depending on the task, useful metrics could include Precision@K and Recall@K for relevance, catalog coverage for how much of the inventory is recommended, and intra-list diversity for the variety within each result list. These metrics should be calculated from reproducible experiments rather than estimated.

## 6. Scaling the design for Big Data

The current implementation calculates pairwise similarity on a single machine. A dense \(N \times N\) matrix grows quadratically with the number of products. For float32 scores, a matrix for one million products alone would require roughly 4 terabytes of decimal storage, before accounting for other memory overhead. That approach is therefore unsuitable for very large catalogs.

A larger production system could use a distributed pipeline:

1. Ingest product catalog updates and interaction events through a suitable data pipeline.
2. Use distributed processing, such as Apache Spark, for large-scale data preparation and feature computation when justified by workload size.
3. Retrieve a smaller set of candidate products using an appropriate search index or approximate-nearest-neighbor method.
4. Apply the hybrid ranking formula to those candidates.
5. Monitor latency, relevance, coverage, and data quality as the catalog changes.

Apache Spark MLlib offers tools for distributed feature processing and collaborative filtering, including Alternating Least Squares (ALS) for user-item interaction data. For text-based item retrieval, Spark's feature tools or a separate vector-search system may be appropriate depending on the architecture. These are possible extensions; they are not implemented by the local Python prototype.

Approximate-nearest-neighbor methods can reduce retrieval work, but their performance and accuracy depend on the index, data, and configuration. They should be measured rather than assumed.

## 7. Practical limitations and future improvements

The prototype has several limitations:

- **Not personalized:** It recommends products related to a selected product, not necessarily to an individual customer's preferences.
- **Lexical matching:** TF-IDF may miss synonyms and semantic relationships.
- **Demonstration data:** The small synthetic dataset cannot validate commercial performance.
- **Fixed weights:** The scoring weights are manually chosen and should be evaluated for a real use case.
- **Single-machine similarity:** A full pairwise matrix becomes impractical for a large catalog.

Future versions could add collaborative filtering from user-item interactions, semantic embeddings, time-aware popularity, and offline ranking evaluation. A production deployment would also need reliable data ingestion, privacy controls, monitoring, and an update strategy for new or changed products.

## 8. Conclusion

This project demonstrates a transparent hybrid recommendation pipeline using TF-IDF, cosine similarity, and popularity signals. The approach balances metadata relevance with product engagement indicators and provides a practical starting point for understanding recommendation systems through Python.

The implementation is intentionally small and reproducible. Its main contribution is an understandable baseline that can be tested and extended, not a claim of production-scale performance. With real interaction data, rigorous evaluation, and an appropriate retrieval architecture, the same core idea could serve as a foundation for more advanced e-commerce recommendation systems.

## References

1. G. Linden, B. Smith, and J. York, “Amazon.com Recommendations: Item-to-Item Collaborative Filtering,” *IEEE Internet Computing*, vol. 7, no. 1, pp. 76–80, 2003.
2. J. Leskovec, A. Rajaraman, and J. D. Ullman, *Mining of Massive Datasets*, 3rd ed., Cambridge University Press, 2020.
3. Scikit-learn, “TfidfVectorizer Documentation.” https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
4. Scikit-learn, “Cosine Similarity Documentation.” https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
5. Apache Spark, “Spark MLlib Guide.” https://spark.apache.org/docs/latest/ml-guide.html

## Authors and contributions

**Dheeraj Vanmore:** Contributed to the design and implementation of the hybrid e-commerce recommendation engine, including TF-IDF-based product similarity, cosine similarity, popularity scoring, and hybrid recommendation ranking. Also contributed to the technical documentation and preparation of the blog post.

**Sahil Gawade:** Contributed to the research and analysis of e-commerce recommendation systems, reviewed the recommendation methodology, and assisted with technical documentation and review of the blog post.

*Prepared for Assignment 10: Big Data Analytics, BE Semester 7, Department of Computer Engineering, Vidyalankar Institute of Technology, Mumbai.*
