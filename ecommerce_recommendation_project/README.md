# Scalable Hybrid E-Commerce Product Recommendation System
### BE Sem-7 Big Data Analytics (BDA) — Assignment No-10
**Vidyalankar Institute of Technology (VIT), Mumbai**  
**Department of Computer Engineering**  
*Course Outcome Mapping: Module 1–6 | CO5 (Create) | PO3, PO9, PO10, PO12*

---

## 📌 Self-Declaration
> **Self-Declaration:** It took me **4** hour(s) and **30** minute(s) to solve the assignment, ensuring accuracy, completeness, and adherence to professional and ethical standards.

---

## 👥 Authors & Contribution Statement

| Member Name | Roll No. / Seat No. | Role & Contribution |
| :--- | :--- | :--- |
| **[Student 1 Name]** | [Roll No. 1] | Algorithmic formulation, hybrid pipeline implementation (`recommendation.py`), evaluation harness (`evaluate_metrics.py`), technical blog writing. |
| **[Student 2 Name]** | [Roll No. 2] | Big Data scalability & distributed architecture analysis (`big_data_pipeline.py`), literature review, IEEE technical paper formatting (`IEEE_TECHNICAL_PAPER.md` / `ieee_paper.tex`), repository benchmarking. |

---

## 🚀 Project Overview

This repository hosts a production-grade **Hybrid E-Commerce Product Recommendation Engine** developed for Big Data Analytics (BDA). The system addresses core challenges in large-scale online retail:
1. **Cold-Start Latency:** New items lack historical customer transactions. We mitigate this using content-based TF-IDF vectorization across titles, categories, brands, and descriptions.
2. **Popularity Bias / Filter Bubbles:** Relying solely on sales causes top items to crowd out recommendations. We apply sublinear logarithmic scaling to ensure long-tail inventory surfaces appropriately.
3. **Big Data Scalability ($O(N^2)$ Memory Bottleneck):** Pairwise dense similarity computation hits memory walls as catalogs expand to millions of items. We formalize this constraint and design an Apache Spark MLlib distributed pipeline with Approximate Nearest Neighbor (ANN) vector indexing.

---

## 📁 Repository Structure

```
ecommerce_recommendation_project/
├── data/
│   └── products.csv             # Product catalog dataset with metadata, ratings, and purchase signals
├── recommendation.py            # Core hybrid recommendation engine (CLI with configurable parameters)
├── evaluate_metrics.py          # Quantitative evaluation benchmark suite (Precision@K, Coverage, Diversity, Latency)
├── big_data_pipeline.py         # Big Data clickstream interaction simulator, sparsity analysis & Spark scaling benchmark
├── sample_output.txt            # Real execution logs across diverse product queries and parameter settings
├── evaluation_results.txt       # Detailed benchmark metrics and parameter sensitivity results
├── BLOG_POST.md                 # 1,765-word publication-ready technical blog post (Medium/Hashnode/LinkedIn)
├── IEEE_TECHNICAL_PAPER.md      # 4-page camera-ready technical paper in IEEE conference format
├── ieee_paper.tex               # IEEE LaTeX conference template ready for Overleaf compilation
├── requirements.txt             # Project dependencies (pandas, numpy, scikit-learn, scipy)
└── README.md                    # Comprehensive project documentation and academic metadata
```

---

## 🔬 Mathematical Formulation

### 1. TF-IDF Textual Feature Extraction
$$\text{Corpus}_i = \text{Title}_i \oplus \text{Category}_i \oplus \text{Brand}_i \oplus \text{Description}_i$$

$$\text{TF-IDF}(t, d_i, D) = \left(\frac{f_{t, d_i}}{\sum_{t'} f_{t', d_i}}\right) \times \left(\log \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} + 1\right)$$

Extracts unigrams and bigrams ($n \in \{1, 2\}$) with stop-word removal.

### 2. Geometric Content Proximity (Cosine Similarity)
$$\text{Sim}_{\text{content}}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} \in [0, 1]$$

### 3. Sublinear Popularity Dynamics
$$\tilde{P}_i = \ln(1 + \text{purchases}_i)$$

$$\text{Score}_{\text{pop}}(i) = 0.70 \cdot \text{MinMax}(\tilde{P}_i) + 0.30 \cdot \text{MinMax}(\text{rating}_i)$$

### 4. Parameterized Hybrid Ranking
$$\text{Score}_{\text{final}}(q, c) = \alpha \cdot \text{Sim}_{\text{content}}(q, c) + (1 - \alpha) \cdot \text{Score}_{\text{pop}}(c)$$

- $\alpha = 0.70$ provides the default balanced recommendation setting.

---

## 📊 Empirical Evaluation & Benchmark Results

Run using `python evaluate_metrics.py` (Top-$K = 5$):

| Similarity Weight ($\alpha$) | Category Precision@5 | Catalog Coverage (%) | Intra-List Diversity | Unique Products Surfaced |
| :---: | :---: | :---: | :---: | :---: |
| **0.1** (Popularity Biased) | 24.0% | 30.0% | 0.982 | 6 / 20 |
| **0.3** | 25.0% | 35.0% | 0.983 | 7 / 20 |
| **0.5** (Balanced) | 27.0% | 50.0% | 0.982 | 10 / 20 |
| **0.7** (Default Hybrid) | 27.0% | 60.0% | 0.982 | 12 / 20 |
| **0.9** (Content Biased) | 33.0% | 100.0% | 0.982 | 20 / 20 |

- **Inference Speed:** 5.887 ms per item query (~170 recommendations/sec single CPU core).

---

## ⚡ Big Data Scalability Analysis

Run using `python big_data_pipeline.py`:

| Catalog Scale ($N$) | Dense Matrix Size (GB) | Single-Node RAM Feasibility | Recommended Big Data Architecture |
| :---: | :---: | :---: | :---: |
| **100** | < 1 MB | Feasible | Single-Node Pandas |
| **10,000** | 0.37 GB | Feasible | Single-Node Pandas |
| **100,000** | 37.25 GB | **Out of Memory (OOM)** | Apache Spark MLlib RDD |
| **1,000,000** | 3.60 Terabytes | **Out of Memory (OOM)** | Distributed Spark Cluster + HDFS |
| **10,000,000** | 363.80 Terabytes | **Out of Memory (OOM)** | Distributed Spark ALS + Vector DB |

---

## 💻 Setup & Execution Guide

### Prerequisites
- Python 3.10+ installed
- Windows PowerShell, Command Prompt, or Linux/macOS Terminal

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Core Recommender
Default query (`Wireless Headphones`, Top 5):
```bash
python recommendation.py
```

Query a custom item with specific top-N items:
```bash
python recommendation.py --product "Gaming Headset" --top 5
```

Adjust the hybrid balance parameter $\alpha$ (e.g. 0.9 for pure content matching):
```bash
python recommendation.py --product "Wireless Headphones" --similarity-weight 0.9
```

### Step 3: Run Quantitative Evaluation Suite
```bash
python evaluate_metrics.py
```

### Step 4: Run Big Data Pipeline Simulation
```bash
python big_data_pipeline.py
```

---

## 🌐 Deliverables for BDA Assignment 10

1. **Technical Blog Post (1,200–1,800 words):**  
   Read [`BLOG_POST.md`](BLOG_POST.md) — formatted for immediate publishing to Medium, Hashnode, or LinkedIn.
2. **IEEE 4-Page Technical Paper:**  
   Read [`IEEE_TECHNICAL_PAPER.md`](IEEE_TECHNICAL_PAPER.md) or compile [`ieee_paper.tex`](ieee_paper.tex) using Overleaf.
3. **Public GitHub Repository:**  
   Contains all code, dataset, evaluation scripts, benchmarks, and clean outputs.

---

## 🔗 External Large-Scale Dataset Links
For evaluating on multi-gigabyte industrial datasets:
- [Amazon Review Data (UCSD / McAuley Lab)](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews)
- [Instacart Market Basket Analysis (Kaggle Big Data Dataset)](https://www.kaggle.com/c/instacart-market-basket-analysis)
- [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
