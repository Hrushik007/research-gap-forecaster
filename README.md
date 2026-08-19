# FOREGA

**Forecasting High-Impact Research Gaps via Temporal Topic Modeling and Evolving Knowledge Graphs**

FoReGa(Forecaster of Research Gaps) is an end-to-end research analytics pipeline that mines large-scale arXiv metadata to surface underexplored, high-potential research directions in AI/ML. It combines temporal topic modeling, multi-layer scholarly knowledge graphs, automated gap detection, and ML-based impact prediction — then validates its predictions against genuinely future (2025) literature.

> Companion paper included in this repo: [`paper`](./paper.pdf)

---

## Overview

Researchers and funding agencies increasingly struggle to identify promising but underexplored directions in fast-moving fields like AI/ML. FoReGa addresses this with a unified five-step pipeline:

1. **Data Collection & Preprocessing** — arXiv metadata (cs.AI, cs.LG, cs.CV, cs.CL, cs.NE; 2019–2024), enriched with Semantic Scholar citation data.
2. **Temporal Topic Modeling** — BERTopic over `all-MiniLM-L6-v2` sentence embeddings, with UMAP + HDBSCAN clustering and topic trend classification (growing / stable / declining).
3. **Knowledge Graph Construction** — a multi-layer, temporally-annotated graph (papers, authors, topics, keywords) built with NetworkX, including co-authorship, topic-similarity, and community structure (Louvain).
4. **Automated Gap Identification** — detects frequency-based, structural/interdisciplinary, citation-based, and "heat-curve" gaps, each scored on novelty, impact, feasibility, and timeliness.
5. **Predictive Impact Modeling & Temporal Validation** — ensemble Random Forest / Gradient Boosting regressors rank gaps by predicted future impact, validated against held-out 2025 papers.

## Key Results

- **300,000+ papers** processed across **68 topics**
- ROC–AUC ≈ **0.62** for discriminating gaps that go on to be "filled" in 2025
- **Precision@K more than 2× the random baseline** for top-ranked gaps
- Feature importance analysis shows topic size, author diversity, and graph degree as the strongest predictors of future impact

## Repository Structure

```
research-gap-forecaster/
├── pipeline.ipynb          
├── paper.pdf  # Full write-up in IEEE format
├── README.md
├── requirements.txt
└── LICENSE
```

## Pipeline Steps (Notebook Sections)

| Step | Section | Description |
|------|---------|-------------|
| 1 | Data Collection & Preprocessing | Load arXiv metadata, clean text, enrich with Semantic Scholar citations |
| 2 | Temporal Topic Modeling | BERTopic + UMAP + HDBSCAN, topics-over-time, trend/drift statistics |
| 3 | Knowledge Graph Construction | NetworkX multi-layer graph, centrality metrics, Louvain communities |
| 4 | Automated Gap Identification | Frequency, structural, citation, and heat-curve gap detection + scoring |
| 5 | Predictive Impact Modeling & Temporal Validation | Ensemble regressors, 2025 held-out evaluation |

## Getting Started

### Prerequisites

- Python 3.9+
- GPU recommended (the notebook is optimized for Kaggle GPU runtimes) for the embedding and BERTopic steps

### Installation

```bash
git clone https://github.com/Hrushik007/research-gap-forecaster.git
cd VIZORA
pip install -r requirements.txt
```

### Running the Pipeline

Open `pipeline.ipynb` in Jupyter, Kaggle, or Colab and run cells sequentially; each step depends on artifacts (dataframes, embeddings, graph objects) produced by the previous one.

```bash
jupyter notebook pipeline.ipynb
```

Core dependencies installed within the notebook:

```
bertopic
sentence-transformers
umap-learn
hdbscan
python-louvain
pandas
numpy
matplotlib
seaborn
scipy
tqdm
networkx
scikit-learn
```

## Data Sources

- [arXiv](https://arxiv.org/) — open access metadata (via the Kaggle arXiv snapshot)
- [Semantic Scholar](https://www.semanticscholar.org/) — citation and reference data
- [OpenAlex](https://openalex.org/) — supplementary scholarly entity/venue data

## Authors

- Hrushik M Hegde

Dept. of CSE (AIML), PES University, Bengaluru, India

## Citation

If you use this work, please cite:

```bibtex
@techreport{research-gap-forecaster,
  title  = {Forecasting High-Impact Research Gaps via Temporal Topic Modeling and Evolving Knowledge Graphs},
  author = {Hegde, Hrushik M and Somangali, Harshini and Jain, Harshil K},
  institution = {PES University, Bengaluru},
  year   = {2026}
}
```

## Acknowledgments

The authors thank Dr. Bhaskarjyotidas and the teaching assistants of the Advanced Data Analytics course at PES University, Bengaluru, for their guidance and support. This work makes use of open scholarly data from arXiv, Semantic Scholar, and OpenAlex.

## License

This project is released under the [MIT License](./LICENSE). See the `LICENSE` file for details.
