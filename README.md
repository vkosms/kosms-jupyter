# kosms-jupyter

Jupyter notebooks exploring machine-learning techniques on Reddit-style data — subreddit embeddings, nearest-neighbor search, clustering, and a personalized feed ranker.

## Notebooks

### `t-Sne.ipynb` — Subreddit map & clustering
Builds a 2D map of subreddits from pairwise user overlap data.

1. Loads `subreddit-overlap` (pairs of subreddits with shared-user counts) into a sparse co-occurrence matrix.
2. Row-normalizes it into a conditional-probability matrix.
3. Reduces dimensionality with `TruncatedSVD` (500 components), then projects the top 10,000 subreddits to 2D with `sklearn.manifold.TSNE` (Barnes-Hut, perplexity 50).
4. Clusters the 2D map with `hdbscan` (`min_samples=5`, `min_cluster_size=20`).
5. Joins cluster IDs back into `subreddit-attrib.csv` and writes `subreddit-attrib-with-clusters.csv`.

Each HDBSCAN cluster represents a topical community of related subreddits.

### `faiss_rank.ipynb` — Nearest-neighbor subreddit search
Given pre-computed PMI (pointwise mutual information) vectors for subreddits, returns the most similar subreddits to a query.

1. Loads `subredditvecspmi.pkl.gz` (a pickled DataFrame of PMI vectors, not included in the repo — must be generated separately).
2. L2-normalizes each row so inner product equals cosine similarity.
3. Builds a `faiss.IndexFlatIP` index and runs a top-*k* search.

Example query for `"Economics"` returns neighbors like `economy`, `TrueReddit`, `NeutralPolitics`, `business`, `Futurology`, `BasicIncome`, etc.

### `LightGBM.ipynb` — Personalized feed ranker
End-to-end pipeline for ranking unseen submissions for a single user, backed by a Postgres database.

1. **Connect** — opens an `asyncpg` pool against a Postgres instance with `interactions`, `submissions`, and `groups` tables.
2. **User profile** — aggregates a user's interactions (upvotes, comments, views, title-clicks, recency) per group.
3. **Candidate set** — pulls submissions from the last 72 hours that the user hasn't seen and aren't NSFW.
4. **Training labels** — marks seen submissions as positive if the user upvoted / commented / clicked the title.
5. **Features** — normalized score, normalized comments, submission freshness (exponential decay over hours), user engagement ratio per group, group recency decay, and an `is_known_group` flag.
6. **Model** — `LGBMClassifier` (100 trees, lr 0.05, depth 6) with `scale_pos_weight` to handle class imbalance. Reports ROC-AUC and a classification report.
7. **Ranking** — scores candidates with `predict_proba`, then applies a diversity penalty (`0.5 ** (group_rank - 1)`) so the feed isn't dominated by one group, and a tunable freshness boost.

Tune `DIVERSITY_PENALTY` and `FRESHNESS_WEIGHT` at the bottom to change the feed's feel.

## Data files

| File | Description |
| --- | --- |
| `subreddit-overlap` | Pairwise subreddit co-occurrence counts (`t1_subreddit`, `t2_subreddit`, `NumOverlaps`). Input to `t-Sne.ipynb`. |
| `subreddit-attrib.csv` | Per-subreddit attributes. Input to `t-Sne.ipynb`. |
| `subreddit-attrib-with-clusters.csv` | `subreddit-attrib.csv` with a `cluster_id` column added — output of `t-Sne.ipynb`. |
| `subreddit-mapping.csv` | Subreddit name mapping / lookup table. |

`faiss_rank.ipynb` additionally requires `subredditvecspmi.pkl.gz` (pickled PMI vectors), which is not committed.

## Requirements

```bash
pip install pandas numpy scipy scikit-learn lightgbm faiss-cpu hdbscan asyncpg
```

- `LightGBM.ipynb` expects a running Postgres database with `interactions`, `submissions`, and `groups` tables — update `DB_CONFIG` in Cell 2.
- Notebooks were last run on Python 3.13.

## Usage

```bash
jupyter lab
```

Open any of the three notebooks and run top-to-bottom. The t-SNE notebook is the starting point if you want to regenerate clusters; the other two are independent of it.
