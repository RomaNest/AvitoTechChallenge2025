# AvitoTech ML Cup 2025 — 7th Place Solution

This is the 7th-place solution for the AvitoTech ML Cup 2025: [https://ods.ai/competitions/avitotechmlchallenge2025](https://ods.ai/competitions/avitotechmlchallenge2025).

The approach revolves around simple baseline models for candidate retrieval and a GBM model for reranking.

## Retrieval Models

* `als.ipynb` — ALS model, same as in the baseline provided by the organisers.
* `co-occurence.ipynb` — item-to-item heuristic with penalties for:

  1. how far apart interactions are,
  2. the time between interactions,
  3. how recent the interaction is,
  4. how popular the item is (for debiasing).
* `title-features.ipynb` — retrieves items with the smallest distance to the centroid of past user interactions and to the last user interaction.
* `tags-similarity.ipynb` — treats different `{"item": "value"}` pairs as unique tags, hashes them, and finds closest items via sparse tag vectors.
* `gnn.py` — simple LightGCN model with different weights by interaction type and recency, with popularity debiasing.
* `transformer.ipynb` — small, CPU-friendly transformer model.
* `utils.py` — contains popularity-based heuristics in the `create_features()` function.

## Final Reranking Model Example

* `lgb-submission.ipynb` — notebook with dataset creation, ranker fitting, and final prediction.

## Feature Dataset for Reranking

All feature logic is in `utils.py` → `create_features()` function.
The dataset is formed by fully joining all retrieval methods. Features include:

1. Ranks or scores from each model.
2. If embeddings are present, dot products for each generated pair.
3. User features (counts and preferences).
4. Item features (popularities and categories).

## Training & Blending

For training, I use negative subsampling (from 2% to 5% zeros are untouched).
For the final private leaderboard submission, I blended multiple GBM ranker predictions from CatBoost and LightGBM models with different `SUBSAMPLE_RATIO`, `SEED`, and loss functions.
The best solo GBM model (CatBoost YetiRank) scores **0.2064 Recall\@40** on the public leaderboard; blending reaches **0.2105 Recall\@40**.

## Differences from the Best Solutions

Winning teams managed to build at least one strong solo retrieval model with a good architecture, while my approach focused on assembling diverse signals via simple retrieval baselines.
