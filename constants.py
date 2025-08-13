from pathlib import Path
import polars as pl

def read_any(path: Path) -> pl.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pl.read_csv(path)
    if suf in (".pq", ".parquet"):
        return pl.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path}")

FILES = {
    "als_all":              "{d}d-back-als-all.csv",
    "als_contact":          "{d}d-back-als-contact.csv",
    "als_user_emb":         "als_user_emb_{d}d.pq",
    "als_item_emb":         "als_item_emb_{d}d.pq",
    "als_17_user_emb":      "als_17_user_emb_{d}d.pq",
    "als_17_item_emb":      "als_17_item_emb_{d}d.pq",
    "i2i_seq_all":          "i2i_17_{d}d.pq",
    "i2i_seq_contact":      "i2i_11_{d}d.pq",
    "i2i_df":               "i2i_{d}d.pq",
    "avg_text_node":        "avg_text_node_{d}d.parquet",
    "avg_text_cookie":      "cluster_text_cookie_{d}d.parquet",
    "last_text_cookie":     "last_item_text_cookie_{d}d.parquet",
    "topk_avg_text_cookie": "cluster_text_pred_{d}d.csv",
    "topk_last_text_cookie":"last_item_text_pred_{d}d.csv",
    "graph_pred":           "top300_graph_emb_{d}d.pq",
    "graph_item_emb":       "item_graph_emb_{d}d.pq",
    "graph_user_emb":       "user_graph_emb_{d}d.pq",
    "tag_cosine":           "top300_tag_cosine1_{d}d.pq",
    "tag_emb_cookie":       "tag_cosine_cookie_{d}d.pq",
    "tag_emb_node":         "tag_cosine_node_{d}d.pq",
    "transformer":          "transformer2_{d}d.parquet",
}

ORDER = [
    "als_all",
    "als_user_emb",
    "als_item_emb",
    "als_contact",
    "als_17_user_emb",
    "als_17_item_emb",
    "i2i_seq_all",
    "i2i_seq_contact",
    "avg_text_node",
    "avg_text_cookie",
    "last_text_cookie",
    "topk_avg_text_cookie",
    "topk_last_text_cookie",
    "graph_pred",
    "graph_item_emb",
    "graph_user_emb",
    "i2i_df",
    "tag_cosine",
    "tag_emb_cookie",
    "tag_emb_node",
    "transformer",
]

# Load everything
dfs = {
    key: read_any(DATA_DIR / tpl.format(d=DAYS))
    for key, tpl in FILES.items()
}

# Build the arg list in the exact order create_features expects
feature_args = [dfs[name] for name in ORDER]

# Call as before
df_14 = create_features(
    df_events_14,
    df_targets_14,
    *feature_args,
    subsample=SUBSAMPLE
)
