import polars as pl
import numpy as np
from datetime import timedelta
from itertools import groupby
import lightgbm as lgb
from catboost import Pool, CatBoostRanker

def recall_at(df_true, df_pred, k=40):
    return  df_true[['node', 'cookie']].join(
        df_pred.group_by('cookie').head(k).with_columns(value=1)[['node', 'cookie', 'value']], 
        how='left',
        on = ['cookie', 'node']
    ).select(
        [pl.col('value').fill_null(0), 'cookie']
    ).group_by(
        'cookie'
    ).agg(
        [
            pl.col('value').sum()/pl.col(
                'value'
            ).count()
        ]
    )['value'].mean()

def create_features(df_train, 
                    df_eval, 
                    als_all,
                    als_user_emb,
                    als_item_emb,
                    als_contact,
                    als_17_user_emb,
                    als_17_item_emb,
                    i2i_pred_all,
                    i2i_pred_contact,
                    avg_text_node,
                    avg_text_cookie,
                    last_text_cookie,
                    topk_avg_text_cookie,
                    topk_last_text_cookie,
                    graph_pred,
                    graph_item_emb,
                    graph_user_emb,
                    i2i_df,
                    tag_cosine,
                    tag_emb_cookie,
                    tag_emb_node,
                    transformer,
                    subsample=None,
                    seed=13,
                    is_submit=False):
    distinct_cookies = df_eval.select(pl.col('cookie').unique())
    tag_cosine = tag_cosine.rename({'score':'tag_score'})

    decay_factor = 0.95
    max_date = df_train.select(pl.max("event_date")).item()

    # tag_cosine = top_n_per_cookie(tag_cosine, 50, 'tag_score')
    # graph_pred = graph_pred.filter(pl.col('rank') < 101)
    # transformer = transformer.filter(pl.col('rank') < 151)

    # als_all = top_n_per_cookie(als_all, 200, 'als-all')
    # als_contact = top_n_per_cookie(als_contact, 200, 'als-17')
    # i2i_pred_all = top_n_per_cookie(i2i_pred_all, 250, 'i2i_11_score')
    # i2i_pred_contact = top_n_per_cookie(i2i_pred_contact, 150, 'i2i_11_score')
    # i2i_df = top_n_per_cookie(i2i_df, 250, 'i2i_score')
    # topk_avg_text_cookie = topk_avg_text_cookie.filter(pl.col('rank') < 151)
    # topk_last_text_cookie = topk_last_text_cookie.filter(pl.col('rank_last_item') < 151)
    
    item_features = (
        df_train
        .with_columns(
            days_diff=(max_date - pl.col("event_date")).dt.total_days(),
            time_decay=pl.lit(decay_factor).pow(
                (max_date - pl.col("event_date")).dt.total_days()
            )
        )
        .with_columns(
            weighted_contact=pl.col("is_contact") * pl.col("time_decay")
        )
        .group_by("node")
        .agg(
            pl.sum("time_decay").alias("count_all"),
            pl.sum("weighted_contact").alias("contact_all")
        )
        .with_columns(
            contact_ratio=pl.col("contact_all") / pl.col("count_all")
        )
    )

    user_features = (
        df_train
        .filter(pl.col('cookie').is_in(distinct_cookies))
        .filter(pl.col('event_date') > max_date - timedelta(days=60))
        .group_by('cookie')
        .agg(
            pl.col('location').mode().first().alias('user_location'), # check correctness
            pl.col('category').mode().first().alias('user_category'),
            pl.col('is_contact').sum().alias('num_contacts'),
            pl.col('is_contact').count().alias('num_events'),
            pl.col('surface').unique().alias('surface_unique_counts'),
            pl.col('location').unique().alias('location_unique_counts')
        )
        .with_columns(
            [pl.col('surface_unique_counts').list.len(), pl.col('location_unique_counts').list.len()]
        )
    )
    # Helper function for time decay calculations
    def add_time_decay(df, max_date, decay_factor):
        return (
            df
            .with_columns(
                (max_date - pl.col("event_date")).dt.total_days().alias("days_diff")
            )
            .with_columns(
                pl.lit(decay_factor).pow(pl.col("days_diff")).alias("time_decay")
            )
        )
    
    # Updated popularity features with time decay
    def create_popularity_feature(base_df, filter_expr, group_cols, agg_col, rank_limit):
        return (
            base_df
            .filter(filter_expr)
            .pipe(add_time_decay, max_date, decay_factor)
            .group_by(group_cols)
            .agg(pl.sum("time_decay").alias(agg_col))
            .filter(pl.col(agg_col) > 0)
            .with_columns(
                pl.col(agg_col).rank(method='ordinal', descending=True)
                .over(group_cols[-1] if len(group_cols) > 1 else group_cols[0])
                .alias(f'rank_{group_cols[-1]}')
            )
            .filter(pl.col(f'rank_{group_cols[-1]}') <= rank_limit)
        )
    
    # Global popular items
    popular_100 = (
        create_popularity_feature(
            df_train, 
            True, 
            ['node'], 
            'count_all_pop', 
            150
        )
        .sort('count_all_pop', descending=True)
        .head(100)
        .pipe(lambda df: distinct_cookies.join(df, how='cross'))
        .drop('rank_node')
    )
    
    # Popular contact items
    popular_100_contact = (
        create_popularity_feature(
            df_train,
            (pl.col('is_contact') == 1),
            ['node'], 
            'count_contact', 
            150
        )
        .sort('count_contact', descending=True)
        .head(50)
        .pipe(lambda df: distinct_cookies.join(df, how='cross'))
        .drop('rank_node')
    )
    
    # Location-based popularity
    popular_100_location = (
        create_popularity_feature(
            df_train,
            True,
            ['node', 'location'], 
            'count_location', 
            250
        )
        .rename({'location': 'user_location'})
        .pipe(lambda df: user_features.join(df, on="user_location", how="left"))
        .select('cookie', 'node', 'rank_location')
    )
    
    # Category-based popularity
    popular_100_category = (
        create_popularity_feature(
            df_train,
            True,
            ['node', 'category'], 
            'count_category', 
            250
        )
        .rename({'category': 'user_category'})
        .pipe(lambda df: user_features.join(df, on="user_category", how="left"))
        .select('cookie', 'node', 'rank_category')
    )

    int_nodes = df_train.group_by("cookie").agg(
        pl.col("node").sort_by("event_date").last().alias("last_node"),
        pl.col("node").mean().alias("avg_node")
    )
    int_contact_nodes = df_train.filter(pl.col('is_contact') == 1).group_by("cookie").agg(
        pl.col("node").sort_by("event_date").last().alias("last_contact_node"),
    )

    df = (
        als_all
        .join(als_contact, on=['cookie', 'node'], how='full', coalesce=True)
        .join(popular_100_category, on=['cookie', 'node'], how='full', coalesce=True)
        .join(popular_100_location, on=['cookie', 'node'], how='full', coalesce=True)
        .join(popular_100, on=['cookie', 'node'], how='full', coalesce=True)
        .join(popular_100_contact, on=['cookie', 'node'], how='full', coalesce=True)
        .join(i2i_pred_all, on=['cookie', 'node'], how='full', coalesce=True)
        .join(i2i_pred_contact, on=['cookie', 'node'], how='full', coalesce=True)
        .join(topk_avg_text_cookie, on=['cookie', 'node'], how='full', coalesce=True)
        .join(topk_last_text_cookie, on=['cookie', 'node'], how='full', coalesce=True)
        .join(graph_pred, on=['cookie', 'node'], how='full', coalesce=True)
        .join(i2i_df, on=['cookie', 'node'], how='full', coalesce=True)
        .join(tag_cosine, on=['cookie', 'node'], how='full', coalesce=True)
        .join(transformer.select('cookie','node', 'proba'), on=['cookie', 'node'], how='full', coalesce=True)
        .join(df_train.select('cookie', 'node'), on=['cookie', 'node'], how='anti') # check
        .join(item_features, on='node', how='left')
        .join(user_features, on='cookie', how='left')
        .join(
            df_train
            .pipe(add_time_decay, max_date, decay_factor)  # Add time decay
            .group_by(['node', 'location'])
            .agg(
                pl.sum('time_decay').alias('count_location')  # Use sum of time_decay instead of len()
            )
            .filter(pl.col('count_location') > 0)
            .rename({'location': 'user_location'})
            .select('node', 'user_location', 'count_location'), 
            on=['node', 'user_location'], 
            how='left'
        )
        .join(
            df_train
            .pipe(add_time_decay, max_date, decay_factor)  # Add time decay
            .group_by(['node', 'category'])
            .agg(
                pl.sum('time_decay').alias('count_category')  # Use sum of time_decay instead of len()
            )
            .filter(pl.col('count_category') > 0)
            .rename({'category': 'user_category'})
            .select('node', 'user_category', 'count_category'), 
            on=['node', 'user_category'], 
            how='left'
        )
        .filter(pl.col('cookie').is_in(distinct_cookies))
        .filter(pl.col('cookie').is_in(df_train.select(pl.col('cookie').unique()))) #?
    )
    
    print('start scale')
    columns_to_scale = [i for i in df.columns if i not in ['node', 'cookie', 'event', 'target', 'user_location', 'user_category']]
    df = df.with_columns([
        (pl.col(col) - pl.col(col).min()) / 
        (pl.col(col).max() - pl.col(col).min()) + 0.1
        for col in columns_to_scale
    ])
    print('end scale')
    
    if not is_submit:
        df = df.join(df_eval, on=['cookie', 'node'], how='left')
        df = df.with_columns(pl.col('target').fill_null(0))

    if subsample:
        zeros = df.filter(pl.col('target') == 0)
        non_zeros = df.filter(pl.col('target') != 0)
        sampled_zeros = zeros.sample(fraction=subsample, shuffle=True, seed=seed)
        df = pl.concat([sampled_zeros, non_zeros])

    df = (
        df
        .join(avg_text_node, on='node', how='left')
        .join(avg_text_cookie, on='cookie', how='left')
        .join(last_text_cookie, on='cookie', how='left')
        .with_columns(
            (pl.col('average_title_projection').cast(pl.List(pl.Float32)) * pl.col('centroid_vector').cast(pl.List(pl.Float32)))
            .list.sum().alias("dot_centroid")
        )
        .with_columns(
            (pl.col('average_title_projection').cast(pl.List(pl.Float32)) * pl.col('last_item_vector').cast(pl.List(pl.Float32)))
            .list.sum().alias("dot_last_item")
        )
        .drop(['average_title_projection', 'centroid_vector', 'last_item_vector', 'centroid_id'])
    )

    df = (
        df
        .join(als_item_emb, on='node', how='left')
        .join(als_user_emb, on='cookie', how='left')
        .with_columns(
            (pl.col('als_emb_user').cast(pl.List(pl.Float32)) * pl.col('als_emb_node').cast(pl.List(pl.Float32)))
            .list.sum().alias("dot_als")
        )
        .drop(['als_emb_user', 'als_emb_node'])
    )

    df = (
        df
        .join(als_17_item_emb, on='node', how='left')
        .join(als_17_user_emb, on='cookie', how='left')
        .with_columns(
            (pl.col('als_emb_user').cast(pl.List(pl.Float32)) * pl.col('als_emb_node').cast(pl.List(pl.Float32)))
            .list.sum().alias("dot_als_17")
        )
        .drop(['als_emb_user', 'als_emb_node'])
    )

    df = (
        df
        .join(graph_item_emb, on='node', how='left')
        .join(graph_user_emb, on='cookie', how='left')
        .with_columns(
            (pl.col('embedding').cast(pl.List(pl.Float32)) * pl.col('embedding_right').cast(pl.List(pl.Float32)))
            .list.sum().alias("dot_graph")
        )
        .drop(['embedding', 'embedding_right', 'id'])
    )

    df = (
        df.join(score_pairs(pairs=df.select('node', 'cookie'), 
                            cookie_emb=tag_emb_cookie,
                            node_by_feat=tag_emb_node),
                on=['node', 'cookie'],
                how='left'
               )
    )
    
    df = (
        df
        .join(int_nodes, on='cookie', how='left')
        .with_columns(abs(pl.col('node') - pl.col('last_node')).alias('last_node_diff'))
        .with_columns(abs(pl.col('node') - pl.col('avg_node')).alias('avg_node_diff'))
        .join(int_contact_nodes, on='cookie', how='left')
        .with_columns(abs(pl.col('node') - pl.col('last_contact_node')).alias('last_contact_node_diff'))
    )

    if not is_submit:
        n_contacts = df_eval.n_unique(subset=['node', 'cookie'])
        n_retrieved = df.select(pl.sum('target')).item()
        print(f'df built, retrieved {n_retrieved} items from {n_contacts} overall, {n_retrieved/n_contacts}% coverage1')

    return df.join(df_train.select('cookie', 'node'), on=['cookie', 'node'], how='anti')


def fit_lgb_ranker(df, features, cat_features, params, target='target', group_column='cookie'):
    # Sort the DataFrame by the group_column(s)
    sorted_df = df.sort(group_column)
    
    # Extract group values as tuples from the sorted DataFrame
    group_values = sorted_df.select(group_column).rows()
    # Calculate group sizes based on consecutive group values
    group_sizes = [len(list(g)) for _, g in groupby(group_values)]
    
    # Convert to pandas DataFrame for LightGBM compatibility
    sorted_pd_df = sorted_df.to_pandas()
    
    X = sorted_pd_df[features]
    y = sorted_pd_df[target]
    
    # Create LightGBM Dataset with group information
    dataset = lgb.Dataset(
        data=X,
        label=y,
        group=group_sizes,
        feature_name=features,
    )
    
    # Train the LightGBM ranker
    ranker = lgb.train(
        params=params,
        train_set=dataset,
        valid_sets=[dataset],
        valid_names=['train'],
    )
    return ranker


def fit_catboost_ranker(df, features, cat_features, params, target='target', group_column='cookie'):
    # Sort the DataFrame by the group_column to ensure groups are consecutive
    sorted_df = df.sort(group_column)
    
    # Extract group values from the sorted DataFrame
    group_values = sorted_df.select(group_column).rows()
    # Calculate group sizes by grouping consecutive values
    group_sizes = [len(list(g)) for _, g in groupby(group_values)]
    
    # Generate group_id array where each group is assigned a unique integer
    group_id = []
    for idx, size in enumerate(group_sizes):
        group_id.extend([idx] * size)
    group_id = np.array(group_id)
    
    # Convert to pandas DataFrame for CatBoost compatibility
    sorted_pd_df = sorted_df.to_pandas()
    
    X = sorted_pd_df[features]
    y = sorted_pd_df[target]
    
    # Create CatBoost Pool with group_id and categorical features
    pool = Pool(
        data=X,
        label=y,
        group_id=group_id,
        cat_features=cat_features  # List of categorical feature names
    )
    # Initialize and train the CatBoost Ranker
    ranker = CatBoostRanker(**params)
    ranker.fit(pool, verbose_eval=100)
    
    return ranker


import polars as pl
from typing import Optional, Iterable

def score_pairs(
    pairs: pl.DataFrame,
    *,
    cookie_emb: pl.DataFrame,
    node_by_feat: pl.LazyFrame,
    cookie_norm: Optional[pl.DataFrame] = None,
    node_norm: Optional[pl.DataFrame] = None,
    batch_size: int = 3_000,
    use_cosine: bool = True,
) -> pl.DataFrame:
    # --- 0.  Guard clauses & pre‑compute norms if needed --------------------
    # assert pairs.columns[:2] == ["cookie", "node"], f"pairs must have ['cookie','node'], got {pairs.columns}"

    if use_cosine:
        if cookie_norm is None:
            cookie_norm = (cookie_emb
                           .group_by("cookie")
                           .agg((pl.col("val").pow(2).sum().sqrt()).alias("norm"))
                          )
        if node_norm is None:
            node_norm = (node_by_feat
                         .explode("blob")
                         .select([
                             pl.col("blob").struct.field("node").alias("node"),
                             pl.col("blob").struct.field("val").alias("val")
                         ])
                         .group_by("node")
                         .agg((pl.col("val").pow(2).sum().sqrt()).alias("norm"))
                        )

    # --- 1.  Helper that scores one cookie batch ----------------------------
    def _score_batch(batch_cookies: Iterable[str], batch_pairs: pl.DataFrame) -> pl.DataFrame:
        # 1.1  restrict cookie‑embeddings
        sub_c_emb = (cookie_emb
                     .lazy()
                     .filter(pl.col("cookie").is_in(batch_cookies)))

        # 1.2  restrict node space to the nodes that actually appear in `batch_pairs`
        nodes_in_batch = batch_pairs["node"].unique()
        sub_n_emb = (node_by_feat
                     .explode("blob")
                     .select([
                         "feature_indices",
                         pl.col("blob").struct.field("node").alias("node"),
                         pl.col("blob").struct.field("val").alias("val")
                     ])
                     .filter(pl.col("node").is_in(nodes_in_batch))
                     .lazy())

        # 1.3  dot‑product over shared features
        dot = (sub_c_emb
               .join(sub_n_emb, on="feature_indices")            # inner join by hash
               .group_by(["cookie", "node"])
               .agg((pl.col("val") * pl.col("val_right")).sum().alias("dot"))
               .collect(streaming=True))

        if use_cosine:
            dot = (dot
                   .join(cookie_norm, on="cookie")
                   .join(node_norm,   on="node")
                   .with_columns((pl.col("dot") / (pl.col("norm") * pl.col("norm_right")))
                                 .alias("dot_tag_score"))
                   .select(["cookie", "node", "dot_tag_score"]))
        else:
            dot = dot.rename({"dot": "dot_tag_score"})

        # 1.4  left‑join back onto requested pairs → missing → 0
        return (batch_pairs
                .join(dot, on=["cookie", "node"], how="left")
                .with_columns(pl.col("dot_tag_score").fill_null(0.0)))

    # --- 2.  Stream batches over cookies ------------------------------------
    cookies = pairs["cookie"].unique()
    out_frames = []

    for i in range(0, len(cookies), batch_size):
        batch = cookies.slice(i, batch_size)
        batch_pairs = pairs.filter(pl.col("cookie").is_in(batch))
        out_frames.append(_score_batch(batch, batch_pairs))

    return pl.concat(out_frames, rechunk=True)


def top_n_per_cookie(df: pl.DataFrame, n: int, score_col: str) -> pl.DataFrame:
    return (
        df
        .with_columns(
            pl.col(score_col)
            .rank(method='ordinal', descending=True)
            .over('cookie')
            .alias('rank')
        )
        .filter(pl.col('rank') <= n)
        .drop('rank')
    )
    