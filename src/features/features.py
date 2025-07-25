import numpy as np
import pandas as pd

def feature_engineering(dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create user-item features for ranker algorithm."""

    # 1.1. create user only features
    user_df = dataframes["user"].copy()

    user_df = user_df.rename(columns={"age": "u_age"})
    user_df["u_gender_is_male"] = np.where(user_df["gender"]=="M", 1, 0)
    job_dummies = pd.get_dummies(user_df["occupation"], prefix="u_job_is").astype(int)
    user_df = user_df[["user_id", "u_age", "u_gender_is_male"]]
    user_df["u_age"] = (user_df["u_age"] - user_df["u_age"].mean()) / user_df["u_age"].std()
    user_df = pd.concat([user_df, job_dummies], axis=1)
    del job_dummies

    # 2.1. create item only features
    item_df = dataframes["item"].copy()

    item_df["i_release_year"] = item_df["release_date"].dt.year
    item_df["i_release_month"] = item_df["release_date"].dt.month
    for col in ["i_release_year", "i_release_month"]:
        item_df[col] = (item_df[col] - item_df[col].mean()) / item_df[col].std()

    item_df = item_df.drop(columns=["movie_title", "release_date", "imdb_url"])
    item_df = item_df.rename(columns={
        col: f"i_genre_is_{col}"
        if col not in ["item_id", "i_release_year", "i_release_month"] else col
        for col in item_df.columns
    })

    # 3. create cross user-item features
    data_df = dataframes["data"].copy()
    #data_df["timestamp"] = pd.to_datetime(data_df["timestamp"], unit="s")
    #data_df["hour"] = data_df["timestamp"].dt.hour
    #data_df["dayofweek"] = data_df["timestamp"].dt.dayofweek
    data_df = data_df.merge(item_df, on="item_id", how="left")

    # 3.1. at the user level
    # genre share (how often user rates each genre)
    genre_cols = [col for col in item_df.columns if col.startswith("i_genre_is_")]
    genre_shares = data_df.groupby("user_id")[genre_cols].mean()
    genre_shares.columns = [f"u_{col.split('_')[-1]}_rating_share" for col in genre_cols]
    for col in genre_shares.columns:
        genre_shares[col] = (genre_shares[col] - genre_shares[col].mean()) / genre_shares[col].std()    

    # average rating per genre
    genre_ratings = data_df[genre_cols].multiply(data_df["rating"], axis=0)
    rating_sums = genre_ratings.groupby(data_df["user_id"]).sum()
    genre_counts = data_df.groupby("user_id")[genre_cols].sum()
    genre_means = rating_sums.div(genre_counts.replace(0, np.nan))
    genre_means.columns = [f"u_{col.split('_')[-1]}_rating_mean" for col in genre_cols]
    for col in genre_means.columns:
        genre_means[col] = (genre_means[col] - genre_means[col].mean()) / genre_means[col].std()  

    del genre_cols, genre_ratings, rating_sums, genre_counts

    user_feats = pd.concat([genre_shares, genre_means], axis=1).reset_index()
    user_df = user_df.merge(user_feats, on="user_id", how="left")
    del genre_shares, genre_means, user_feats

    # ratings distribution
    user_feats = data_df.groupby("user_id").agg({
        "rating": ["count", "nunique", "mean", "std"]
        })
    user_feats.columns = ["u_rating_count", "u_rating_nunique", "u_rating_mean", "u_rating_std"]
    for col in user_feats.columns:
        user_feats[col] = (user_feats[col] - user_feats[col].mean()) / user_feats[col].std()

    user_feats = user_feats.reset_index()

    user_df = user_df.merge(right=user_feats, on="user_id", how="left").fillna(0)
    del user_feats

    # 3.2. at the item level
    # ratings distribution
    item_feats = data_df.groupby("item_id").agg({
        "rating": ["count", "nunique", "mean", "std"]
        })
    item_feats.columns = ["i_rating_count", "i_rating_nunique", "i_rating_mean", "i_rating_std"]
    for col in item_feats.columns:
        item_feats[col] = (item_feats[col] - item_feats[col].mean()) / item_feats[col].std()

    item_feats = item_feats.reset_index()
    item_df = item_df.merge(right=item_feats, on="item_id", how="left").fillna(0)
    del item_feats

    return {"user": user_df, "item": item_df }
