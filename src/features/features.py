import numpy as np
import pandas as pd

def feature_engineering(dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create user-item features for ranker algorithm."""

    # 1.1. create user only features
    user_df = dataframes['user'].copy()

    user_df = user_df.rename(columns={'age': 'u_age'})
    user_df['u_gender_is_male'] = np.where(user_df['gender']=='M', 1, 0)
    job_dummies = pd.get_dummies(user_df['occupation'], prefix='u_job_is').astype(int)
    user_df = user_df[['user_id', 'u_age', 'u_gender_is_male']]
    user_df = pd.concat([user_df, job_dummies], axis=1)
    del job_dummies

    # 2.1. create item item features
    item_df = dataframes['item'].copy()

    item_df['i_release_year'] = item_df['release_date'].dt.year
    item_df['i_release_month'] = item_df['release_date'].dt.year
    item_df = item_df.drop(columns=['movie_title', 'release_date', 'imdb_url'])
    item_df = item_df.rename(columns={
        col: f"i_gender_is_{col}"
        if col not in ['item_id', 'i_release_year', 'i_release_month'] else col
        for col in item_df.columns
    })

    # 3.1. create user-item features at the user level
    data_df = dataframes['data'].copy()
    
    user_feats = data_df.groupby('user_id').agg({
        'rating': ['count', 'nunique', 'mean', 'std']
        })
    user_feats.columns = ['u_rating_count', 'u_rating_nunique', 'u_rating_mean', 'u_rating_std']
    user_feats = user_feats.reset_index()

    user_df = user_df.merge(right=user_feats, on='user_id', how='left')
    del user_feats

    # 3.2. create item-user features at the item level
    item_feats = data_df.groupby('item_id').agg({
        'rating': ['count', 'nunique', 'mean', 'std']
        })
    item_feats.columns = ['i_rating_count', 'i_rating_nunique', 'i_rating_mean', 'i_rating_std']
    item_feats = item_feats.reset_index()

    item_df = item_df.merge(right=item_feats, on='item_id', how='left')
    del item_feats

    return {'user': user_df, 'item': item_df }
