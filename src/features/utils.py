import pandas as pd

def build_rank_input(
        ratings: pd.DataFrame, features: dict[str, pd.DataFrame]
        ) -> dict[str, pd.DataFrame | list]:
    """Create X, y, and group sets."""

    data = {}

    # number of ratings per user
    data['group'] = ratings.groupby('user_id').size().to_list()
    # user and item features, and ratings
    data['X'] = (
        ratings.copy()
        .merge(features['user'], on='user_id', how='left')
        .merge(features['item'], on='item_id', how='left')
        .drop(columns=['user_id', 'item_id', 'rating'])
        )
    data['y'] = ratings['rating'].round().astype(int)

    return data
