import pandas as pd

def baseline_model(dataframes: dict[str, pd.DataFrame], n: int=10):
    """Compute baseline model - most popular items per genre."""

    # merge ratings with items dataframe
    df = dataframes['data'].merge(
        dataframes['item'].drop(columns=['movie_title', 'release_date', 'imdb_url']),
        how='left', on='item_id'
        )

    # top-n items by genre
    top_items_by_genre = {
        genre: (
            df[df[genre] == 1]
            .groupby('item_id')
            .size()
            .nlargest(n)
            .index.tolist()
        )
        for genre in df.columns[4:]
    }

    # get top genre per user
    user_genre = (
        df.groupby('user_id')[df.columns[4:]].sum()
        .idxmax(axis=1)
        .reset_index(name='genre')
    )

    # build recommendations
    result = pd.concat(
        [
            pd.DataFrame({
                'user_id': user_genre[user_genre['genre'] == genre]['user_id'].values.repeat(n)
                , 'item_id': top_items_by_genre[genre] * len(user_genre[user_genre['genre'] == genre])
            })
            for genre in top_items_by_genre
        ],
        ignore_index=True
    )

    # merge with actual ratings
    result = (
        result
        .merge(
            dataframes['data'][['user_id', 'item_id', 'rating']]
            , on=['user_id', 'item_id'], how='left'
            )
        .dropna()
        .sort_values('user_id')
        .reset_index(drop=True)
    )

    # select users with n//2 ratings
    users_ratings = result.groupby('user_id').size().reset_index().rename(columns={0: 'size'})
    result = result[result['user_id'].isin(
        users_ratings[users_ratings['size']>=n//2]['user_id'].to_list()
        )]
    result = result.groupby('user_id').head(n//2).reset_index(drop=True)
    result['est_rating'] = n//2 - result.groupby('user_id').cumcount()
    
    return result
