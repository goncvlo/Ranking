import pandas as pd


def hottest_items(ui_matrix: pd.DataFrame, df_items: pd.DataFrame, top_k: int = 10):
    """Hottest unrated items of favorite user genre."""

    # extract genre columns
    genre_cols = (
        df_items
        .drop(columns=["item_id", "movie_title", "release_date", "imdb_url"])
        .columns
        .to_list()
        )
    
    df = ui_matrix.merge(
        df_items[["item_id"] + genre_cols],
        how="left", on="item_id"
        )

    # top items per genre based on avg rating
    items_by_genre = {
        genre: (
            df[df[genre] == 1]
            .groupby("item_id")["rating"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )
        for genre in genre_cols
    }

    # top genre per user (based on most interactions)
    user_genre = (
        df.groupby("user_id")[genre_cols].sum()
        .idxmax(axis=1)
        .reset_index(name="genre")
    )

    # dict of items seen per user
    rated_items = (
        df.groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    # top-k recommendations per user
    user_recs = {}
    for _, row in user_genre.iterrows():
        user = row["user_id"]
        genre = row["genre"]
        u_rated_items = rated_items.get(user, set())

        # top items in user's favorite genre, excluding seen
        top_items = [
            item for item in items_by_genre[genre]
            if item not in u_rated_items # for baseline model change to "in"
            ][:top_k]
        user_recs[user] = top_items
        
    user_recs = pd.DataFrame([
        {"user_id": user, "item_id": item}
        for user, items in user_recs.items()
        for item in items
    ])

    return user_recs


def popular_items(ui_matrix: pd.DataFrame, top_k: int = 10):

    # views/interactions per item
    item_popularity = ui_matrix.groupby("item_id").size().sort_values(ascending=False)
    popular_items = item_popularity.index.tolist()

    # user interacted items
    rated_items = ui_matrix.groupby("user_id")["item_id"].apply(set).to_dict()

    # for each user select popular items excluding his/her rated items
    user_recs = {}
    for user, seen_items in rated_items.items():
        recommended = [item for item in popular_items if item not in seen_items][:top_k]
        user_recs[user] = recommended
    
    user_recs = pd.DataFrame([
        {'user_id': user, 'item_id': item}
        for user, items in user_recs.items()
        for item in items
        ])
    
    return user_recs
