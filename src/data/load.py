import pandas as pd

COLUMN_NAMES = {
    'data': ["user_id", "item_id", "rating", "timestamp"],
    'user': ["user_id", "age", "gender", "occupation", "zip_code"],
    'item': [
        "item_id", "movie_title", "release_date", "video_release_date", "imdb_url",
        "unknown", "action", "adventure", "animation", "children", "comedy",
        "crime", "documentary", "drama", "fantasy", "film_noir", "horror",
        "musical", "mystery", "romance", "sci_fi", "thriller", "war", "western"
    ]
}

def load_data(config: dict) -> dict[str, pd.DataFrame]:
    """Load ratings, items, and users datasets from local directory."""

    dataframes = {}

    # load each dataset as defined in the sub_paths
    for key in config['sub_paths']:
        sep = '\t' if key == 'data' else '|'

        dataframes[key] = pd.read_csv(
            f"{config['path']}/u.{key}",
            sep=sep,
            header=None,
            names=COLUMN_NAMES[key],
            encoding='latin-1'
        )

    return dataframes
