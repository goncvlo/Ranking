import pandas as pd

# column names of three datasets
column_names = {
    'data': ["user_id", "item_id", "rating", "timestamp"]
    , 'user': ["user_id", "age", "gender", "occupation", "zip_code"]
    , 'item': [
        "movie_id", "movie_title", "release_date", "video_release_date", "imdb_url"
        , "action", "adventure", "animation", "children", "comedy"
        , "crime", "documentary", "drama", "fantasy", "film_noir"
        , "horror", "musical", "mystery", "romance", "sci_fi"
        , "thriller", "war", "western", "unknown"
        ]
        }

def load_data(config: dict) -> dict:
    """Load ratings, items and users datasets."""

    dataframes ={}

    for df_name in config['sub_paths']:
        
        sep = ['\t' if df_name=='data' else '|'][0]

        dataframes[df_name] = pd.read_csv(
            config['path'] + df_name
            , sep=sep, header=None, encoding='latin-1'
            , names = column_names[df_name]
        )

    return dataframes