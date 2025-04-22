import numpy as np
import pandas as pd

def prepare_data(dataframes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Transform dataframes."""

    # user set
    dataframes['user']['occupation'] = np.where(
        dataframes['user']['occupation']=='none'
        , 'other' # could be set as unemployed
        , dataframes['user']['occupation']
        )
    
    # item set
    dataframes['item'] = dataframes['item'].drop(columns=['video_release_date'])
    dataframes['item']['release_date'] = pd.to_datetime(
        dataframes['item']['release_date'], format='%d-%b-%Y', errors='coerce'
        )
    
    # ratings set
    dataframes['data']['rating'] = np.where(
        dataframes['data']['rating'].isin([1, 2, 3]), 1, dataframes['data']['rating']-2
    )
    
    return dataframes
