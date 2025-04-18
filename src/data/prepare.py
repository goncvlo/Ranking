import pandas as pd

def prepare_data(dataframes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Transform the dataframes."""

    dataframes['item'] = dataframes['item'].drop(columns=['video_release_date'])

    return dataframes
