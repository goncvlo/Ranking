import yaml
import pandas as pd

from src.data.load import load_data
from src.data.prepare import prepare_data
from src.features.features import feature_engineering

# read config
with open('..\main\config.yml', 'r') as file:
    config=yaml.load(file, Loader= yaml.SafeLoader)
del file

# load and prepare data
dataframes = load_data(config=config['data_loader'])
dataframes = prepare_data(dataframes=dataframes)
user_item_features = feature_engineering(dataframes=dataframes)

# create a full user-item cartesian product (all possible interactions)
all_user_item_pairs = pd.MultiIndex.from_product(
    [dataframes['data']['user_id'].unique(), dataframes['data']['item_id'].unique()]
    , names=['user_id', 'item_id']).to_frame(index=False)

# merge with actual ratings to find which ones exist
all_user_item_pairs = all_user_item_pairs.merge(
    dataframes['data'][['user_id', 'item_id']].copy()
    , on=['user_id', 'item_id'], how='left', indicator=True
    )

# filter for those not in the ratings (i.e., where the user has not rated the item)
missing_ratings = all_user_item_pairs[all_user_item_pairs['_merge'] == 'left_only']

# keep only the columns you want
missing_ratings = missing_ratings[['user_id', 'item_id']]

missing_ratings = (
    missing_ratings
    .merge(user_item_features['user'], how='left', on='user_id')
    .merge(user_item_features['item'], how='left', on='item_id')
    )