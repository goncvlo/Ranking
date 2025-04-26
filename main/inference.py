import yaml
import pandas as pd
import joblib

from src.data.load import load_data
from src.data.prepare import prepare_data
from src.features.features import feature_engineering

# read config
with open('..\Ranking\main\config.yml', 'r') as file:
    config=yaml.load(file, Loader= yaml.SafeLoader)

def inference(user_ids: list, config: dict=config) -> pd.DataFrame:
    """Inference pipeline."""

    # load and prepare data
    dataframes = load_data(config=config['data_loader'])
    dataframes = prepare_data(dataframes=dataframes)

    if not user_ids or set(user_ids).issubset(set(dataframes['user']['user_id'])):
        return pd.DataFrame(columns=['user_id', 'item_id', 'score'])

    # generate candidates from all the missing items
    candidates = pd.MultiIndex.from_product(
        [dataframes['data']['user_id'].unique(), dataframes['data']['item_id'].unique()]
        , names=['user_id', 'item_id']
        ).to_frame(index=False)

    candidates = candidates.merge(
        dataframes['data'][['user_id', 'item_id']].copy()
        , on=['user_id', 'item_id'], how='left', indicator=True
        )

    candidates = candidates[
        (candidates['_merge'] == 'left_only')
        & (candidates['user_id'].isin(user_ids))
        ][['user_id', 'item_id']]

    # create features
    user_item_features = feature_engineering(dataframes=dataframes)
    candidates = (
        candidates
        .merge(user_item_features['user'], how='left', on='user_id')
        .merge(user_item_features['item'], how='left', on='item_id')
        )
    
    del user_item_features, dataframes

    # load model and get top-3 recommendations
    clf = joblib.load(config['model']['path'])
    feature_cols = [col for col in candidates.columns if col not in ['user_id', 'item_id']]
    candidates['score'] = clf.predict(candidates[feature_cols])

    candidates = (
        candidates
        .sort_values(['user_id', 'score'], ascending=[True, False])
        .groupby('user_id').head(3)
        [['user_id', 'item_id', 'score']]
        )
    
    return candidates
