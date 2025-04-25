import yaml
import pandas as pd
from xgboost import XGBRanker
import joblib

from src.data.load import load_data
from src.data.prepare import prepare_data
from src.models.cv_iterator import leave_last_k
from src.features.features import feature_engineering
from src.models.retrieval import candidate_generation
from src.features.utils import build_rank_input

# read config
with open('..\main\config.yml', 'r') as file:
    config=yaml.load(file, Loader= yaml.SafeLoader)
del file

# load and prepare data
dataframes = load_data(config=config['data_loader'])
dataframes = prepare_data(dataframes=dataframes)

# create user-item features
user_item_features = feature_engineering(dataframes=dataframes)
# perform positive and negative sampling
candidates = candidate_generation(dataframes['data'], config['model']['retrieval'])
df_train = pd.concat(
    [dataframes['data'].iloc[:,:3], candidates['positive'], candidates['negative']]
    , ignore_index=True
    )
del candidates

df_train = build_rank_input(ratings=df_train, features=user_item_features)

clf = XGBRanker(**config['model']['hyper_params'])
clf.fit(
    df_train['X'], df_train['y'].astype(int)
    , group=df_train['group']
    , verbose=False
    )

# save model
joblib.dump(clf, 'artifacts/model.joblib')