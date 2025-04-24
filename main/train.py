import yaml
import pandas as pd
from xgboost import XGBRanker
import joblib

from src.data.load import load_data
from src.data.prepare import prepare_data
from src.models.cv_iterator import leave_last_k
from src.data.features import feature_engineering
from src.models.candidate import candidate_generation
from src.data.utils import build_rank_input

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
df_train_neg = candidate_generation(dataframes['data'], n=30, positive_sampling=False)
df_train_pos = candidate_generation(dataframes['data'], n=10, positive_sampling=True)
df_train = pd.concat([dataframes['data'].iloc[:,:3], df_train_neg, df_train_pos], ignore_index=True)
del df_train_neg, df_train_pos

df_train = build_rank_input(ratings=df_train, features=user_item_features)

clf = XGBRanker(**config['model']['hyper_params'])
clf.fit(
    df_train['X'], df_train['y'].astype(int)
    , group=df_train['group']
    , verbose=False
    )

# save model
joblib.dump(clf, 'artifacts/model.joblib')