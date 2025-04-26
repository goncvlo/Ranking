from pathlib import Path
import yaml
import pandas as pd
from xgboost import XGBRanker
import joblib

from src.data.load import load_data
from src.data.prepare import prepare_data
from src.features.features import feature_engineering
from src.models.retrieval import candidate_generation
from src.features.utils import build_rank_input

# read config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "main" / "config.yml"
with open(CONFIG_PATH, "r") as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)


def train(config: dict = config):
    """Train pipeline."""

    # load and prepare data
    dataframes = load_data(config=config["data_loader"])
    dataframes = prepare_data(dataframes=dataframes)

    # generate candidates and create user-item features
    train_df = candidate_generation(
        dataframes["data"], config["model"]["retrieval"]
        )
    train_df = pd.concat([
        dataframes["data"].iloc[:, :3],
        train_df["positive"],
        train_df["negative"]
        ], ignore_index=True)

    user_item_features = feature_engineering(dataframes=dataframes)
    ranking_input = build_rank_input(train_df, user_item_features)

    del train_df, user_item_features, dataframes

    # build and save model
    model = XGBRanker(**config["model"]["ranking"]["hyper_params"])
    model.fit(
        ranking_input["X"],
        ranking_input["y"].astype(int),
        group=ranking_input["group"],
        verbose=False,
    )
    joblib.dump(model, config["model"]["path"])


if __name__ == "__main__":
    train(config)
