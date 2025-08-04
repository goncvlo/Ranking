from pathlib import Path
import yaml
import pandas as pd
import joblib

from src.data.load import load_data
from src.data.prepare import prepare_data
from src.models.co_visit import CoVisit
from src.features.features import feature_engineering
from src.features.utils import build_rank_input
from src.models.utils import set_global_seed

# read config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "main" / "config.yml"
with open(CONFIG_PATH, "r") as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

# ensure reproducibility
set_global_seed(seed=config["general"]["seed"])


def inference(user_id: int, config: dict = config) -> pd.DataFrame:
    """Inference pipeline."""

    # load and prepare data
    dfs = load_data(config=config["data_loader"])
    dfs = prepare_data(dataframes=dfs, config=config["data_preparation"])

    if user_id not in dfs["user"]["user_id"].unique():
        return pd.DataFrame(columns=["user_id", "item_id", "movie_title"])

    # get candidates
    candidates = (
        CoVisit(methods=["directional"], k=50)
        .fit(ui_matrix=dfs["data"])
        .rename(columns={"score": "rating"})
        )
    candidates["rating"] = candidates["rating"].round()
    candidates = candidates[candidates["user_id"] == user_id]

    # feature engineering
    user_item_features = feature_engineering(dataframes=dfs)
    df = build_rank_input(ratings=candidates, features=user_item_features)
    del user_item_features

    # load model and get top-3 recommendations
    algorithm = list(config["train"]["ranker"].keys())[0]
    clf = joblib.load(f'{config["train"]["path"]}/{algorithm}.joblib')
    candidates["score"] = clf.predict(X=df["X"])

    candidates = (
        candidates
        .drop_duplicates(subset=["user_id", "item_id"])
        .sort_values(["user_id", "score"], ascending=[True, False])
        .groupby("user_id").head(3)[["user_id", "item_id"]]
        .merge(
            dfs["item"][["item_id", "movie_title"]],
            how="left", on="item_id"
            )
    )
    return candidates
