from pathlib import Path
import yaml
import pandas as pd
import joblib

from src.data.load import load_data
from src.data.prepare import prepare_data
from src.features.features import feature_engineering

# read config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "main" / "config.yml"
with open(CONFIG_PATH, "r") as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)


def inference(user_id: int, config: dict = config) -> pd.DataFrame:
    """Inference pipeline."""
    # load and prepare data
    dataframes = load_data(config=config["data_loader"])
    dataframes = prepare_data(dataframes=dataframes)

    if user_id not in dataframes['user']['user_id'].unique():
        return pd.DataFrame(columns=["user_id", "item_id", "movie_title"])

    # get candidates through anti-train set
    candidates = pd.MultiIndex.from_product(
        [
            dataframes["data"]["user_id"].unique(),
            dataframes["data"]["item_id"].unique(),
        ],
        names=["user_id", "item_id"],
    ).to_frame(index=False)

    candidates = candidates.merge(
        dataframes["data"][["user_id", "item_id"]].copy(),
        on=["user_id", "item_id"],
        how="left",
        indicator=True,
    )

    candidates = candidates[
        (candidates["_merge"] == "left_only")
        & (candidates["user_id"] == user_id)
    ][["user_id", "item_id"]]

    # create features
    user_item_features = feature_engineering(dataframes=dataframes)
    candidates = candidates.merge(
        user_item_features["user"], how="left", on="user_id"
    ).merge(user_item_features["item"], how="left", on="item_id")
    del user_item_features

    # load model and get top-3 recommendations
    clf = joblib.load(config["model"]["path"])
    feature_cols = [
        col for col in candidates.columns if col not in ["user_id", "item_id"]
    ]
    candidates["score"] = clf.predict(candidates[feature_cols])

    candidates = (
        candidates.sort_values(["user_id", "score"], ascending=[True, False])
        .groupby("user_id").head(3)[["user_id", "item_id"]]
        .merge(
            dataframes["item"][["item_id", "movie_title"]],
            how="left", on="item_id"
            )
    )
    return candidates
