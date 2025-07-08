from pathlib import Path
import yaml
import pandas as pd
import joblib

from src.data.load import load_data
from src.data.prepare import prepare_data
from src.models.retrieval import Retrieval
from src.models.baseline import popular_items
from src.models.co_visit import CoVisit
from src.features.features import feature_engineering
from src.features.utils import build_rank_input
from src.models.ranker import Ranker
from src.models.utils import set_global_seed


# read config
CONFIG_PATH = Path(__file__).resolve().parent.parent / "main" / "config.yml"
with open(CONFIG_PATH, "r") as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)

# ensure reproducibility
set_global_seed(seed=config["general"]["seed"])


def train(config: dict = config):
    """Train pipeline."""

    # load and prepare data
    dfs = load_data(config=config['data_loader'])
    dfs = prepare_data(dataframes=dfs, config=config["data_preparation"])

    # train and log retrieval models
    for algorithm, params in config["train"]["retrieval"].items():
        clf = Retrieval(algorithm=algorithm, params=params)
        clf.fit(trainset=dfs["data"])
        joblib.dump(clf, f'main/artifacts/{algorithm}.joblib')

    # negative sampling for ranking model
    neg_sample_1 = popular_items(
        ui_matrix=dfs["data"],
        k=config["train"]["negative_sample"]["popular"])
    neg_sample_2 = CoVisit(methods=["negative"]).fit(ui_matrix=dfs["data"])
    neg_sample = pd.concat([neg_sample_1, neg_sample_2], ignore_index=True)

    neg_sample = neg_sample[["user_id", "item_id"]]
    neg_sample["rating"] = list(config["data_preparation"]["rating_conversion"].keys())[0]

    del neg_sample_1, neg_sample_2

    # feature engineering for ranking model
    user_item_features = feature_engineering(dataframes=dfs)
    df = pd.concat([dfs["data"], neg_sample], ignore_index=True)
    df = build_rank_input(ratings=df.iloc[:, :3], features=user_item_features)

    del neg_sample, user_item_features

    # train and log ranker models
    for algorithm, params in config["train"]["ranker"].items():
        clf = Ranker(algorithm=algorithm, params=params)
        clf.fit(X=df["X"], y=df["y"], group=df["group"])
        joblib.dump(clf, f'main/artifacts/{algorithm}.joblib')


if __name__ == "__main__":
    train(config)
