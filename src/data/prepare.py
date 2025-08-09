import numpy as np
import pandas as pd


def prepare_data(
    dataframes: dict[str, pd.DataFrame], config: dict
) -> dict[str, pd.DataFrame]:
    """Transform dataframes."""

    # user encoder mapping
    user2id = {uid: i for i, uid in enumerate(dataframes["data"]["user_id"].unique())}
    for df in ["data", "user"]:
        dataframes[df]["user_id_encoded"] = dataframes[df]["user_id"].map(user2id)

    # user set
    dataframes["user"]["occupation"] = np.where(
        dataframes["user"]["occupation"] == "none",
        "other",  # could be set as unemployed
        dataframes["user"]["occupation"],
    )

    # item set
    dataframes["item"] = dataframes["item"].drop(columns=["video_release_date"])
    dataframes["item"]["release_date"] = pd.to_datetime(
        dataframes["item"]["release_date"], format="%d-%b-%Y", errors="coerce"
    )

    # ratings set
    map = {v: int(k) for k, vals in config["rating_conversion"].items() for v in vals}
    dataframes["data"]["rating"] = dataframes["data"]["rating"].map(map)
    for k, v in config["rating_flag"].items():
        dataframes["data"][f"{k}_flag"] = np.where(
            dataframes["data"]["rating"].isin(v), 1, 0
        )

    return dataframes
