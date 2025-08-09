import random
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def leave_last_k(df: pd.DataFrame, config: dict):
    """
    Splits a DataFrame into train and test sets by leaving the last `k` interactions
    (based on timestamp) per user in the test set.
    """

    # sort by user and timestamp DESC (latest first)
    df_sorted = df.sort_values(by=["user_id", "timestamp"], ascending=[True, False])

    # rank ratings per timestamp for each user
    df_sorted["rank"] = df_sorted.groupby(by=["user_id"]).cumcount()

    # test set = last k per user (i.e. first k rows of df_sorted)
    test_df = (
        df_sorted[df_sorted["rank"] < config["leave_last_k"]]
        .drop(columns="rank")
        .reset_index(drop=True)
    )
    train_df = (
        df_sorted[df_sorted["rank"] >= config["leave_last_k"]]
        .drop(columns="rank")
        .reset_index(drop=True)
    )

    return train_df, test_df


def set_global_seed(seed: int = 42):
    # os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def _check_user_count(users, expected=943):
    if len(users) != expected:
        warnings.warn(
            f"Number of users is {len(users)}, expected {expected}.", UserWarning
        )


class ClassificationDataset(Dataset):
    def __init__(self, df_train, features, num_negatives: int = 4):
        self.df = df_train[df_train["rating"] > 0].reset_index(drop=True)

        self.user_feats = (
            features["user"]
            .drop(columns=["user_id", "user_id_encoded"])
            .to_numpy(dtype=np.float32)
        )
        self.item_feats = (
            features["item"]
            .drop(columns=["item_id", "item_id_encoded"])
            .to_numpy(dtype=np.float32)
        )

        # for negative sampling
        self.user_pos_items = (
            self.df.groupby("user_id_encoded")["item_id_encoded"].apply(set).to_dict()
        )
        self.num_items = features["item"].shape[0]
        self.num_negatives = num_negatives

        # Pre-build all (user, item, label) triples
        self.samples = []
        for _, row in self.df.iterrows():
            user_idx = row["user_id_encoded"]
            pos_idx = row["item_id_encoded"]

            # Positive
            self.samples.append((user_idx, pos_idx, 1))

            # Negatives
            negs = set()
            while len(negs) < self.num_negatives:
                neg_idx = np.random.randint(0, self.num_items)
                if neg_idx not in self.user_pos_items[user_idx]:
                    negs.add(neg_idx)

            for neg_idx in negs:
                self.samples.append((user_idx, neg_idx, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_idx, item_idx, label = self.samples[idx]
        user_vec = torch.from_numpy(self.user_feats[user_idx])
        item_vec = torch.from_numpy(self.item_feats[item_idx])
        label = torch.tensor(label, dtype=torch.float32)
        return user_vec, item_vec, label
