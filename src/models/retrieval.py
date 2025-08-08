import heapq
from collections import defaultdict
from typing import Union

import numpy as np
import pandas as pd
import torch
from surprise import SVD, CoClustering, KNNWithMeans
from torch.utils.data import DataLoader

from src.data.load import load_ratings
from src.models.utils import BPRDataset


def bpr_loss_multi(model, user_vec, pos_vec, neg_vecs):
    batch_size, num_neg, _ = neg_vecs.size()

    user_emb, pos_emb = model(user_vec, pos_vec)  # [B, D]
    user_expanded = user_emb.unsqueeze(1).expand(-1, num_neg, -1)  # [B, K, D]

    neg_embs = model.item_mlp(neg_vecs.view(-1, neg_vecs.size(-1))).view(
        batch_size, num_neg, -1
    )  # [B, K, D]

    # scores (cosine similarity since embeddings normalized)
    pos_score = (user_emb * pos_emb).sum(dim=1, keepdim=True)  # [B, 1]
    neg_score = (user_expanded * neg_embs).sum(dim=2)  # [B, K]

    diff = pos_score - neg_score  # [B, K]
    return -torch.mean(torch.nn.functional.logsigmoid(diff))


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int],
        dropout: list[float],
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim, dropout_rate in zip(hidden_layers, dropout):
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            if dropout_rate > 0:
                layers.append(torch.nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(torch.nn.Linear(prev_dim, output_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return torch.nn.functional.normalize(self.mlp(x), p=2, dim=-1)


class TwoTower(torch.nn.Module):
    def __init__(
        self,
        user_dim: int,
        item_dim: int,
        embedding_dim: int,
        user_layers: list[int],
        item_layers: list[int],
        user_dropout: list[float],
        item_dropout: list[float],
        lr: float,
        num_negatives: int = 4,
    ):
        self.lr = lr
        self.num_negatives = num_negatives
        super().__init__()
        self.user_mlp = MLP(user_dim, embedding_dim, user_layers, user_dropout)
        self.item_mlp = MLP(item_dim, embedding_dim, item_layers, item_dropout)

    def forward(self, user_vec, item_vec):
        user_emb = self.user_mlp(user_vec)
        item_emb = self.item_mlp(item_vec)
        return user_emb, item_emb

    def dot_score(self, user_vec, item_vec):
        user_emb, item_emb = self.forward(user_vec, item_vec)
        return (user_emb * item_emb).sum(dim=1)

    def fit(
        self, trainset: pd.DataFrame, features: dict[str, pd.DataFrame], epochs: int = 5
    ):

        # convert to DataLoader instance and set optimizer
        dataset = BPRDataset(trainset, features, num_negatives=self.num_negatives)
        dataset = DataLoader(dataset, batch_size=256, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        for epoch in range(epochs):
            self.train()
            total_loss = 0.0

            for user_vec, pos_vec, neg_vecs in dataset:
                optimizer.zero_grad()
                loss = bpr_loss_multi(self, user_vec, pos_vec, neg_vecs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

    @torch.no_grad()
    def top_n(self, features: dict[str, pd.DataFrame], k=10, device="cpu"):
        self.eval()

        features["user"] = features["user"].sort_values(by=["user_id_encoded"])
        features["item"] = features["item"].sort_values(by=["item_id_encoded"])
        user_ids = features["user"]["user_id"].values
        item_ids = features["item"]["item_id"].values
        num_items = len(features["item"])

        # convert to tensors and compute emdeddings
        user_vecs = torch.tensor(
            features["user"].drop(columns=["user_id", "user_id_encoded"]).values,
            dtype=torch.float32,
        ).to(device)
        item_vecs = torch.tensor(
            features["item"].drop(columns=["item_id", "item_id_encoded"]).values,
            dtype=torch.float32,
        ).to(device)

        user_embs = self.user_mlp(user_vecs)  # [num_users, D]
        item_embs = self.item_mlp(item_vecs)  # [num_items, D]

        # compute scores: [num_users, num_items]
        scores = torch.matmul(user_embs, item_embs.T)

        # for each user, get top-k items
        topk_scores, topk_indices = torch.topk(scores, k, dim=1)

        # convert to DataFrame
        preds = []
        for u_idx, user_id in enumerate(user_ids):
            items = item_ids[topk_indices[u_idx].cpu().numpy()]
            scores_u = topk_scores[u_idx].cpu().numpy()
            for item, score in zip(items, scores_u):
                preds.append((user_id, item, score))

        return pd.DataFrame(
            preds, columns=["user_id", "item_id", "rating"]
        ).sort_values(by=["user_id", "rating"], ascending=[True, False])


# supported algorithms
ALGORITHMS = {
    "SVD": SVD,
    "CoClustering": CoClustering,
    "KNNWithMeans": KNNWithMeans,
    "TwoTower": TwoTower,
}


class Retrieval:
    """Retrieval model class."""

    def __init__(self, algorithm: str, params: dict = None):
        self.algorithm = algorithm
        self.params = params or {}

        if self.algorithm in ALGORITHMS:
            self.model = ALGORITHMS[self.algorithm](**self.params)
        else:
            raise NotImplementedError(
                f"{algorithm} isn't supported. Select from {list(ALGORITHMS.keys())}."  # nosec
            )

    def fit(
        self,
        trainset: pd.DataFrame,
        features: dict[str, pd.DataFrame] = None,
        epochs: int = None,
    ):
        if not isinstance(self.model, TwoTower):
            trainset = load_ratings(df=trainset)
            trainset = trainset.build_full_trainset()
            self.model.fit(trainset=trainset)
        else:
            self.model.fit(trainset=trainset, features=features, epochs=epochs)

    def predict(self, testset: pd.DataFrame):
        testset = load_ratings(df=testset)
        testset = [testset.df.loc[i].to_list() for i in range(len(testset.df))]
        return self.model.test(testset=testset, verbose=False)

    def top_n(
        self,
        user_ids: list,
        k: int = 10,
        top: bool = True,
        features: dict[str, pd.DataFrame] = None,
    ):

        if isinstance(self.model, TwoTower):
            return self.model.top_n(features=features, k=k)

        user_ids = list(set(user_ids))
        top_n_items = []
        # convert raw user ids to inner ids
        user_inner_ids = [self.model.trainset.to_inner_uid(u) for u in user_ids]
        signal = -1 if top else 1

        if isinstance(self.model, SVD):

            # get user latent factors for all users [shape is (num_users, n_factors)]
            # compute scores for all items for all users [shape is (num_users, num_items)]
            U = self.model.pu[user_inner_ids]
            S = np.dot(U, self.model.qi.T)
            user_biases = self.model.bu[user_inner_ids].reshape(-1, 1)
            item_biases = self.model.bi.reshape(1, -1)
            global_bias = self.model.trainset.global_mean
            S = S + user_biases + item_biases + global_bias

            # get top-n items based on its score
            for idx, (user_raw_id, user_inner_id) in enumerate(
                zip(user_ids, user_inner_ids)
            ):

                # items user already rated
                rated_items = {
                    item_inner_id
                    for (item_inner_id, _) in self.model.trainset.ur[user_inner_id]
                }

                # mask scores of rated items by setting very low score
                scores = S[idx]
                scores[list(rated_items)] = signal * np.inf

                # get top-n item indices and sort it
                top_item_inner_ids = np.argpartition(signal * scores, k)[:k]
                top_item_inner_ids = top_item_inner_ids[
                    np.argsort(signal * scores[top_item_inner_ids])
                ]

                #  convert inner ids to raw ids and get scores
                for item_inner_id in top_item_inner_ids:
                    item_raw_id = self.model.trainset.to_raw_iid(item_inner_id)
                    score = scores[item_inner_id]
                    top_n_items.append((user_raw_id, item_raw_id, score))

        elif isinstance(self.model, KNNWithMeans):

            for user_raw_id, user_inner_id in zip(user_ids, user_inner_ids):
                # get nearest neighbors (inner ids)
                neighbors = self.model.get_neighbors(user_inner_id, k=self.model.k)

                # get items rated by neighbors
                neighbor_items = defaultdict(list)
                for neighbor_id in neighbors:
                    for item_inner_id, rating in self.model.trainset.ur[neighbor_id]:
                        neighbor_items[item_inner_id].append(rating)

                # remove items already rated by target user
                rated_items = {
                    item_inner_id
                    for (item_inner_id, _) in self.model.trainset.ur[user_inner_id]
                }
                candidate_items = {
                    item_inner_id: ratings
                    for item_inner_id, ratings in neighbor_items.items()
                    if item_inner_id not in rated_items
                }

                # compute mean rating as estimated score
                # self.model.estimate(user_inner_id, item_inner_id) for KNN rating, slower
                predictions = [
                    (item_inner_id, np.mean(ratings))
                    for item_inner_id, ratings in candidate_items.items()
                ]

                # select top-n
                if top:
                    top_n = heapq.nlargest(k, predictions, key=lambda x: x[1])
                else:
                    top_n = heapq.nsmallest(k, predictions, key=lambda x: x[1])
                for item_inner_id, score in top_n:
                    item_raw_id = self.model.trainset.to_raw_iid(item_inner_id)
                    top_n_items.append((user_raw_id, item_raw_id, score))

        elif isinstance(self.model, CoClustering):

            # candidate set = non rated items, can it be improved?
            all_inner_items = list(self.model.trainset.all_items())
            all_raw_items = {
                item_inner_id: self.model.trainset.to_raw_iid(item_inner_id)
                for item_inner_id in all_inner_items
            }

            for user_raw_id, user_inner_id in zip(user_ids, user_inner_ids):

                rated_items = {
                    item_inner_id
                    for (item_inner_id, _) in self.model.trainset.ur[user_inner_id]
                }
                candidate_items = [
                    item_inner_id
                    for item_inner_id in all_inner_items
                    if item_inner_id not in rated_items
                ]

                predictions = []
                for item_inner_id in candidate_items:
                    item_raw_id = all_raw_items[item_inner_id]
                    est = self.model.predict(user_raw_id, item_raw_id).est
                    predictions.append((item_raw_id, est))

                if top:
                    top_n = heapq.nlargest(k, predictions, key=lambda x: x[1])
                else:
                    top_n = heapq.nsmallest(k, predictions, key=lambda x: x[1])
                for item_raw_id, score in top_n:
                    top_n_items.append((user_raw_id, item_raw_id, score))

        else:
            raise NotImplementedError(f"{self.algorithm} isn't supported.")

        if top:
            top_n_items = pd.DataFrame(
                top_n_items, columns=["user_id", "item_id", "rating"]
            ).sort_values(by=["user_id", "rating"], ascending=[True, False])
        else:
            top_n_items = pd.DataFrame(
                top_n_items, columns=["user_id", "item_id", "rating"]
            ).sort_values(by=["user_id", "rating"], ascending=[True, True])

        return top_n_items
