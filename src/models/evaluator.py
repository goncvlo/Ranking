import numpy as np
import pandas as pd
from surprise import accuracy
from sklearn.metrics import ndcg_score
from typing import Union

from src.models.ranker import Ranker
from src.models.retrieval import Retrieval
from src.models.utils import _check_user_count


def _at_k(y_pred, y_true, k, score_fn):
    pred = y_pred.groupby("user_id")["item_id"].apply(lambda x: set(x[:k]))
    true = y_true.groupby("user_id")["item_id"].apply(set)
    users = pred.index.intersection(true.index)
    _check_user_count(users)
    return score_fn(pred, true, users)

def recall_at_k(y_pred, y_true, k=10):
    return _at_k(y_pred, y_true, k, lambda p, t, u: 
        np.mean([len(p[i] & t[i]) / len(t[i]) if len(t[i]) else 0 for i in u])
        )

def precision_at_k(y_pred, y_true, k=10):
    return _at_k(y_pred, y_true, k, lambda p, t, u: 
        np.mean([len(p[i] & t[i]) / len(p[i]) if len(p[i]) else 0 for i in u])
        )

def hit_rate_at_k(y_pred, y_true, k=10):
    return _at_k(y_pred, y_true, k, lambda p, t, u: 
        np.mean([1 if len(p[i] & t[i]) > 0 else 0 for i in u])
        )

# supported metrics
metrics = {
    "retrieval": {
        #"rmse": accuracy.rmse,
        #"mse": accuracy.mse,
        #"mae": accuracy.mae,
        #"fcp": accuracy.fcp,
        "recall": recall_at_k,
        "precision": precision_at_k,
        "hit_rate": hit_rate_at_k
        },
    "ranker": {
        "ndcg": ndcg_score
        }
    }


class Evaluation:
    def __init__(self, clf: Union[Retrieval, Ranker]):
        self.model = clf

    def fit(self, metric: str = None, k: int = 10, features: dict[str, pd.DataFrame] = None, **datasets):
        if isinstance(self.model, Retrieval):
            return self._evaluate_retrieval(metric, k=k, features=features, **datasets)
        elif isinstance(self.model, Ranker):
            return self._evaluate_ranker(metric, **datasets)
        else:
            raise TypeError("Unsupported model type.")

    def _evaluate_retrieval(self, metric, k, features: dict[str, pd.DataFrame] = None, **datasets):
        selected_metrics = {metric: metrics["retrieval"][metric]} if metric else metrics["retrieval"]
        results = []

        # candidate-based evaluation
        train_set = datasets.get("train")
        if train_set is None:
            raise ValueError("`train` set is required for recall/precision/hit_rate evaluation.")

        y_candidates = self.model.top_n(user_ids=train_set["user_id"].unique(), k=k, top=True, features=features)

        for name, testset in datasets.items():
            row = {"dataset": name}
            for metric_name, func in selected_metrics.items():
                if metric_name in ["recall", "precision", "hit_rate"]:
                    row[f"{metric_name}@{k}"] = round(func(y_pred=y_candidates, y_true=testset, k=k), 5)
            results.append(row)

        return pd.DataFrame(results).set_index("dataset")

    def _evaluate_ranker(self, metric, **datasets):
        selected_metrics = {metric: metrics["ranker"][metric]} if metric else metrics["ranker"]
        results = []

        for name, (group, X, y) in datasets.items():
            y_pred = np.asarray(self.model.predict(X=X))
            y_true = np.asarray(y)

            y_true_splits = np.split(y_true, np.cumsum(group)[:-1])
            y_pred_splits = np.split(y_pred, np.cumsum(group)[:-1])

            row = {"dataset": name}
            for metric_name, func in selected_metrics.items():
                metric_groups = [
                    round(func(y_true=[true], y_score=[pred]), 5)
                    for true, pred in zip(y_true_splits, y_pred_splits)
                ]
                row[metric_name] = float(np.mean(metric_groups))
            results.append(row)

        return pd.DataFrame(results).set_index("dataset")
    

def coverage(y_pred: pd.DataFrame, catalog: pd.DataFrame):
    # coverage: proportion of unique items recommended
    recommended_items = y_pred["item_id"].unique()
    total_items = catalog["item_id"].unique()
    
    return len(recommended_items) / len(total_items)


def novelty(y_pred: pd.DataFrame, catalog: pd.DataFrame):
    # novelty: items that are less popular (inverted normalized popularity)
    item_popularity = catalog['item_id'].value_counts(normalize=True)
    novelty_score =  y_pred['item_id'].map(
        lambda x: 1 - item_popularity.get(x, 0)
        ).mean()

    return novelty_score
