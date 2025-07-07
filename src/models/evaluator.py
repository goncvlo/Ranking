import numpy as np
import pandas as pd
from surprise import accuracy
from sklearn.metrics import ndcg_score
from typing import Union

from src.models.ranker import Ranker
from src.models.retrieval import Retrieval
from src.models.utils import _check_user_count


def recall_at_k(y_pred, y_true, k=10):
    pred = y_pred.groupby("user_id")["item_id"].apply(lambda x: set(x[:k]))
    true = y_true.groupby("user_id")["item_id"].apply(set)

    users = pred.index.intersection(true.index)
    _check_user_count(users)

    recalls = [
        len(pred[u] & true[u]) / len(true[u]) if len(true[u]) > 0 else 0
        for u in users
    ]

    return sum(recalls) / len(recalls)


def precision_at_k(y_pred, y_true, k=10):
    pred = y_pred.groupby("user_id")["item_id"].apply(lambda x: set(x[:k]))
    true = y_true.groupby("user_id")["item_id"].apply(set)

    users = pred.index.intersection(true.index)
    _check_user_count(users)

    precisions = [
        len(pred[u] & true[u]) / len(pred[u]) if len(pred[u]) > 0 else 0
        for u in users
    ]

    return sum(precisions) / len(precisions)


def hit_rate_at_k(y_pred, y_true, k=10):
    pred = y_pred.groupby("user_id")["item_id"].apply(lambda x: set(x[:k]))
    true = y_true.groupby("user_id")["item_id"].apply(set)

    users = pred.index.intersection(true.index)
    _check_user_count(users)

    hits = [
        1 if len(pred[u] & true[u]) > 0 else 0
        for u in users
    ]

    return sum(hits) / len(hits)


# supported metrics
metrics = {
    "retrieval": {
        "rmse": accuracy.rmse,
        "mse": accuracy.mse,
        "mae": accuracy.mae,
        "fcp": accuracy.fcp,
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

    def fit(self, metric: str = None, **datasets):
        """
        Args:
        - metric: str, optional
            Metric name to compute. If None, computes all metrics.
        - datasets: keyword arguments
            Dataset name mapped to tuple of (X, y), e.g., train=(X_train, y_train)

        Returns:
        - float if one metric and one dataset, otherwise pd.DataFrame
        """
        results = []

        # select metric(s)
        if isinstance(self.model, Retrieval):
            selected_metrics = {metric: metrics["retrieval"][metric]} if metric else metrics["retrieval"]

            for name, testset in datasets.items():
                y_pred = self.model.predict(testset=testset)

                row = {"dataset": name}
                for metric_name, func in selected_metrics.items():
                    if metric_name in ["rmse", "mse", "mae", "fcp"]:
                        row[metric_name] = round(func(predictions=y_pred, verbose=False), 5)
                results.append(row)

            # candidate retrieval evaluation - precision, recall and hit rate
            train_set = datasets["train"]
            y_candidates = self.model.top_n(user_ids=train_set["user_id"].unique(), n=10, top=True)
            row_train = results[0]
            row_valid_or_test = results[1]
            if "validation" in datasets.keys():
                y_relevant = datasets["validation"]
            else:
                y_relevant = datasets["test"]
            for metric_name, func in selected_metrics.items():
                if metric_name in ["recall", "precision", "hit_rate"]:
                    row_valid_or_test[metric_name] = round(func(y_pred=y_candidates, y_true=y_relevant, k=10), 5)
                    row_train[metric_name] = -1
            
            
        elif isinstance(self.model, Ranker):
            selected_metrics = {metric: metrics["ranker"][metric]} if metric else metrics["ranker"]

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

        # format output
        results_df = pd.DataFrame(results).set_index("dataset")
        if len(results_df.columns) == 1 and results_df.shape[0] == 1:
            return results_df.iloc[0, 0]
        else:
            return results_df


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
