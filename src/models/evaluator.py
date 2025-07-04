import numpy as np
import pandas as pd
from surprise import accuracy
from sklearn.metrics import ndcg_score
from typing import Union

from src.models.ranker import Ranker
from src.models.retrieval import Retrieval

# supported metrics
metrics = {
    "retrieval": {
        "rmse": accuracy.rmse,
        "mse": accuracy.mse,
        "mae": accuracy.mae,
        "fcp": accuracy.fcp
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
                    row[metric_name] = round(func(predictions=y_pred, verbose=False), 5)
                results.append(row)

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


def recall(y_pred: pd.DataFrame, y_true: pd.DataFrame, hit_rate: bool = False):

    y_pred = y_pred[["user_id", "item_id"]].drop_duplicates().copy()
    y_true = y_true[["user_id", "item_id"]].drop_duplicates().copy()
    y_true["true_flag"] = 1
    if y_true.groupby("user_id")["item_id"].nunique().unique().shape[0]==1:
        div = y_true.groupby(["user_id"])["item_id"].nunique().unique()[0]
    else:
        raise NotImplementedError("Users with diff num. of y_true items.")

    # merge actuals with candidates
    y_pred = (
        y_pred
        .merge(y_true, how="left", on=["user_id", "item_id"])
        .fillna({"true_flag": 0})
    )

    # compute source table for recall
    df = (
        y_pred
        .groupby(by=["user_id"])["true_flag"].sum()
        .reset_index()
        .groupby(by=["true_flag"]).size()
        .reset_index().rename(columns={0: "size"})
        )
    df["size_pct"] = (df["size"]/df["size"].sum()*100).round(1)

    recall_score = 0
    if hit_rate:
        recall_score = df[df["true_flag"]>0]["size_pct"].sum()/100    
    else:
        for v in range(0, div+1):
            if not df[df["true_flag"]==v]["size"].empty:
                recall_score += (v/div)* df[df["true_flag"]==v]["size"].values[0]
        recall_score = recall_score / df["size"].sum()

    return recall_score


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
