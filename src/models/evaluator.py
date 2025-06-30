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

            for name, (X, y, group) in datasets.items():
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


def recs_score(recommendations: pd.DataFrame, catalog: pd.DataFrame) -> dict[str, float]:
    """
    Calculate recommendation quality metrics: coverage and novelty.

    Parameters:
    - recommendations: top-k recommended items.
    - catalog: available ratings set.
    """
    # coverage: proportion of unique items recommended
    recommended_items = recommendations["item_id"].unique()
    total_items = catalog["item_id"].unique()
    coverage_score = len(recommended_items) / len(total_items)
    
    # novelty: items that are less popular (inverted normalized popularity)
    item_popularity = catalog['item_id'].value_counts(normalize=True)
    novelty_score =  recommendations['item_id'].map(
        lambda x: 1 - item_popularity.get(x, 0)
        ).mean()

    return {
        "coverage": coverage_score,
        "novelty": novelty_score
    }
