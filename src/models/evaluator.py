import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from typing import Sequence

def evaluation(y_true, y_pred, group: Sequence[int]) -> float:
    """Calculate mean NDCG score over query groups."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length.")
    if sum(group) != len(y_true):
        raise ValueError("Sum of group sizes must equal the number of predictions.")

    y_true_splits = np.split(y_true, np.cumsum(group)[:-1])
    y_pred_splits = np.split(y_pred, np.cumsum(group)[:-1])

    ndcgs = [
        ndcg_score([true], [pred])
        for true, pred in zip(y_true_splits, y_pred_splits)
    ]
    return float(np.mean(ndcgs))


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
