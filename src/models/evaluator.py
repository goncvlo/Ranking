import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

def evaluation(y_true: pd.Series, y_pred: pd.Series, group: list) -> float:
    """Calculate evaluation metric score per group."""

    offset = 0
    ndcgs = []
    for group_size in group:
        y_true_group = y_true[offset:offset+group_size]
        preds_group = y_pred[offset:offset+group_size]
        ndcgs.append(ndcg_score([y_true_group], [preds_group]))
        offset += group_size
    
    return np.mean(ndcgs)
