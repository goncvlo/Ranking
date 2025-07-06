import numpy as np
import random
import pandas as pd
import warnings


def leave_last_k(df: pd.DataFrame, config: dict):
    """
    Splits a DataFrame into train and test sets by leaving the last `k` interactions
    (based on timestamp) per user in the test set.
    """
    
    # sort by user and timestamp DESC (latest first)
    df_sorted = df.sort_values(by=['user_id', 'timestamp'], ascending=[True, False])

    # rank ratings per timestamp for each user
    df_sorted['rank'] = df_sorted.groupby(by=['user_id']).cumcount()

    # test set = last k per user (i.e. first k rows of df_sorted)
    test_df = df_sorted[df_sorted['rank'] < config['leave_last_k']].drop(columns='rank').reset_index(drop=True)
    train_df = df_sorted[df_sorted['rank'] >= config['leave_last_k']].drop(columns='rank').reset_index(drop=True)

    return train_df, test_df


def set_global_seed(seed: int = 42):
    #os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def _check_user_count(users, expected=943):
    if len(users) != expected:
        warnings.warn(
            f"Number of users is {len(users)}, expected {expected}.",
            UserWarning
        )
