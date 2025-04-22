import pandas as pd
from surprise import Dataset, Reader, SVD
from collections import defaultdict

def candidate_generation(df: pd.DataFrame, n: int, positive_sampling: bool) -> pd.DataFrame:
    """Candidate generation for either positive or negative sampling."""

    # read and build training set
    train_set = Dataset.load_from_df(
        df[["user_id", "item_id", "rating"]]
        , reader=Reader(rating_scale=(1, 3))
        )
    train_set = train_set.build_full_trainset()

    # select and fit model
    clf = SVD()
    clf.fit(train_set)

    # get candidates - pool of not interacted items
    test_set = train_set.build_anti_testset()
    predictions = clf.test(test_set)
    top_n = top_or_bottom_n(predictions, n=n, get_top=positive_sampling)

    # convert to DataFrame
    top_n = [(uid, iid, est) for uid, items in top_n.items() for iid, est in items]
    top_n = pd.DataFrame(top_n, columns=['user_id', 'item_id', 'rating'])

    return top_n

def top_or_bottom_n(predictions, n: int=10, get_top: bool=True):
    """Return either the top-N or bottom-N recommendation for each user from a set of predictions.

    Args:
        predictions (list): List of predictions, as returned by the test method of an algorithm.
        n (int): Number of recommendations to output for each user. Default is 10.
        get_top (bool): If True, returns the top-N recommendations. If False, returns the bottom-N.

    Returns:
        A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """
    
    # first, map the predictions to each user.
    recommendations = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        recommendations[uid].append((iid, est))
    
    # sort the predictions for each user and retrieve either the top-n or bottom-n.
    for uid, user_ratings in recommendations.items():
        # sort by estimated rating
        user_ratings.sort(key=lambda x: x[1], reverse=True if get_top else False)
        recommendations[uid] = user_ratings[:n]

    return recommendations
