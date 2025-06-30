import pandas as pd
from surprise import SVD, CoClustering
from collections import defaultdict

from src.data.load import load_ratings

# supported algorithms
algorithms = {
    "SVD": SVD
    , "CoClustering": CoClustering
}


class Retrieval:
    """Retrieval model class."""

    def __init__(self, algorithm: str, params: dict = None):
        self.algorithm = algorithm
        self.params = params or {}

        if self.algorithm in algorithms:
            self.model = algorithms[self.algorithm](**self.params)
        else:
            raise NotImplementedError(
                f"{algorithm} isn't supported. Select from {list(algorithms.keys())}."
            )
    
    def fit(self, trainset: pd.DataFrame):
        trainset = load_ratings(df=trainset)
        trainset = trainset.build_full_trainset()
        self.model.fit(trainset=trainset)

    def predict(self, testset: pd.DataFrame):
        testset = load_ratings(df=testset)
        testset = [testset.df.loc[i].to_list() for i in range(len(testset.df))]
        return self.model.test(testset=testset, verbose=False)
    

def candidate_generation(df: pd.DataFrame, config: dict[str, str | int]) -> pd.DataFrame:
    """Candidate generation for either positive or negative sampling."""

    results={}
    # read and build training set
    train_set = load_ratings(df=df)
    train_set = train_set.build_full_trainset()
    test_set = train_set.build_anti_testset()

    for sample_method in config.keys():
        # select and fit model
        clf = algorithms[config[sample_method]['method']]
        clf.fit(train_set)

        # get candidates - pool of not interacted items, and add it results dict
        pred = clf.test(test_set)
        recs = top_or_bottom_n(
            pred, n=config[sample_method]['num'], get_top=sample_method=='positive'
            )
        results[sample_method] = recs

    return results

def top_or_bottom_n(predictions, n: int=10, get_top: bool=True) -> pd.DataFrame:
    """Return either the top-N or bottom-N recommendation for each user
    from a set of predictions.

    Args:
        predictions (list): List of predictions, as returned by the method of algorithm.
        n (int): Number of recommendations to output for each user. Default is 10.
        get_top (bool): If True, returns the top-N recs. Else, returns the bottom-N.
    """
    
    # first, map the predictions to each user.
    recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        recs[uid].append((iid, est))
    
    # sort the predictions for each user and retrieve either the top-n or bottom-n.
    for uid, user_ratings in recs.items():
        # sort by estimated rating
        user_ratings.sort(key=lambda x: x[1], reverse=True if get_top else False)
        recs[uid] = user_ratings[:n]
    
    # convert to DataFrame
    recs = [(uid, iid, est) for uid, items in recs.items() for iid, est in items]
    recs = pd.DataFrame(recs, columns=['user_id', 'item_id', 'rating'])

    return recs
