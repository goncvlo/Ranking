import numpy as np
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
    
    def top_n(self, user_ids: list, n: int = 10):

        if not isinstance(self.model, SVD):
            raise NotImplementedError(f"{self.algorithm} isn't supported.")

        # convert raw user ids to inner ids
        user_ids = list(set(user_ids)) # remove duplicates
        user_inner_ids = [self.model.trainset.to_inner_uid(u) for u in user_ids]

        # get user latent factors for all users [shape is (num_users, n_factors)]
        # compute scores for all items for all users [shape is (num_users, num_items)]
        U = self.model.pu[user_inner_ids]
        S = np.dot(U, self.model.qi.T)
        user_biases = self.model.bu[user_inner_ids].reshape(-1, 1)
        item_biases = self.model.bi.reshape(1, -1)
        global_bias = self.model.trainset.global_mean
        S = S + user_biases + item_biases + global_bias

        # get top-n items based on its score
        top_n_items = []
        for idx, (user_raw_id, user_inner_id) in enumerate(zip(user_ids, user_inner_ids)):

            # items user already rated
            rated_items = set(iid for (iid, _) in self.model.trainset.ur[user_inner_id])

            # mask scores of rated items by setting very low score
            scores = S[idx]
            scores[list(rated_items)] = -np.inf
            
            # get top-n item indices and sort it
            top_n_iids = np.argpartition(-scores, n)[:n]
            top_n_iids = top_n_iids[np.argsort(-scores[top_n_iids])]

            #  convert inner ids to raw ids and get scores
            for item_inner_id in top_n_iids:
                item_raw_id = self.model.trainset.to_raw_iid(item_inner_id)
                score = scores[item_inner_id]
                top_n_items.append((user_raw_id, item_raw_id, score))

        return pd.DataFrame(top_n_items, columns=["user_id", "item_id", "score"])
