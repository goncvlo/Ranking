import numpy as np
import pandas as pd
from surprise import SVD, CoClustering, KNNWithMeans
from collections import defaultdict
import heapq


from src.data.load import load_ratings

# supported algorithms
algorithms = {
    "SVD": SVD
    , "CoClustering": CoClustering
    , "KNNWithMeans": KNNWithMeans
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
    
    def top_n(self, user_ids: list, n: int = 10, top: bool = True):

        user_ids = list(set(user_ids))
        top_n_items = []
        # convert raw user ids to inner ids
        user_inner_ids = [self.model.trainset.to_inner_uid(u) for u in user_ids]
        signal = -1 if top else 1

        if isinstance(self.model, SVD):

            # get user latent factors for all users [shape is (num_users, n_factors)]
            # compute scores for all items for all users [shape is (num_users, num_items)]
            U = self.model.pu[user_inner_ids]
            S = np.dot(U, self.model.qi.T)
            user_biases = self.model.bu[user_inner_ids].reshape(-1, 1)
            item_biases = self.model.bi.reshape(1, -1)
            global_bias = self.model.trainset.global_mean
            S = S + user_biases + item_biases + global_bias

            # get top-n items based on its score
            for idx, (user_raw_id, user_inner_id) in enumerate(zip(user_ids, user_inner_ids)):

                # items user already rated
                rated_items = set(item_inner_id for (item_inner_id, _) in self.model.trainset.ur[user_inner_id])

                # mask scores of rated items by setting very low score
                scores = S[idx]
                scores[list(rated_items)] = signal*np.inf
                
                # get top-n item indices and sort it
                top_item_inner_ids = np.argpartition(signal*scores, n)[:n]
                top_item_inner_ids = top_item_inner_ids[np.argsort(signal*scores[top_item_inner_ids])]

                #  convert inner ids to raw ids and get scores
                for item_inner_id in top_item_inner_ids:
                    item_raw_id = self.model.trainset.to_raw_iid(item_inner_id)
                    score = scores[item_inner_id]
                    top_n_items.append((user_raw_id, item_raw_id, score))
        
        elif isinstance(self.model, KNNWithMeans):
            
            for user_raw_id, user_inner_id in zip(user_ids, user_inner_ids):
                # get nearest neighbors (inner ids)
                neighbors = self.model.get_neighbors(user_inner_id, k=self.model.k)

                # get items rated by neighbors
                neighbor_items = defaultdict(list)
                for neighbor_id in neighbors:
                    for item_inner_id, rating in self.model.trainset.ur[neighbor_id]:
                        neighbor_items[item_inner_id].append(rating)

                # remove items already rated by target user
                rated_items = set(item_inner_id for (item_inner_id, _) in self.model.trainset.ur[user_inner_id])
                candidate_items = {
                    item_inner_id: ratings for item_inner_id, ratings in neighbor_items.items()
                    if item_inner_id not in rated_items
                }

                # compute mean rating as estimated score
                # self.model.estimate(user_inner_id, item_inner_id) for KNN rating, slower
                predictions = [
                    (item_inner_id, np.mean(ratings)) for item_inner_id, ratings in candidate_items.items()
                ]

                # select top-n
                if top:
                    top_n = heapq.nlargest(n, predictions, key=lambda x: x[1])
                else:
                    top_n = heapq.nsmallest(n, predictions, key=lambda x: x[1])
                for item_inner_id, score in top_n:
                    item_raw_id = self.model.trainset.to_raw_iid(item_inner_id)
                    top_n_items.append((user_raw_id, item_raw_id, score))

        elif isinstance(self.model, CoClustering):

            # candidate set = non rated items, can it be improved?
            all_inner_items = list(self.model.trainset.all_items())
            all_raw_items = {item_inner_id: self.model.trainset.to_raw_iid(item_inner_id) for item_inner_id in all_inner_items}

            for user_raw_id, user_inner_id in zip(user_ids, user_inner_ids):

                rated_items = set(item_inner_id for (item_inner_id, _) in self.model.trainset.ur[user_inner_id])
                candidate_items = [item_inner_id for item_inner_id in all_inner_items if item_inner_id not in rated_items]

                predictions = []
                for item_inner_id in candidate_items:
                    item_raw_id = all_raw_items[item_inner_id]
                    est = self.model.predict(user_raw_id, item_raw_id).est
                    predictions.append((item_raw_id, est))
                
                if top:
                    top_n = heapq.nlargest(n, predictions, key=lambda x: x[1])
                else:
                    top_n = heapq.nsmallest(n, predictions, key=lambda x: x[1])
                for item_raw_id, score in top_n:
                    top_n_items.append((user_raw_id, item_raw_id, score))
    
        else:
            raise NotImplementedError(f"{self.algorithm} isn't supported.")

        if top:
            top_n_items = (
                pd.DataFrame(top_n_items, columns=["user_id", "item_id", "score"])
                .sort_values(by=["user_id", "score"], ascending=[True, False])
                )
        else:
            top_n_items = (
                pd.DataFrame(top_n_items, columns=["user_id", "item_id", "score"])
                .sort_values(by=["user_id", "score"], ascending=[True, True])
                )
            
        return top_n_items