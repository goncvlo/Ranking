from collections import defaultdict
from itertools import combinations
import pandas as pd


methods_list = ["positive", "negative", "weighted", "directional"]


class CoVisit:
    """Co-Visitation Matrices class."""

    def __init__(self, methods: list[str], top_k: int = 10):
        if not set(methods).issubset(methods_list):
            raise NotImplementedError(
                f"{methods} isn't supported. Select from {methods_list}."
            )

        self.methods = methods
        self.top_k = top_k

    def fit(self, ui_matrix: pd.DataFrame):

        self.matrices = dict()
        self.candidates = dict()

        for method in self.methods:
            # compute co-visitation matrix
            temp_matrix = ui_matrix.copy()
            if method=="positive":
                temp_matrix = ui_matrix[ui_matrix["positive_flag"]==1]
            elif method=="negative":
                temp_matrix = ui_matrix[ui_matrix["negative_flag"]==1]
            
            self.matrices[method] = self.compute(ui_matrix=temp_matrix, method=method)

            # get candidates for each user
            self.candidates[method] = self._get_candidates_all_users(
                ui_matrix=temp_matrix, cov_matrix=self.matrices[method]
                )
        
        # Convert candidates dict to DataFrame with method info
        self.candidates = [
             (method, user_id, item_id, score)
            for method, method_dict in self.candidates.items()
            for user_id, items in method_dict.items()
            for item_id, score in items
            ]
        self.candidates = pd.DataFrame(self.candidates, columns=["method", "user_id", "item_id", "score"])
        return self.candidates

    def compute(self, ui_matrix: pd.DataFrame, method: str):
        """Compute the different co-visitation matrices."""
        matrix = defaultdict(lambda: defaultdict(float))

        if method=="weighted":

            for user_id, group in ui_matrix.groupby("user_id"):
                items = group[["item_id", "rating"]].drop_duplicates() # filter out duplicates
                for (item_i, w_i), (item_j, w_j) in combinations(items.itertuples(index=False), 2):
                    weight = w_i * w_j  # or (w_i + w_j) / 2
                    matrix[item_i][item_j] += weight

        elif method=="directional":

            ui_matrix = ui_matrix.sort_values(["user_id", "timestamp"])

            # in each session (user), for each (i -> j) where j comes after i
            for user_id, group in ui_matrix.groupby("user_id"):
                items = group["item_id"].tolist()
                for i in range(len(items) - 1):
                    item_i = items[i]
                    item_j = items[i + 1]
                    matrix[item_i][item_j] += 1  # directed edge: i -> j

        else:

            user_items = ui_matrix.groupby("user_id")["item_id"].apply(list)

            for items in user_items:
                unique_items = set(items) # filter out duplicates
                for item_i, item_j in combinations(sorted(unique_items), 2):
                    matrix[item_i][item_j] += 1
        
        return matrix

    def _get_candidates(self, user_items: list, cov_matrix: dict):
        """Get top-k candidates for a given user using a co-visit matrix."""

        candidate_scores = defaultdict(float)

        for item in user_items:
            neighbors = cov_matrix.get(item, {})
            for neighbor, score in neighbors.items():
                if neighbor not in user_items:  # filter out interacted items
                    candidate_scores[neighbor] += score

        # top-k scored candidates
        return sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
    
    def _get_candidates_all_users(self, ui_matrix: pd.DataFrame, cov_matrix: dict) -> dict:
        """Get top-k candidates for all users using a co-visit matrix."""

        user_items_map = ui_matrix.groupby("user_id")["item_id"].apply(set).to_dict()
        user_candidates = dict()

        for user_id, items in user_items_map.items():
            candidates = self._get_candidates(items, cov_matrix)
            user_candidates[user_id] = candidates

        return user_candidates
