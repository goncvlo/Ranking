from xgboost import XGBRanker
from lightgbm import LGBMRanker

# supported algorithms
algorithms = {
    "XGBRanker": XGBRanker,
    "LGBMRanker": LGBMRanker
}

class Ranker:
    """Ranking model class."""

    def __init__(self, algorithm: str, params: dict = None):
        self.algorithm = algorithm
        self.params = params or {}

        if self.algorithm in algorithms:
            self.model = algorithms[self.algorithm](**self.params)
        else:
            raise NotImplementedError(
                f"{algorithm} isn't supported. Select from {list(algorithms.keys())}."
            )
    
    def fit(self, X, y, group=None):
        """Fit ranking model."""
        self.model.fit(X=X, y=y, group=group)

    def predict(self, X):
        """Predict with ranking model."""
        return self.model.predict(X=X)
    