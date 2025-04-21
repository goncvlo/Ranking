import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from xgboost import XGBRanker
import optuna

class BayesianSearch:
    def __init__(self, config: dict, algorithm: str):
        self.algorithm = algorithm
        self.config = config[self.algorithm]
        self.fixed_params = self.config['fixed']
        self.float_params = set(self.config['float_params'])

    def suggest_hyperparams(self, trial: optuna.trial.Trial) -> dict:
        """Suggest hyperparameters based on the config, using trial."""
        tunable_params = {
            hp: (
                trial.suggest_float(name=hp, low=bounds[0], high=bounds[1])
                if hp in self.float_params
                else trial.suggest_int(name=hp, low=bounds[0], high=bounds[1])
            )
            for hp, bounds in self.config['tunable'].items()
        }
        return {**tunable_params, **self.fixed_params}

    def fit(
            self
            , df_train: dict[str, pd.DataFrame | list]
            , df_valid: dict[str, pd.DataFrame | list]
            , trial: optuna.trial.Trial
            ) -> float:
        
        # set suggested hyper-parameters
        hyperparams = self.suggest_hyperparams(trial)
        clf = XGBRanker(**hyperparams)
        
        # fit the model with early stopping if needed
        clf.fit(
            df_train['X'], df_train['y']
            , group=df_train['group']
            #, eval_set=[(df_valid['X'], df_valid['y'])]
            #, eval_group=[df_valid['group']], early_stopping_rounds=50
            , verbose=False
        )

        # compute predictions and evaluate
        preds = clf.predict(df_valid['X'])
        ndcg_score_result = self._calculate_ndcg(df_valid, preds)

        return np.mean(ndcg_score_result)

    def _calculate_ndcg(self, df_valid: dict[str, list | pd.DataFrame], preds: np.ndarray) -> list:
        """Helper function to calculate NDCG score per group."""
        offset = 0
        ndcgs = []
        for group_size in df_valid['group']:
            y_true_group = df_valid['y'][offset:offset+group_size]
            preds_group = preds[offset:offset+group_size]
            ndcgs.append(ndcg_score([y_true_group], [preds_group]))
            offset += group_size
        return ndcgs
    