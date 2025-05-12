import pandas as pd
import optuna
from src.models.ranker import Ranker
from src.models.evaluator import evaluation

class BayesianSearch:
    def __init__(self, config: dict, algorithm: str):
        self.algorithm = algorithm
        self.config = config[self.algorithm]
        self.fixed_params = self.config['fixed']
        self.float_params = set(self.config['float_params'])

    def fit(
            self
            , df_train: dict[str, pd.DataFrame | list]
            , df_valid: dict[str, pd.DataFrame | list]
            , trial: optuna.trial.Trial
            ) -> float:
        
        # set suggested hyper-parameters
        hyperparams = self._suggest_hyperparams(trial)
        clf = Ranker(algorithm=self.algorithm, params=hyperparams).model
        
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
        eval_metric = evaluation(df_valid['y'], preds, df_valid['group'])

        return eval_metric
    
    def _suggest_hyperparams(self, trial: optuna.trial.Trial) -> dict:
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
    