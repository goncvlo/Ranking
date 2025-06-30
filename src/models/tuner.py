import pandas as pd
import optuna

from src.models.ranker import Ranker
from src.models.retrieval import Retrieval
from src.models.evaluator import Evaluation


# supported methods
methods = ["retrieval", "ranker"]


class BayesianSearch:
    def __init__(self, config: dict, method: str, algorithm: str):
        # check and assign method
        if method not in methods:
            raise NotImplementedError(
                f"{method} isn't supported. Select from {methods}."
            )
        
        self.method = method        
        self.algorithm = algorithm
        self.scoring_metric = config[self.method]["scoring_metric"]
        self.param_grid = config[self.method][self.algorithm]
        self.fixed_params = self.param_grid["fixed"]
        self.float_params = set(self.param_grid["float_params"])
        self.results = dict()

    def fit(
            self
            , df_train: dict[str, pd.DataFrame | list]
            , df_valid: dict[str, pd.DataFrame | list]
            , trial: optuna.trial.Trial
            ) -> float:
        
        # set suggested hyper-parameters
        hyperparams = self._suggest_hyperparams(trial)

        if self.method == "ranker":
            # set algorithm instance
            clf = Ranker(algorithm=self.algorithm, params=hyperparams).model
            
            # fit the model with early stopping if needed
            clf.fit(
                df_train["X"], df_train["y"]
                , group=df_train["group"]
                #, eval_set=[(df_valid["X"], df_valid["y"])]
                #, eval_group=[df_valid["group"]], early_stopping_rounds=50
            )

            # compute predictions and evaluate
            scorer = Evaluation(clf=clf)
            score = scorer.fit(
                train=(df_train["X"], df_train["y"], df_train["group"]),
                validation=(df_valid["X"], df_valid["y"], df_valid["group"]),
                )
            self.results[trial.number] = score
            eval_metric = score.loc["validation", self.scoring_metric]

        elif self.method == "retrieval":
            # set algorithm instance
            clf = Retrieval(algorithm=self.algorithm, params=hyperparams)

            # fit model
            clf.fit(trainset=df_train["X"])

            # compute predictions and evaluate
            scorer = Evaluation(clf=clf)
            score = scorer.fit(train=df_train["X"], validation=df_valid["X"])
            self.results[trial.number] = score
            eval_metric = score.loc["validation", self.scoring_metric]

        return eval_metric
    
    def _suggest_hyperparams(self, trial: optuna.trial.Trial) -> dict:
        """Suggest hyperparameters based on the config, using trial."""
        tunable_params = {
            hp: (
                trial.suggest_float(name=hp, low=bounds[0], high=bounds[1])
                if hp in self.float_params
                else trial.suggest_int(name=hp, low=bounds[0], high=bounds[1])
            )
            for hp, bounds in self.param_grid['tunable'].items()
        }
        return {**tunable_params, **self.fixed_params}
    