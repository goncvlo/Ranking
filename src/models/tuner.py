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
        self.artifacts = dict()
        for artifact_type in ["models", "evaluation_metrics"]:
            self.artifacts[artifact_type] = dict()

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
            clf = Ranker(algorithm=self.algorithm, params=hyperparams)
            
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
                train=(df_train["group"], df_train["X"], df_train["y"]),
                validation=(df_valid["group"], df_valid["X"], df_valid["y"])
                )
            self.artifacts["evaluation_metrics"][trial.number] = score
            self.artifacts["models"][trial.number] = clf
            eval_metric = score.loc["validation", self.scoring_metric]

        elif self.method == "retrieval":
            # set algorithm instance
            clf = Retrieval(algorithm=self.algorithm, params=hyperparams)

            # fit model
            clf.fit(trainset=df_train["X"])

            # compute predictions and evaluate
            scorer = Evaluation(clf=clf)
            score = scorer.fit(train=df_train["X"], validation=df_valid["X"])
            self.artifacts["evaluation_metrics"][trial.number] = score
            self.artifacts["models"][trial.number] = clf
            eval_metric = score.loc["validation", self.scoring_metric]

        # log input samples
        self.input_sample = df_train["X"].head()

        return eval_metric
    
    def _suggest_hyperparams(self, trial: optuna.trial.Trial) -> dict:
        """Suggest hyperparameters based on the config, using trial."""
        tunable_params = {}

        for hp, bounds in self.param_grid['tunable'].items():
            if hp in self.param_grid["float_params"]:
                tunable_params[hp] = trial.suggest_float(hp, bounds[0], bounds[1])
            elif hp in self.param_grid["categ_params"]:
                tunable_params[hp] = trial.suggest_categorical(hp, bounds)
            else:
                tunable_params[hp] = trial.suggest_int(hp, bounds[0], bounds[1])

        hyperparams = {**tunable_params, **self.param_grid["fixed"]}

        if self.algorithm=="KNNWithMeans":
            # merge hyperparams in sim_options param
            sim_options = {
                "name": hyperparams["name"]
                , "user_based": hyperparams["user_based"]
                , "min_support": hyperparams["min_support"]
            }

            del hyperparams["name"], hyperparams["user_based"], hyperparams["min_support"]
            hyperparams["sim_options"] = sim_options
        
        return hyperparams
    