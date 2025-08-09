import os
import pickle  # nosec
import subprocess  # nosec
import time
import webbrowser
from datetime import datetime
from typing import Union

import mlflow
import pandas as pd
from lightgbm import LGBMModel
from optuna.study import Study
from surprise import AlgoBase
from xgboost import XGBModel

from src.models.ranker import Ranker
from src.models.retrieval import Retrieval
from src.models.tuner import BayesianSearch


class Logging:

    def __init__(
        self, experiment_name: str, run_name: str, input_sample: pd.DataFrame = None
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.input_sample = input_sample

    def log_run(
        self,
        study: Study = None,
        tuner: BayesianSearch = None,
        clf: Union[Retrieval, Ranker] = None,
    ):

        with get_run(self.experiment_name, self.run_name) as parent:
            now = datetime.now().strftime("%d%b%Y_%H%M%S").upper()
            with mlflow.start_run(run_name=now, nested=True):

                if study and tuner:
                    self._log_tuner(study=study, tuner=tuner)
                elif clf:
                    self._log_training(clf=clf)

    def _log_tuner(self, study: Study, tuner: BayesianSearch):

        # objects to be logged
        metrics = {
            "duration_secs": study.best_trial.duration.total_seconds(),
            "trial_index": study.best_trial.number,
            tuner.metric: study.best_trial.value,
        }

        # logging
        mlflow.log_params(params=tuner.artifacts["models"][-1].params)
        mlflow.log_metrics(metrics=metrics)
        mlflow.log_dict(dictionary=tuner.param_grid, artifact_file="param_grid.yaml")
        self._log_df(study.trials_dataframe(), "bayes_search", "stats")
        self._log_df(tuner.artifacts["metrics_test"], "metrics_test", "stats")

        # convert trials results dictionary to dataframe
        results = []
        for split_id, split_data in tuner.artifacts["metrics_valid"].items():
            for dataset, metric in split_data.items():
                row = {"trial_index": split_id, "metric": dataset}
                row.update(metric)
                results.append(row)
        results = pd.DataFrame(results)
        self._log_df(results, "metrics_valid", "stats")

        # log best validation model and test model
        for i_model, name in zip([-1, metrics["trial_index"]], ["valid", "test"]):
            self._log_model(
                clf=tuner.artifacts["models"][i_model],
                name=f"{tuner.algorithm}_{name}",
                input_sample=self.input_sample,
            )

    def _log_training(self, clf: Union[Retrieval, Ranker]):

        mlflow.log_params(params=clf.params)
        self._log_model(clf=clf, name=clf.algorithm, input_sample=self.input_sample)

    def _log_df(self, df: pd.DataFrame, name: str, path: str):

        df.to_csv(f"{name}.csv")
        mlflow.log_artifact(f"{name}.csv", artifact_path=path)
        os.remove(f"{name}.csv")

    def _log_model(
        self,
        clf: Union[Retrieval, Ranker],
        name: str,
        input_sample: Union[pd.DataFrame, None] = None,
    ):

        model = clf.model

        if isinstance(model, AlgoBase):
            with open(f"{name}.pkl", "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(f"{name}.pkl", artifact_path=name)
            os.remove(f"{name}.pkl")

        else:
            output_sample = clf.predict(input_sample)
            input_sample = input_sample.astype(
                {
                    col: "float"
                    for col in input_sample.select_dtypes(include="int").columns
                }
            )
            signature = mlflow.models.signature.infer_signature(
                input_sample, output_sample
            )

            if isinstance(model, XGBModel):
                mlflow.xgboost.log_model(
                    model,
                    name=name,
                    input_example=input_sample,
                    signature=signature,
                    model_format="json",
                )

            elif isinstance(model, LGBMModel):
                mlflow.lightgbm.log_model(
                    model, name=name, input_example=input_sample, signature=signature
                )


def launch_mlflow():
    """Launch MLflow UI on localhost and open it in a browser."""
    subprocess.Popen(["mlflow", "ui"])  # nosec
    time.sleep(2)
    webbrowser.open(f"http://127.0.0.1:5000")
    mlflow.autolog(disable=True)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")


def get_run(experiment_name: str, run_name: str):
    client = mlflow.tracking.MlflowClient()

    # get or create experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    # set the experiment context for mlflow
    mlflow.set_experiment(experiment_name)

    # search for run by name
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
    )

    if runs:
        # resume existing run (will inherit correct experiment)
        existing_run_id = runs[0].info.run_id
        return mlflow.start_run(run_id=existing_run_id)
    else:
        # create new run in the selected experiment
        return mlflow.start_run(run_name=run_name)
