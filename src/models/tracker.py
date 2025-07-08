import os
import pandas as pd
import time
from typing import Union
from datetime import datetime
import mlflow
import subprocess
import webbrowser
from optuna.study import Study
from surprise import AlgoBase
from xgboost import XGBModel
from lightgbm import LGBMModel
import pickle
import ast

from src.models.tuner import BayesianSearch
from src.models.ranker import Ranker
from src.models.retrieval import Retrieval


class Logging:

    def __init__(self, experiment_name: str, run_name: str, input_sample: pd.DataFrame = None):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.input_sample = input_sample

    def log_run(
            self,
            study: Study = None, tuner: BayesianSearch = None,
            clf: Union[Retrieval, Ranker] = None
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
            tuner.metric: study.best_trial.value
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
                input_sample=self.input_sample
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
            clf: Union[Retrieval, Ranker], name: str,
            input_sample: Union[pd.DataFrame, None] = None
            ):
        
        model = clf.model

        if isinstance(model, AlgoBase):
            with open(f"{name}.pkl", "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(f"{name}.pkl", artifact_path=name)
            os.remove(f"{name}.pkl")
        
        else:
            output_sample = clf.predict(input_sample)
            input_sample = input_sample.astype({col: 'float' for col in input_sample.select_dtypes(include='int').columns})
            signature = mlflow.models.signature.infer_signature(input_sample, output_sample)
            
            if isinstance(model, XGBModel):
                mlflow.xgboost.log_model(
                    model,
                    name=name,
                    input_example=input_sample,
                    signature=signature,
                    model_format="json"
                )

            elif isinstance(model, LGBMModel):
                mlflow.lightgbm.log_model(
                    model,
                    name=name,
                    input_example=input_sample,
                    signature=signature
                    )
    
                
def launch_mlflow():
    """Launch MLflow UI on localhost and open it in a browser."""
    subprocess.Popen(["mlflow", "ui"])
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
        order_by=["start_time DESC"]
    )
    
    if runs:
        # resume existing run (will inherit correct experiment)
        existing_run_id = runs[0].info.run_id
        return mlflow.start_run(run_id=existing_run_id)
    else:
        # create new run in the selected experiment
        return mlflow.start_run(run_name=run_name)


def load_params(experiment_name: str, parent_run_name: str) -> dict:
    """
    Get hyperparameters (logged params) from the first child run
    under a given parent run name in the specified MLflow experiment.

    :param experiment_name: Name of the MLflow experiment
    :param parent_run_name: The mlflow.runName tag of the parent run
    :return: Dictionary of parameters from the first child run
    """
    client = mlflow.tracking.MlflowClient()

    # get experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    experiment_id = experiment.experiment_id

    # locate the parent run by tag name
    parent_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{parent_run_name}'",
        max_results=1
    )
    if not parent_runs:
        raise ValueError(f"Parent run '{parent_run_name}' not found.")
    parent_run_id = parent_runs[0].info.run_id

    # get first child run under parent
    child_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        order_by=["start_time DESC"],
        max_results=1
    )
    if not child_runs:
        raise ValueError("No child runs found under the specified parent run.")

    raw_params = child_runs[0].data.params
    return {k: cast_param_value(v) for k, v in raw_params.items()}


def cast_param_value(value: str):
    """Attempt to cast param string to int, float, or bool."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    elif value[0]=="{" and value[-1]=="}":
        return ast.literal_eval(value)
    return value  # fallback to string


def load_model(experiment_name: str, parent_run_name: str):
    """
    Load a model artifact from the first child run under a parent run in a given MLflow experiment.

    :param experiment_name: Name of the MLflow experiment
    :param parent_run_name: The mlflow.runName tag of the parent run
    :return: Loaded model (XGBRanker, LGBMRanker, or pickled model)
    """
    client = mlflow.tracking.MlflowClient()

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    experiment_id = experiment.experiment_id

    parent_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{parent_run_name}'",
        max_results=1
    )
    if not parent_runs:
        raise ValueError(f"Parent run '{parent_run_name}' not found.")
    parent_run_id = parent_runs[0].info.run_id

    child_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        order_by=["start_time DESC"],
        max_results=1
    )
    if not child_runs:
        raise ValueError("No child runs found under the specified parent run.")

    child_run = child_runs[0]
    params = child_run.data.params
    if params is None:
        raise ValueError("No 'algorithm' param found in the child run.")

    # assumes model artifact logged at "model"
    model_uri = f"mlflow-artifacts:/{experiment_id}/{child_run.info.run_id}/artifacts/model"
    if parent_run_name == "XGBRanker":
        return mlflow.xgboost.load_model(model_uri)
    elif parent_run_name == "LGBMRanker":
        return mlflow.lightgbm.load_model(model_uri)
    else:
        # fallback: download and load pickle manually
        model_uri = f"mlflow-artifacts:/{experiment_id}/{child_run.info.run_id}/artifacts/{parent_run_name}/{parent_run_name}.pkl"
        local_path = mlflow.artifacts.download_artifacts(model_uri)
        with open(local_path, "rb") as f:
            return pickle.load(f)
