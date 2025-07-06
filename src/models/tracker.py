import os
import pandas as pd
import time
from typing import Union
from datetime import datetime
import mlflow
import subprocess
import webbrowser
import optuna
from surprise import AlgoBase
from xgboost import XGBModel
from lightgbm import LGBMModel
import pickle

from src.models.tuner import BayesianSearch


def log_run(
        experiment_name: str,
        study: optuna.study.Study,
        tuner: BayesianSearch
        ):

    with get_or_create_run(run_name=tuner.algorithm, experiment_name=experiment_name) as parent:
        today_date = datetime.now().strftime("%d%b%Y").upper()
        with mlflow.start_run(run_name=today_date, nested=True):

            # adding fixed params
            hyper_params = study.best_trial.params | tuner.param_grid["fixed"]
            log_artifact(artifact=hyper_params, artifact_name="params")
            log_artifact(artifact=study.best_trial.duration.total_seconds(), artifact_name="duration_secs")
            log_artifact(artifact=study.best_trial.number, artifact_name="trial_index")
            log_artifact(artifact=study.best_trial.value, artifact_name=tuner.scoring_metric)
            log_artifact(artifact=study.trials_dataframe(), artifact_name="bayes_search", artifact_path="stats")
            log_artifact(artifact=tuner.param_grid, artifact_name="param_grid")
            log_model(artifact=tuner.artifacts["models"][study.best_trial.number].model, artifact_name="model_instance", input_sample=tuner.input_sample)

            # convert trials results dictionary to dataframe
            results = []
            for split_id, split_data in tuner.artifacts["evaluation_metrics"].items():
                for dataset, metrics in split_data.items():
                    row = {'split_id': split_id, 'metric': dataset}
                    row.update(metrics)
                    results.append(row)
            results = pd.DataFrame(results)
            log_artifact(artifact=results, artifact_name="evaluation_metrics", artifact_path="stats")


def launch_mlflow():
    """Launch MLflow UI on localhost and open it in a browser."""
    subprocess.Popen(["mlflow", "ui"])
    time.sleep(2)
    webbrowser.open(f"http://127.0.0.1:5000")
    mlflow.autolog(disable=True)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")


def log_artifact(artifact: Union[pd.DataFrame, dict, float, int], artifact_name: str, artifact_path: str = "default"):
    """Save object locally, log it as an artifact, and delete it."""

    if isinstance(artifact, pd.DataFrame):
        artifact.to_csv(f"{artifact_name}.csv")
        mlflow.log_artifact(f"{artifact_name}.csv", artifact_path=artifact_path)
        os.remove(f"{artifact_name}.csv")
    
    if isinstance(artifact, dict):
        if artifact_name=="params":
            mlflow.log_params(params=artifact)
        else:
            mlflow.log_dict(dictionary=artifact, artifact_file=f"{artifact_name}.yaml")
    
    if isinstance(artifact, (float, int)):
        mlflow.log_metric(key=artifact_name, value=artifact)


def get_or_create_run(run_name: str, experiment_name: str):
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
    

def log_model(artifact: Union[AlgoBase, XGBModel, LGBMModel], artifact_name: str, input_sample: pd.DataFrame):

    output_example = artifact.predict(input_sample)
    input_example = input_sample.astype({col: 'float' for col in input_sample.select_dtypes(include='int').columns})
    signature = mlflow.models.signature.infer_signature(input_example, output_example)


    if isinstance(artifact, AlgoBase):
        with open(f"{artifact_name}.pkl", "wb") as f:
            pickle.dump(artifact, f)
        mlflow.log_artifact(f"{artifact_name}.pkl", artifact_path=artifact_name)
        os.remove(f"{artifact_name}.pkl")
    
    if isinstance(artifact, XGBModel):
        mlflow.xgboost.log_model(
            artifact,
            name=artifact_name,
            input_example=input_example,
            signature=signature,
            model_format="json"
        )

    if isinstance(artifact, LGBMModel):
        mlflow.lightgbm.log_model(
            artifact,
            name=artifact_name,
            input_example=input_example,
            signature=signature
            )


def load_model(path: str, algorithm: str) -> Union[AlgoBase, XGBModel, LGBMModel]:

    if algorithm=="XGBRanker":
        return mlflow.xgboost.load_model(path)
    elif algorithm=="LGBMRanker":
        return mlflow.lightgbm.load_model(path)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)
        