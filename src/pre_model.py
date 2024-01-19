import os
import sys
import mlflow
import logging
import pandas as pd
from sklearn.metrics import log_loss
from dotenv import load_dotenv, find_dotenv
from pycaret.classification import ClassificationExperiment
load_dotenv(find_dotenv())
sys.path.append(os.getcwd())


def look_for_best_model(data, target_col, ignore_cols, experiment):
    experiment.setup(data, target=target_col, session_id=123, ignore_features=ignore_cols, log_experiment=True,
                     experiment_name='batch1', fold=5)
    experiment.add_metric('logloss', 'Log Loss', log_loss, greater_is_better=False)
    best_model = experiment.compare_models()
    logging.info(best_model)
    return best_model


def run_pipeline(data, target_col, ignore_cols):
    experiment = ClassificationExperiment()
    best_model = look_for_best_model(data, target_col, ignore_cols, experiment)
    os.makedirs('models', exist_ok=True)
    experiment.save_model(best_model, 'looking_for_best_model')


def main():
    data = pd.read_parquet('data/preprocessed/data.parquet')
    # set mlflow tracking uri
    # mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    # mlflow.set_tracking_uri("http://localhost:5000")

    target_col = 'Churn'
    ignore_cols = ['customerID']
    run_pipeline(data, target_col, ignore_cols)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main()
