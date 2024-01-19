import os
import sys
import mlflow
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from dotenv import load_dotenv, find_dotenv
from pycaret.classification import ClassificationExperiment
load_dotenv(find_dotenv())

sys.path.append(os.getcwd())
from src.utils import read_config

random_seed = 123
np.random.seed(random_seed)


def look_for_best_model(data, target_col, numerical_columns, categorical_columns, experiment):
    experiment.setup(data, target=target_col, session_id=random_seed, log_experiment=True,
                     categorical_features=categorical_columns, numeric_features=numerical_columns,
                     experiment_name='batch1', fold=5)
    experiment.add_metric('logloss', 'Log Loss', log_loss, greater_is_better=False)
    best_model = experiment.compare_models()
    logging.info(best_model)
    return best_model


def run_pipeline(data, target_col, numerical_columns, categorical_columns):
    experiment = ClassificationExperiment()
    best_model = look_for_best_model(data, target_col, numerical_columns, categorical_columns, experiment)
    os.makedirs('models', exist_ok=True)
    experiment.save_model(best_model, 'looking_for_best_model')


def main():
    config = read_config(['model'])
    input_file = os.path.join(config['input']['path'], config['input']['file_name'])

    data = pd.read_parquet(input_file)
    # set mlflow tracking uri
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    # mlflow.set_tracking_uri("http://localhost:5000")

    target_col = config['target_column']
    numerical_columns = list(data.select_dtypes(include=['float64']).columns)
    categorical_columns = list(data.select_dtypes(include=['category']).columns)
    run_pipeline(data, target_col, numerical_columns, categorical_columns)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main()
