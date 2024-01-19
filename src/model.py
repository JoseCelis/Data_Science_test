import os
import sys
import mlflow
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from dotenv import load_dotenv, find_dotenv
from sklearn.model_selection import train_test_split
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
load_dotenv(find_dotenv())

sys.path.append(os.getcwd())
from src.utils import read_config

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', None)


random_seed = 123
np.random.seed(random_seed)


def split_data_into_train_test(data, target_col):
    train_data, test_data = train_test_split(data, random_state=random_seed, test_size=0.8, stratify=data[target_col])
    return train_data, test_data


def train_model(experiment, train_data, test_data, target_col, numerical_columns, categorical_columns):
    experiment.setup(train_data,
                     numeric_features=numerical_columns,
                     categorical_features=categorical_columns,
                     target=target_col,
                     session_id=random_seed,
                     data_split_stratify=[target_col],
                     feature_selection=False,
                     fold=5,
                     verbose=False)
    catboost_model = experiment.create_model('catboost')
    tuned_model = experiment.tune_model(catboost_model, search_library='optuna', search_algorithm='tpe')
    return tuned_model


def main():
    config = read_config(['model'])
    input_file = os.path.join(config['input']['path'], config['input']['file_name'])
    data = pd.read_parquet(input_file)

    # set mlflow tracking uri
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

    target_col = config['target_column']
    numerical_columns = list(data.select_dtypes(include=['float64']).columns)
    categorical_columns = list(data.select_dtypes(include=['category']).columns)

    train_data, test_data = split_data_into_train_test(data, target_col)
    experiment = ClassificationExperiment()
    tuned_model = train_model(experiment, train_data, test_data, target_col, numerical_columns, categorical_columns)
    experiment.save_model(tuned_model, f'tuned_model')
    predicted_target = experiment.predict_model(tuned_model, data=test_data)
    print(predicted_target)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main()
