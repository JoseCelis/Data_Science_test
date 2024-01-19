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

random_seed = 123
np.random.seed(random_seed)


def split_data_into_train_test(data, target_col):
    train_data, test_data = train_test_split(data, random_state=random_seed, test_size=0.8, stratify=data[target_col])
    return train_data, test_data


def train_model(experiment, train_data, test_data, target_col):
    balancing_data = experiment.setup(train_data,
                                      target=target_col,
                                      session_id=random_seed,
                                      fix_imbalance=True,
                                      verbose=False,
                                      keep_features=[train_data.columns],
                                      # categorical_features=[]
                                      )
    balanced_train_data = balancing_data.dataset_transformed.copy()
    experiment.setup(balanced_train_data,
                     target=target_col,
                     session_id=random_seed,
                     data_split_stratify=[target_col],
                     feature_selection=False,
                     fold=5,
                     verbose=False)
    adaboost_model = experiment.create_model('ada')
    tuned_model = experiment.tune_model(adaboost_model, search_library='optuna', search_algorithm='tpe')
    return tuned_model


def main():
    data = pd.read_parquet('data/preprocessed/data.parquet')
    target_col = 'Churn'

    # mlflow.set_tracking_uri("http://localhost:5000")

    train_data, test_data = split_data_into_train_test(data, target_col)
    experiment = ClassificationExperiment()
    tuned_model = train_model(experiment, train_data, test_data, target_col)
    experiment.save_model(tuned_model, f'tuned_model')
    # predicted_target = experiment.predict_model(tuned_model, data=test_data)
    # print(predicted_target)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main()
