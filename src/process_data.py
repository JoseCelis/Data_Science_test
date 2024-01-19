import os
import sys
import logging
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', None)


def detect_and_fill_empty_cells(data):
    """
    check for null values in object columns and fill the
    :param data:
    :return:
    """
    object_columns = data.select_dtypes('object').columns
    data[object_columns] = data[object_columns].replace(' ', np.nan).replace('', np.nan)
    if data.isna().any().any():
        logging.info(f'{data.isna().sum()} null values detected.')
        # drop rows and columns where all cells are NaN
        data = data.dropna(how='all').dropna(how='all', axis=1)
        # manually check which columns need to be converted from string to float
        data['TotalCharges'] = data['TotalCharges'].astype('float64')
        data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)
    else:
        logging.info(f'No null values detected.')
    return data


def modify_type_of_columns(data, to_cat_cols=None, to_float_cols=None, to_int_cols=None, to_str_cols=None):
    """
    convert type of columns on request
    :return:
    """
    if to_cat_cols:
        data[to_cat_cols] = data[to_cat_cols].astype('category')
    elif to_float_cols:
        data[to_float_cols] = data[to_float_cols].astype(float)
    elif to_int_cols:
        data[to_int_cols] = data[to_int_cols].astype(int)
    elif to_str_cols:
        data[to_str_cols] = data[to_str_cols].astype(str)
    else:
        logging.info('Type of the columns will be preserved')
    return data


def process_raw_data(data):
    data = detect_and_fill_empty_cells(data)
    data = modify_type_of_columns(data, to_cat_cols=['gender', 'SeniorCitizen', 'Partner', 'Dependents'], to_int_cols=['tenure'])
    # parquet format preserves the dtypes unlike csv format
    os.makedirs('data/preprocessed/', exist_ok=True)
    data.to_parquet('data/preprocessed/data.parquet')
    return None


def main():
    data = pd.read_csv('https://raw.githubusercontent.com/srees1988/predict-churn-py/main/customer_churn_data.csv')
    process_raw_data(data)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main()
