import os
import sys
import logging
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
from src.utils import read_config

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
        # n the EDA II noticed the distributions do not change after dropping nulls
        data = data.dropna(axis=0)
    else:
        logging.info(f'No null values detected.')
    return data


def modify_type_of_columns(data, to_cat_cols=None, to_float_cols=None, to_int_cols=None, to_str_cols=None,
                           to_date_time_cols=None):
    """
    convert type of columns on request
    :return:
    """
    logging.info(f'some of the variables type wiill be modified on request')
    if to_cat_cols:
        data.loc[:, to_cat_cols] = data[to_cat_cols].astype('category')
    if to_float_cols:
        data.loc[:, to_float_cols] = data[to_float_cols].astype(float)
    if to_int_cols:
        data.loc[:, to_int_cols] = data[to_int_cols].astype(int)
    if to_str_cols:
        data.loc[:, to_str_cols] = data[to_str_cols].astype(str)
    if to_date_time_cols:
        for col in to_date_time_cols:
            data.loc[:, col] = pd.to_datetime(data[col])
    return data


def process_target_column(data, target_column, target_map):
    """
    converts the target column to bool
    :param data:
    :param target_column:
    :param target_map:
    :return:
    """
    logging.info(f'The target column {target_column} will be converted to bool')
    data[target_column] = data[target_column].map(target_map) #.astype(bool)
    return data


def process_raw_data(data, config):
    target_column = config['target_column']
    drop_columns = config['drop_columns']
    target_map = config['target_map']

    data.drop(columns=drop_columns, inplace=True)
    data = detect_and_fill_empty_cells(data)
    data = modify_type_of_columns(data, to_cat_cols=['gender', 'house_type'], to_date_time_cols=['lastVisit'])

    data = process_target_column(data, target_column, target_map)
    return data


def main():
    config = read_config(['preprocess'])
    input_file = os.path.join(config['input']['path'], config['input']['file_name'])
    output_file = os.path.join(config['output']['path'], config['output']['file_name'])

    data = pd.read_csv(input_file)
    data = process_raw_data(data, config)
    # parquet format preserves the dtypes unlike csv format
    os.makedirs(config['output']['path'], exist_ok=True)
    data.to_parquet(output_file)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    main()
