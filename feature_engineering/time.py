"""
The module `feature_engineering.time` includes functions and classes to process and extract datetime features

References:
    https://github.com/h2oai/driverlessai-recipes/blob/master/transformers/nlp/text_similarity_transformers.py

"""

import numpy as np
import pandas as pd


def convert_datetime(df, col, unix_time=False):
    """
    Convert datetime columns to pandas datetime format

    :param df        : pd.DataFrame, input dataframe
    :param col       : str, column name of datetime
    :param unix_time : boolean, whether the column is in unix time format (seconds from 1970-01-01)

    :return          : pd.Series, converted datetime columns
    """
    if not isinstance(col, str):
        raise ValueError(f'{col} is not a string')

    if unix_time:
        return pd.to_datetime(df[col], unit='s')
    else:
        return pd.to_datetime(df[col])


def extract_datetime_components(df, cols, drop_raw=False, unix_time=False):
    """
    Extract datetime components from datetime columns

    :param df        : pd.DataFrame, input dataframe
    :param cols      : str or list of str, column names of datetime
    :param drop_raw  : boolean, whether to drop the raw datetime column after extraction
    :param unix_time : boolean, whether the column is in unix time format (seconds from 1970-01-01)

    :return df       : pd.DataFrame, input dataframe with extracted features added
    """
    if isinstance(cols, str):
        cols = [cols]

    for col in cols:
        df[col] = convert_datetime(df, col, unix_time)
        df[col + '_year']   = df[col].dt.year.astype(np.int16)
        df[col + '_month']  = df[col].dt.month.astype(np.int8)
        df[col + '_day']    = df[col].dt.day.astype(np.int8)
        df[col + '_hour']   = df[col].dt.hour.astype(np.int8)
        df[col + '_minute'] = df[col].dt.minute.astype(np.int8)
        df[col + '_second'] = df[col].dt.second.astype(np.int8)

    if drop_raw:
        df.drop(labels=cols, axis=1, inplace=True)

    return df
