
import os
import psutil
import time
import logging
import random
import numpy as np
import pandas as pd
import torch

from contextlib import contextmanager
from collections import OrderedDict

@contextmanager
def timer(name):
    """ @author: lopuhin @address https://www.kaggle.com/lopuhin/ """
    start_time = time.time()
    yield
    print('[{} done in {:.3f} s.]'.format(name, time.time() - start_time))
    

def memory_usage():
    """ @author: rdizzl3 @address: https://www.kaggle.com/rdizzl3 """
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    print('[Memory used {:.2f} GB]'.format(memory_use))


def reduce_memory_usage(df):
    """ @author: gemartin @address: https://www.kaggle.com/gemartin """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    dtypes = {}
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                    dtypes[col] = np.int8
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                    dtypes[col] = np.int16
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                    dtypes[col] = np.int32
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                    dtypes[col] = np.int64
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                    dtypes[col] = np.float16
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                    dtypes[col] = np.float32
                else:
                    df[col] = df[col].astype(np.float64)
                    dtypes[col] = np.float64
        else: 
            df[col] = df[col].astype('category')
            dtypes[col] = 'category'

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.2f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df, dtypes


def get_logger(logger_name):
    """
    Get logger to log information during code running

    :param logger_name: str, title of log file
    :return logger    : logger object

    Examples
    --------
    >>> logger_main = get_logger('main')
    >>> logger_main.info('Start . . .')
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{}.log'.format(logger_name))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(levelname)s]%(asctime)s:%(name)s:%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def set_torch_seed(seed=2019):
    """ Set random seed when using pytorch """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def sort_dict_values(dictionary, ascending=False):
    """ Sort dictionary items by its values """
    if ascending:
        output = OrderedDict(sorted(dictionary.items(), reverse=False, key=lambda x: x[1]))
    else:
        output = OrderedDict(sorted(dictionary.items(), reverse=True, key=lambda x: x[1]))
    return output
