
import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(mean_squared_error(y_true, y_pred))


def multilabel_f1(y_true, y_pred):
    """

    Refer from @author: onodera @address: https://www.kaggle.com/onodera

    Multi-labels f1 score

    Parameters
    ----------
    y_true    : iterable object (list, array)
    y_pred    : iterable object (list, array)

    Returns
    -------

    Examples
    --------
    >>> y_true = [1, 2, 3]
    >>> y_pred = [2, 3]
    >>> multilabel_f1(y_true, y_pred)
    ... 0.8
    """
    y_true, y_pred = set(y_true), set(y_pred)
    precision = np.sum([1 for i in y_pred if i in y_true]) / len(y_pred)
    recall = np.sum([1 for i in y_true if i in y_pred]) / len(y_true)
    if precision + recall == 0:
        return 0
    else:
        return (2 * precision * recall) / (precision + recall)
