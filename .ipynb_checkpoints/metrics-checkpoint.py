
import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(mean_squared_error(y_true, y_pred))