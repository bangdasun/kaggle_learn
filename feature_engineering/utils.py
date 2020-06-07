
import pandas as pd
from sklearn.base import TransformerMixin


class FeatureSelector(TransformerMixin):
    """ Select features from pandas dataframe """
    def __init__(self, features):
        if isinstance(features, list):
            self.features = features
        elif isinstance(features, str):
            self.features = [features]
        else:
            raise ValueError(f'`feature` cannot be in type: {type(features)}')

    def fit(self, df: pd.DataFrame, y=None):
        return self

    def transform(self, df: pd.DataFrame):
        return df[self.features]
