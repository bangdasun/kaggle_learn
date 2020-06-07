"""
The module `feature_engineering.text.text_meta` includes functions and classes to extract meta features from text
    - number of characters
    - number of words
    - number of unique words
    ...

"""

import re
import string
import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from kaggle_learn.utils import convert_to_numpy_1d_array

def text_len(x):
    """ Extract string length """
    try:
        return len(str(x))
    except Exception as e:
        print(f'Exception raised:\n{e}')
        return 0


def count_word(x, sep=None):
    """ Extract number of words in a string """
    try:
        return len(str(x).split(sep))
    except Exception as e:
        print(f'Exception raised:\n{e}')
        return 0


def count_unique_word(x, sep=None):
    """ Extract number of unique words in a string """
    try:
        return len(set(str(x).split(sep)))
    except Exception as e:
        print(f'Exception raised:\n{e}')
        return 0


def count_symbol(x, symbol=None):
    """ Extract number of symbol in a string """
    try:
        return str(x).count(symbol)
    except Exception as e:
        print(f'Exception raised:\n{e}')
        return 0


def count_capital_letters(x):
    """ Extract number of captial letters in a string """
    try:
        return sum([s.isupper() for s in list(str(x))])
    except Exception as e:
        print(f'Exception raised:\n{e}')
        return 0


def count_common_words(x, y):
    """ Extract number of common word between two strings """
    try:
        words, cnt = x.split(), 0
        for w in words:
            if y.find(w) >= 0:
                cnt += 1
        return cnt
    except Exception as e:
        print(f'Exception raised:\n{e}')
        return 0


def search_symbol(x, symbol=None):
    """ Search symbol and return first match place """
    result = re.search(symbol, str(x))
    try:
        return result.start()
    except Exception as e:
        print(f'Exception raised:\n{e}')
        return -1


def extract_capital_letters(x):
    """ Extract capital letters from string """
    try:
        return ''.join([s for s in x if s.isupper()])
    except Exception as e:
        print(f'Exception raised:\n{e}')
        return ''


class TextMetaTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)


class TextLenTransformer(TextMetaTransformer):

    def transform(self, X):
        X = convert_to_numpy_1d_array(X)
        output = list(map(lambda x: len(x), X))
        return np.array(output).reshape(-1, 1)


class CountWordTransform(TextMetaTransformer):
    unique = True

    def transform(self, X):
        X = convert_to_numpy_1d_array(X)
        if self.unique:
            output = list(map(lambda x: len(set(x.split())), X))
        else:
            output = list(map(lambda x: len(x.split()), X))
        return np.array(output).reshape(-1, 1)


class CountSymbolTransformer(TextMetaTransformer):

    def __init__(self, symbol):
        self.symbol = symbol

    def transform(self, X):
        X = convert_to_numpy_1d_array(X)
        output = list(map(lambda x: x.count(self.symbol), X))
        return np.array(output).reshape(-1, 1)


class CountUpperCaseTransformer(TextMetaTransformer):

    def transform(self, X):
        X = convert_to_numpy_1d_array(X)
        output = list(map(lambda x: sum([s.isupper() for s in x]), X))
        return np.array(output).reshape(-1, 1)


class CountUpperWordCaseTransformer(TextMetaTransformer):

    def transform(self, X):
        X = convert_to_numpy_1d_array(X)
        output = list(map(lambda x: sum([s.isupper() for s in x.split()]), X))
        return np.array(output).reshape(-1, 1)


class CountNumericWordTransformer(TextMetaTransformer):

    def transform(self, X):
        X = convert_to_numpy_1d_array(X)
        output = list(map(lambda x: sum([s.isnumeric() for s in x.split()]), X))
        return np.array(output).reshape(-1, 1)


class CountPunctuationsTransformer(TextMetaTransformer):
    punctuations = string.punctuation

    def transform(self, X):
        X = convert_to_numpy_1d_array(X)
        output = list(map(lambda x: len([s for s in x if s in self.punctuations]), X))
        return np.array(output).reshape(-1, 1)


class WordPositionTransformer(TextMetaTransformer):

    def __init__(self, symbol):
        self.symbol = symbol

    def _get_position(self, x):
        result = re.search(self.symbol, x)
        try:
            return result.start()
        except Exception as e:
            print(f'Exception raised:\n{e}')
            return -1

    def transform(self, X):
        X = convert_to_numpy_1d_array(X)
        output = list(map(self._get_position, X))
        return np.array(output).reshape(-1, 1)
