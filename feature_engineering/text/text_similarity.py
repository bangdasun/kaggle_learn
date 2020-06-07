"""
The module `feature_engineering.text.text_similarity` includes functions and classes to calculate text similarity

References:
    https://github.com/h2oai/driverlessai-recipes/blob/master/transformers/nlp/text_similarity_transformers.py

"""

import nltk
import editdistance
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class NGramsSimilarityTransformer(TransformerMixin):
    """ Text similarity based on n-gram """

    def __init__(self, ngrams=1, method='ngrams'):
        self.ngrams = ngrams
        self.method = method

    def _score(self, text_set_1, text_set_2):
        score = -1
        if self.method == 'ngrams':
            score = len(text_set_1.intersection(text_set_2))
        elif self.method == 'jaccard':
            score = len(text_set_1.intersection(text_set_2)) / len(text_set_1.union(text_set_2))
        elif self.method == 'dice':
            score = 2 * len(text_set_1.intersection(text_set_2)) / (len(text_set_1) + len(text_set_2))
        else:
            raise NotImplementedError(f'Not support method {self.method}.')
        return score

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = []
        if isinstance(X, pd.DataFrame):
            X = X.values

        text_arr_1 = X[:, 0]
        text_arr_2 = X[:, 1]
        for idx in range(X.shape[0]):
            try:
                text_set_1 = set(nltk.ngrams(str(text_arr_1[idx]).lower().split(), self.ngrams))
                text_set_2 = set(nltk.ngrams(str(text_arr_2[idx]).lower().split(), self.ngrams))
                output.append(self._score(text_set_1, text_set_2))
            except Exception as e:
                output.append(-1)
                print(f'Exception raised:\n{e}')
        return np.array(output).reshape(-1, 1)


class TermEditDistanceTransformer(TransformerMixin):
    """ Text similarity based on word edit distance """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = []
        if isinstance(X, pd.DataFrame):
            X = X.values

        text_arr_1 = X[:, 0]
        text_arr_2 = X[:, 1]
        for idx in range(X.shape[0]):
            try:
                term_lst_1 = str(text_arr_1[idx]).lower().split()
                term_lst_2 = str(text_arr_2[idx]).lower().split()
                output.append(editdistance.eval(term_lst_1, term_lst_2))
            except Exception as e:
                output.append(-1)
                print(f'Exception raised:\n{e}')
        return np.array(output).reshape(-1, 1)
