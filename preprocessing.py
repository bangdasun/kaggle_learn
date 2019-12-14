
import logging
import numpy as np
import pandas as pd

from typing import Union, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.exceptions import NotFittedError
from keras.preprocessing import text, sequence


class ValueEncoder(BaseEstimator, TransformerMixin):
    """ Replace old values with new values

    This is estimator is able to replace old values by specified values
    Currently only works on pandas.DataFrame

    """
    def __init__(self, old_to_new: dict):
        """
        Init use mapping from missing value to filling value

        :param old_to_new: format - {feature: [dtype, {old_value: new_value}]}
        """
        self.old_to_new = old_to_new

    def fit(self, df: pd.DataFrame, y: None):
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        for col, info in self.old_to_new.items():
            dtype, old_to_new = info[0], info[1]
            for old_val, new_val in old_to_new.items():
                if np.isnan(old_val):
                    logging.warning('Old value `np.nan` cannot be found using `==`'
                                    ' `use pd.DataFrame.fillna()` impute first.')
                df_processed.loc[df_processed[col] == old_val, col] = new_val
                if dtype == 'numerical':
                    df_processed[col] = df_processed[col].astype(np.float)
                elif dtype == 'categorical':
                    df_processed[col] = df_processed[col].astype(str)
                else:
                    logging.warning('dtype {} not defined'.format(dtype))

        return df_processed


class CategoricalLabelEncoder(OrdinalEncoder):
    """ Encode categorical features as integers

    This estimator is able to handle new categories in transforming testing data
    Currently only works on pandas.DataFrame.

    """

    def __init__(self, categories: [List[list], List[np.ndarray], str] = 'auto',
                 dtype=np.float64):
        """
        See
            https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#sklearn.preprocessing.OrdinalEncoder
        for detailed descriptions

        :param categories:
        :param dtype:
        """
        super().__init__(categories=categories, dtype=dtype)
        self.mapping = None
        self.columns = None

    def fit(self, df: pd.DataFrame, y: Union[np.ndarray, None] = None):
        """
        Fit the encoder

        :param df: input data (feature)
        :param y: label, not needed by default
        :return:
        """
        self.columns = list(df.columns)
        df_processed = df.copy()
        super().fit(df_processed, y)

        # get the existing categories for each categorical feature
        self.mapping = {self.columns[index]: list(categories) for index, categories in enumerate(self.categories_)}
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encoding categorical features, new categories are encoded as missing (-1)

        :param df:   input data (feature)
        :return:
        """
        df_processed = df.copy()

        try:
            return super().transform(df_processed)
        except ValueError:
            logging.warning('New categories are encoded as -1')
        new_categories_mask, df_processed = self._mask_new_categories(df)
        df_processed = super().transform(df_processed)

        # replace the new categories with -1 (indicating missing)
        for col_index in range(df_processed.shape[1]):
            df_processed[new_categories_mask.iloc[:, col_index].values, col_index] = -1

        return df_processed

    def _mask_new_categories(self, df: pd.DataFrame):
        """
        Get the mask for new categories

        :param df: input data (feature)
        :return:
        """
        new_categories_mask = df.copy()
        new_categories_mask.loc[:, :] = False
        df_processed = df.copy()

        if self.mapping is None:
            raise NotFittedError('CategoricalEncoder needs to be fitted to get the categories mapping!')

        for feat, categories in self.mapping.items():
            # get the mask for new categories
            new_categories_mask_col = ~(df_processed[feat].isin(categories))
            if np.sum(new_categories_mask_col) > 0:
                logging.warning('New categories found in {}!'.format(feat))

                # replace the new categories with existing category
                df_processed.loc[new_categories_mask_col, feat] = categories[0]
                new_categories_mask.loc[new_categories_mask_col, feat] = True
        return new_categories_mask, df_processed


def process_text_to_sequence(X_train, X_test, **kwargs):
    """

    Process text data (array) to equal length sequences use keras

    :param X_train : np.array with shape (m, )
    :param X_test  : np.array with shape (n, )
    :param kwargs  : other parameters needed
    :return:
    """

    max_features = kwargs.get('max_features', 10000)
    max_len = kwargs.get('max_len', 50)

    tokenizer = text.Tokenizer(num_words=max_features, lower=True, split=' ',
                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                               char_level=False, oov_token=None, document_count=0)
    tokenizer.fit_on_texts(list(X_train) + list(X_test))

    # process text to sequence
    X_train_sequence = tokenizer.texts_to_sequences(X_train)
    X_test_sequence = tokenizer.texts_to_sequences(X_test)

    # truncate / padding
    X_train_sequence_pad = sequence.pad_sequences(X_train_sequence, maxlen=max_len)
    X_test_sequence_pad = sequence.pad_sequences(X_test_sequence, maxlen=max_len)

    return X_train_sequence, X_test_sequence, X_train_sequence_pad, X_test_sequence_pad, tokenizer


def load_pretrained_word_embeddings(embedding_path, tokenizer, **kwargs):
    """

    Load pretrained word embeddings

    :param embedding_path : str, example: './embeddings/glove.840B.300d/glove.840B.300d.txt'
    :param tokenizer      : keras tokenizer, return from process_text_to_sequence
    :param kwargs         : other parameters needed
    :return:
    """

    embedding_size = kwargs.get('embedding_size', 300)
    max_features = kwargs.get('max_features', 10000)

    def _get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(_get_coefs(*o.strip().rsplit(' ')) for o in open(embedding_path))
    word_index = tokenizer.word_index
    num_words = min(max_features, len(word_index))
    embeddings_matrix = np.zeros((num_words, embedding_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embeddings_vector = embeddings_index.get(word)
        # oov or not
        if embeddings_vector is not None:
            embeddings_matrix[i] = embeddings_vector

    return embeddings_matrix

