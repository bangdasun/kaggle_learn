"""
The module `preprocessing.text` includes functions and classes to preprocess text data
    - tokenize and process unstructued text data into sequence data
    - load word embeddings
    - normalize text

"""

import re
import string
import warnings
import numpy as np

from kaggle_learn.utils import convert_to_numpy_1d_array
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer, SnowballStemmer
from keras.preprocessing import text, sequence

warnings.filterwarnings('ignore')


def process_text_to_sequence(X_train, X_test, **kwargs):
    """
    Process text data (array) to equal length sequences use keras

    Parameters
    ----------
    X_train : np.array with shape (m, )
    X_test  : np.array with shape (n, )
    kwargs  : other parameters needed

    Returns
    -------

    """
    max_features = kwargs.get('max_features', 10000)
    max_len = kwargs.get('max_len', 50)

    tokenizer = text.Tokenizer(num_words=max_features, lower=True, split=' ',
                               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                               char_level=False, oov_token=None)
    tokenizer.fit_on_texts(list(X_train) + list(X_test))

    # process text to sequence
    X_train_sequence = tokenizer.texts_to_sequences(X_train)
    X_test_sequence = tokenizer.texts_to_sequences(X_test)

    # truncate / padding
    X_train_sequence_pad = sequence.pad_sequences(X_train_sequence, maxlen=max_len)
    X_test_sequence_pad = sequence.pad_sequences(X_test_sequence, maxlen=max_len)

    return dict(X_train_sequence=X_train_sequence,
                X_test_sequence=X_test_sequence,
                X_train_sequence_pad=X_train_sequence_pad,
                X_test_sequence_pad=X_test_sequence_pad,
                tokenizer=tokenizer)


def load_pretrained_word_embeddings(embedding_path, tokenizer, **kwargs):
    """
    Load pretrained word embeddings

    Parameters
    ----------
    embedding_path : str, example: './embeddings/glove.840B.300d/glove.840B.300d.txt'
    tokenizer      : keras tokenizer, return from process_text_to_sequence
    kwargs         : other parameters needed

    Returns
    -------

    """
    embedding_size = kwargs.get('embedding_size', 300)
    max_features = kwargs.get('max_features', 10000)

    def _get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(_get_coefs(*o.strip().rsplit(' ')) for o in open(embedding_path, encoding='utf-8', errors='ignore'))
    word_index = tokenizer.word_index
    num_words = min(max_features, len(word_index))
    embeddings_matrix = np.zeros((num_words, embedding_size))
    for word, i in word_index.items():
        i -= 1
        if i >= max_features:
            continue
        embeddings_vector = embeddings_index.get(word)
        # oov or not
        if embeddings_vector is not None:
            embeddings_matrix[i] = embeddings_vector

        # make zero as start
        word_index[word] -= 1

    return dict(embeddings_matrix=embeddings_matrix,
                word_index=word_index)


def clean_regex(x, pattern_mapping):
    """ Clean text based on pre-defined regular expressions """
    for old_pattern, new_pattern in pattern_mapping.items():
        x = re.sub(old_pattern, new_pattern, x)
    return x


def separate_punctuation(x, punctuations):
    """ Add space around pre-defined punctuations """
    for p in punctuations:
        x = x.replace(p, f' {p} ')
    return x


def remove_words(x, words):
    """ Remove pre-defined words """
    return ' '.join([w for w in x.split() if w not in words])


def strip_space(x):
    """ Strip extra spaces """
    return ' '.join([s.strip() for s in x.split()])


def stemming(x, method='snowball'):
    """ Apply stemming on text """
    __builtin_stemmer = {'snowball': SnowballStemmer, 'porter': PorterStemmer}
    stemmer = __builtin_stemmer[method]('english')
    return ' '.join([stemmer.stem(w) for w in x.split()])


class RegexCleaner(TransformerMixin):
    """ Clean text based on pre-defined regular expressions """
    def __init__(self, pattern_mapping=None, strip_space=True):
        self.pattern_mapping = {} if pattern_mapping is None else pattern_mapping
        self.strip_space = strip_space

    def _replace_text(self, text):
        for old_pattern, new_pattern in self.pattern_mapping.items():
            text = re.sub(old_pattern, new_pattern, text)

        if self.strip_space:
            text = ' '.join([s.strip() for s in text.split()])

        return text

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = convert_to_numpy_1d_array(X)
        output = list(map(self._replace_text, X))
        return np.array(output).reshape(-1, 1)


class PunctuationSeparator(TransformerMixin):
    """ Add space around pre-defined punctuations """
    def __init__(self, punctuations=None, strip_space=True):
        self.punctuations = string.punctuation if punctuations is None else punctuations
        self.strip_space = strip_space

    def _add_separator(self, text):
        for p in self.punctuations:
            text = text.replace(p, f' {p} ')

        if self.strip_space:
            text = ' '.join([s.strip() for s in text.split()])

        return text

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = convert_to_numpy_1d_array(X)
        output = list(map(self._add_separator, X))
        return np.array(output).reshape(-1, 1)


class StopwordsRemover(TransformerMixin):
    """ Remove stopwords """
    def __init__(self, stop_words=None):
        self.stop_words = stopwords.words('english') if stop_words is None else stop_words

    def _remove_words(self, text):
        return ' '.join([w for w in text.split() if w not in self.stop_words])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = convert_to_numpy_1d_array(X)
        output = list(map(self._remove_words, X))
        return np.array(output).reshape(-1, 1)


class Stemmer(TransformerMixin):
    """ Apply stemming on text """
    __builtin_stemmer = {'snowball': SnowballStemmer, 'porter': PorterStemmer}

    def __init__(self, method='snowball'):
        self.stemmer = self.__builtin_stemmer[method]('english')

    def _stemming(self, text):
        return ' '.join([self.stemmer.stem(w) for w in text.split()])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = convert_to_numpy_1d_array(X)
        output = list(map(self._stemming, X))
        return np.array(output).reshape(-1, 1)
