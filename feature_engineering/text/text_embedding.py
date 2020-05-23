"""
The module `feature_engineering.text.text_embedding` includes functions and classes to get text embeddings

"""

import numpy as np

from sklearn.base import TransformerMixin
from kaggle_learn.preprocessing.text import process_text_to_sequence, load_pretrained_word_embeddings
from kaggle_learn.feature_engineering.text.utils import (
    get_word_weight,
    lookup_pretrained_index,
    get_pretrained_index_weight,
    get_sentence_pretrained_index,
    pad_sequences,
    get_word_weights_sequence,
    get_weighted_average,
    remove_pc,
    sif_embeddings
)


class SIFEmbeddingTransformer(TransformerMixin):
    """
    Extract embeddings for sentences using SIF

    References
        - https://openreview.net/pdf?id=SyK00v5xx
        - https://github.com/PrincetonML/SIF
    """
    def __init__(self, word_weight_filename,
                 word_embedding_filename,
                 weight_para=1e-3,
                 embedding_size=300,
                 max_features=10000,
                 max_len=100):
        self.word_weight_filename = word_weight_filename
        self.word_embedding_filename = word_embedding_filename
        self.weight_para = weight_para
        self.embedding_size = embedding_size
        self.max_features = max_features
        self.max_len = max_len

    def _load_embedding(self, X, word_embedding_file):
        output_sequence = process_text_to_sequence(X, np.array([]), max_features=self.max_features, max_len=self.max_len)
        output_embedding = load_pretrained_word_embeddings(
            word_embedding_file,
            output_sequence['tokenizer'],
            embedding_size=self.embedding_size,
            max_features=self.max_features,
            max_len=self.max_len
        )
        return output_embedding

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        word_embedding = self._load_embedding(X, self.word_embedding_filename)
        embeddings_matrix = word_embedding['embeddings_matrix']
        word_index = word_embedding['word_index']

        # get word_to_weights dictionary
        word_weight = get_word_weight(self.word_weight_filename, self.weight_para)
        # get word_index_to_weights
        index_weights = get_pretrained_index_weight(word_index, word_weight)
        # get word_index as sentences
        word_index_sentence, mask = get_sentence_pretrained_index(list(X), word_index)
        # get weights as sequences
        weights = get_word_weights_sequence(word_index_sentence, mask, index_weights)
        # get weighted average embedding and remove principal component
        sentence_embeddings = sif_embeddings(embeddings_matrix, word_index_sentence, weights)

        return sentence_embeddings
