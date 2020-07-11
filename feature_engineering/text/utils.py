
import numpy as np
from sklearn.decomposition import TruncatedSVD


def get_word_weight(weight_filename, alpha=1e-3):
    """ Get word term-frequency from large corpus file """
    # when the parameter makes no sense, use unweighted
    if alpha <= 0:
        alpha = 1.0

    words_weight = {}
    with open(weight_filename) as f:
        lines = f.readlines()

    # total term-frequency
    N = 0
    for word_tf in lines:
        word_tf = word_tf.strip()
        if len(word_tf) == 0:
            continue
        word_tf = word_tf.split()
        if len(word_tf) == 2:
            word = word_tf[0]
            tf = word_tf[1]
            words_weight[word] = float(tf)
            N += float(tf)
        else:
            print('{} is not a valid (word, termfrequency) record'.format(word_tf))

    # normalize weights by alpha and N
    for word, tf in words_weight.items():
        words_weight[word] = alpha / (alpha + tf / N)

    return words_weight


def lookup_pretrained_index(words_pretrained_index, word):
    word = word.lower()
    if len(word) > 1 and word[0] == '#':
        word = word.replace("#", "")

    if word in words_pretrained_index:
        return words_pretrained_index[word]
    elif 'UUUNKKK' in words_pretrained_index:
        return words_pretrained_index['UUUNKKK']
    else:
        return len(words_pretrained_index) - 1


def get_pretrained_index_weight(word_pretrained_index, words_weight):
    """ Get the map from word index in pretrained embeddings and weights """
    index_weights = {}
    for word, idx in word_pretrained_index.items():
        if word in words_weight:
            index_weights[idx] = words_weight[word]
        else:
            index_weights[idx] = 1.0
    return index_weights


def get_sentence_pretrained_index(sentences, words_pretrained_index):
    """
    Given a list of sentences, output array of word indices
    that can be fed into the algorithms.
    Since sentences have different length, 0 will be padded at
    the end for sentence length less than max length

    Parameters
    ----------
    sentences
    words_pretrained_index

    Returns
    -------
    word_index_sentence, mask.
     word_index_sentence[i, :] is the word indices in sentence i
     mask[i,:] is the mask for sentence i (0 means no word at the location)

    """
    def get_sequence(sentence, words_pretrained_index):
        return [lookup_pretrained_index(words_pretrained_index, word) for word in sentence.split()]

    sequence = [get_sequence(sentence, words_pretrained_index) for sentence in sentences]
    word_index_sentence, mask = pad_sequences(sequence)
    return word_index_sentence, mask


def pad_sequences(sequences):
    """ Padding 0 to sequences that shorter than max length """
    lengths = [len(s) for s in sequences]
    n_samples = len(sequences)
    maxlen = np.max(lengths)

    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, sentence in enumerate(sequences):
        x[idx, :lengths[idx]] = sentence
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype='float32')
    return x, x_mask


def get_word_weights_sequence(sequences, mask, index_weights):
    """ Get word weights for sentences """
    weight = np.zeros(sequences.shape).astype('float32')

    for i in range(sequences.shape[0]):
        for j in range(sequences.shape[1]):
            if mask[i, j] > 0 and sequences[i, j] >= 0:
                weight[i, j] = index_weights[sequences[i, j]]

    weight = np.asarray(weight, dtype='float32')
    return weight


def get_weighted_average(embedding_matrix, word_index_sentence, weights):
    """ Compute the weighted average word embeddings """
    n_samples = word_index_sentence.shape[0]
    embedding_matrix_avg = np.zeros((n_samples, embedding_matrix.shape[1]))
    for i in range(n_samples):
        total_weights = np.count_nonzero(weights[i, :])
        embedding_matrix_avg[i, :] = weights[i, :].dot(embedding_matrix[word_index_sentence[i, :], :]) / total_weights
    return embedding_matrix_avg


def remove_pc(X, n_components_rm=1, **kwargs):
    """ Remove the projection on the principal components """
    n_components = kwargs.get('n_components', 1)
    random_state = kwargs.get('random_state', 2020)
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    svd.fit(X)
    pc = svd.components_

    if n_components_rm == 1:
        X_processed = X - X.dot(pc.transpose()) * pc
    else:
        X_processed = X - X.dot(pc.transpose()).dot(pc)
    return X_processed


def sif_embeddings(embeddings_matrix, word_index_sentence, weights):
    """ Get SIF embeddings """
    embeddings_matrix_avg = get_weighted_average(embeddings_matrix, word_index_sentence, weights)
    embeddings_matrix_avg[np.isnan(embeddings_matrix_avg)] = 0.0
    embeddings_matrix_avg_rm_pc = remove_pc(embeddings_matrix_avg)
    return embeddings_matrix_avg_rm_pc
