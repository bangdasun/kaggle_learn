
import os
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, Dense, Embedding, GRU, LSTM
from keras.layers import GlobalMaxPooling1D, Conv1D, Dropout
from keras import optimizers


def make_rnn_classifier(input_size, output_size, embedding_matrix, **kwargs):
    """

    Template for build RNN based classification model, can be used for
    - text classification

    Model structure refers from:
    https://www.kaggle.com/konohayui/bi-gru-cnn-poolings

    Parameters
    ----------
    input_size       : tuple, input size, i.e. sequence length, example: (70, )
    output_size      : int, output size (number of classes)
    embedding_matrix : array, embedding lookup matrix
    kwargs           : other parameters needed

    Returns
    -------

    """
    max_features = kwargs.get("max_features", 10000)
    embedding_size = kwargs.get("embedding_size", 300)
    lstm_units = kwargs.get("lstm_units", 60)
    conv_filters = kwargs.get("conv_filters", 32)
    conv_kernel_size = kwargs.get("conv_kernel_size", 3)
    dropout_rate = kwargs.get("dropout_rate", 0.2)
    dense_units = kwargs.get("dense_units", 64)
    loss = kwargs.get("loss", "binary_crossentropy")
    optimizer = kwargs.get("optimizer", optimizers.Adam())
    metrics = kwargs.get("metrics", ["accuracy"])

    # input layer
    input = Input(shape=input_size)
    x = input

    # hidden layers
    x = Embedding(input_dim=max_features, output_dim=embedding_size, trainable=False,
                  weights=[embedding_matrix])(x)
    x = LSTM(units=lstm_units, return_sequences=True)(x)
    x = Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, padding="valid")(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(units=dense_units, activation="relu")(x)
    x = Dropout(rate=dropout_rate)(x)

    # output layer, activation should changed to softmax for multiclass
    x = Dense(units=output_size, activation="sigmoid")(x)

    model = Model(inputs=[input], outputs=x)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model

