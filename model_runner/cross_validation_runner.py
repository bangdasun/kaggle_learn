
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


_builtin_classifier = {
    'lr': LogisticRegression,
    'knn': KNeighborsClassifier,
    'svm': SVC,
    'rf': RandomForestClassifier,
    'mlp': MLPClassifier
}


def run_classifier_kfold_cv(X_train, y_train, X_test=None, classifier='rf', classifier_params=None, **kwargs):
    """

    Run classification model with K folds cross validation
    If test data is given, the final prediction (probabilities) for test data are averaged over all folds

    :param X_train          : np.array, training data
    :param y_train          : np.array, training label
    :param X_test           : np.array, test data
    :param classifier       : str, classifier name
    :param classifier_params: dict, classifier training parameters
    :param kwargs           : other parameters needed
    :return:
    """

    folds = kwargs.get('n_folds', 5)
    shuffle = kwargs.get('shuffle', True)
    random_state = kwargs.get('random_state', 2019)
    kfold = KFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
    oof_preds = np.zeros(y_train.shape[0])
    oof_preds_proba = np.zeros(y_train.shape[0])
    output = {}
    clf_list = []

    if classifier_params is None:
        classifier_params = {}
    if not isinstance(classifier_params, dict):
        raise ValueError('Argument `classifier_params` has to be dictionary or None by default.')

    for n_fold, (trn_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        X_train_, X_val_ = X_train[trn_idx], X_train[val_idx]
        y_train_, y_val_ = y_train[trn_idx], y_train[val_idx]
        clf = _builtin_classifier[classifier](**classifier_params)
        clf.fit(X_train_, y_train_)
        oof_preds[val_idx] = clf.predict(X_val_)
        oof_preds_proba[val_idx] = clf.predict_proba(X_val_)
        clf_list.append(clf)

    # save out-of-fold predictions
    output['oof_preds'] = oof_preds
    output['oof_preds_proba'] = oof_preds_proba

    # run prediction on test data
    if X_test is not None:
        test_preds_proba = np.zeros(X_test.shape[0])
        for clf_ in clf_list:
            test_preds_proba += clf_.predict_proba(X_test) / folds
        output['test_preds_proba'] = test_preds_proba

    return output
