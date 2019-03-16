
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

def run_lgbm(X_train, y_train, X_valid, y_valid, X_test,
             features, features_categorical, train_params, **kwargs):
    """

    Run lightgbm model

    :param X_train              : np.array, training data
    :param y_train              : np.array, training label
    :param X_val                : np.array, validation data
    :param y_val                : np.array, validation label
    :param X_test               : np.array, test data
    :param train_params         : dict, lightgbm training parameters
    :param features             : list (str), features
    :param features_categorical : list (str), categorical features
    :param kwargs               : other parameters needed for running lightgbm
    :return:
                                - lightgbm model
                                - validation data predictions
                                - test data predictions
                                - feature importance (gain and split) dataframe
    """

    if 'num_boost_round' in kwargs:
        num_boost_round = kwargs['num_boost_round']
    else:
        num_boost_round = 1000

    if 'early_stopping_rounds' in kwargs:
        early_stopping_rounds = kwargs['early_stopping_rounds']
    else:
        early_stopping_rounds = 200

    if 'verbose_eval' in kwargs:
        verbose_eval = kwargs['verbose_eval']
    else:
        verbose_eval = 100

    X_train_lgb = lgb.Dataset(X_train, y_train, feature_name=features, categorical_feature=features_categorical)
    X_valid_lgb = lgb.Dataset(X_valid, y_valid, feature_name=features, categorical_feature=features_categorical)

    lgb_model = lgb.train(train_params, train_set=X_train_lgb,
                          valid_sets=[X_train_lgb, X_valid_lgb],
                          num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=verbose_eval)

    y_valid_preds = lgb_model.predict(X_valid, num_iteration=lgb_model.best_iteration)
    y_test_preds = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

    importance_df = pd.DataFrame()
    importance_df['features'] = features
    importance_df['importance_gain'] = lgb_model.feature_importance(importance_type='gain')
    importance_df['importance_split'] = lgb_model.feature_importance(importance_type='split')

    return lgb_model, y_valid_preds, y_test_preds, importance_df
