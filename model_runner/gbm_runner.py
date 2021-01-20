"""
The module `model_runner.gbm_runner` includes functions to run lightgbm and xgboost

"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb


def run_lgbm(X_train, y_train, X_valid, y_valid, X_test,
             features, features_categorical, train_params, **kwargs):
    """

    Run lightgbm model

    Parameters
    ----------
    X_train              : np.array, training data
    y_train              : np.array, training label
    X_val                : np.array, validation data
    y_val                : np.array, validation label
    X_test               : np.array, test data
    train_params         : dict, lightgbm training parameters
    features             : list (str), features
    features_categorical : list (str), categorical features
    kwargs               : other parameters needed for running lightgbm

    Returns
    -------
                                - lightgbm model
                                - validation data predictions
                                - test data predictions
                                - feature importance (gain and split) dataframe
    """
    num_boost_round = kwargs.get("num_boost_round", 1000)
    early_stopping_rounds = kwargs.get("early_stopping_rounds", 200)
    verbose_eval = kwargs.get("verbose_eval", 100)

    if train_params is None:
        train_params = {}

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
    importance_df["features"] = features
    importance_df["importance_gain"] = lgb_model.feature_importance(importance_type="gain")
    importance_df["importance_split"] = lgb_model.feature_importance(importance_type="split")

    return lgb_model, y_valid_preds, y_test_preds, importance_df


def run_xgb(X_train, y_train, X_valid, y_valid, X_test, features, train_params, **kwargs):
    """

    Run xgboost model
    See more at https://xgboost.readthedocs.io/en/latest/python/python_api.html

    Parameters
    ----------
    X_train              : np.array or scipy sparse matrix, training data
    y_train              : np.array, training label
    X_val                : np.array or scipy sparse matrix, validation data
    y_val                : np.array, validation label
    X_test               : np.array or scipy sparse matrix, test data
    train_params         : dict, lightgbm training parameters
    features             : list (str), features
    kwargs               : other parameters needed for running lightgbm

    Returns
    -------

    """
    num_boost_round = kwargs.get("num_boost_round", 1000)
    early_stopping_rounds = kwargs.get("early_stopping_rounds", 200)
    verbose_eval = kwargs.get("verbose_eval", 100)

    if train_params is None:
        train_params = {}

    X_train_xgb = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    X_valid_xgb = xgb.DMatrix(X_valid, label=y_valid, feature_names=features)
    X_test = xgb.DMatrix(X_test, feature_names=features)

    xgb_model = xgb.train(train_params, dtrain=X_train_xgb,
                          evals=[(X_train_xgb, "train"), (X_valid_xgb, "eval")],
                          num_boost_round=num_boost_round,
                          early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=verbose_eval)

    y_valid_preds = xgb_model.predict(X_valid_xgb)
    y_test_preds = xgb_model.predict(X_test)

    return xgb_model, y_valid_preds, y_test_preds
