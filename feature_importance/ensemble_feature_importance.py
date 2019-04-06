
import numpy as np
import pandas as pd
import lightgbm as lgb


def lgbm_permutation_importance(X_train, y_train, features, features_categorical, train_params,
                                shuffle=True, seed=None, **kwargs):
    """
    Get feature permutation importance based on lightgbm

    :param X_train              : np.array, training data
    :param y_train              : np.array, training label
    :param features             : list (str), features
    :param features_categorical : list (str), categorical features
    :param train_params         : dict, lightgbm training parameters
    :param shuffle              : boolean, whether or not to shuffle label
    :param seed                 : int, random seed for label shuffling
    :param kwargs               : other parameters needed for running lightgbm
    :return:
    """
    if shuffle:
        np.random.seed(seed)
        y_train = np.random.permutation(y_train)

    if 'num_boost_round' in kwargs:
        num_boost_round = kwargs['num_boost_round']
    else:
        num_boost_round = 1000

    X_train_lgb = lgb.Dataset(X_train, y_train, free_raw_data=False, silent=True)
    lgbm_model = lgb.train(params=train_params, train_set=X_train_lgb,
                           feature_name=features, categorical_feature=features_categorical,
                           num_boost_round=num_boost_round)
    importance_df = pd.DataFrame()
    importance_df['features'] = features
    importance_df['importance_gain'] = lgbm_model.feature_importance(importance_type='gain')
    importance_df['importance_split'] = lgbm_model.feature_importance(importance_type='split')
    return importance_df
