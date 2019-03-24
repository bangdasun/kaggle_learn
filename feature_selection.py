
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


def get_null_feature_importance(X_train, y_train, features, features_categorical, train_params,
                                shuffle=True, seed=None, num_runs=80, method='lgb',  **kwargs):
    """
    Get feature null importance based on lightgbm

    :param X_train              : np.array, training data
    :param y_train              : np.array, training label
    :param features             : list (str), features
    :param features_categorical : list (str), categorical features
    :param train_params         : dict, lightgbm training parameters
    :param shuffle              : boolean, whether or not to shuffle label
    :param seed                 : int, random seed for label shuffling
    :param num_runs             : int, number of runs to get the null importance distribution
    :param method               : str, currently only support lightgbm
    :param kwargs               : other parameters needed for running lightgbm
    :return:
    """
    if method != 'lgb':
        raise NotImplementedError('Currently only support method = "lgb".')

    actual_importance_df = lgbm_permutation_importance(X_train, y_train,
                                                       features, features_categorical, train_params,
                                                       shuffle=False, seed=seed, **kwargs)

    null_importance_df = pd.DataFrame()
    for i in range(num_runs):
        importance_df = lgbm_permutation_importance(X_train, y_train,
                                                    features, features_categorical, train_params,
                                                    shuffle=shuffle, seed=seed, **kwargs)
        importance_df['run'] = i + 1
        null_importance_df = pd.concat([null_importance_df, importance_df], axis=0)

        if i % 10 == 0:
            print('Running epoch {} . . .'.format(i + 1))

    feature_scores = []
    for feat in features:
        feat_null_importance_gain = null_importance_df.loc[null_importance_df['features'] == feat, 'importance_gain'].values
        feat_act_importance_gain = actual_importance_df.loc[actual_importance_df['features'] == feat, 'importance_gain'].mean()
        gain_score = np.log(1e-10 + feat_act_importance_gain / (1 + np.percentile(feat_null_importance_gain, 75)))

        feat_null_importance_split = null_importance_df.loc[null_importance_df['features'] == feat, 'importance_split'].values
        feat_act_importance_split = actual_importance_df.loc[actual_importance_df['features'] == feat, 'importance_split'].mean()
        split_score = np.log(1e-10 + feat_act_importance_split / (1 + np.percentile(feat_null_importance_split, 75)))
        feature_scores.append((feat, gain_score, split_score))

    return pd.DataFrame(feature_scores, columns=['features', 'gain_score', 'split_score'])
