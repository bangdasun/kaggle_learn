
import numpy as np
import pandas as pd
import lightgbm as lgb
from feature_importance.ensemble_feature_importance import lgbm_permutation_importance


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
