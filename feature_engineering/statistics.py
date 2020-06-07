"""
The module `feature_engineering.statistics` includes functions and classes to extract statistical features from data

"""

import numpy as np
import pandas as pd


def add_group_stats(df, cols, value, method, new_cols_out=False):
    """
    Refer from @author: plantsgo @address: https://www.kaggle.com/plantsgo

    Extract simple groupby statistical features

    :param df           : pd.DataFrame, input dataframe
    :param cols         : str or list of str, columns to groupby
    :param value        : str, column to be calculated on
    :param method       : str or list of str, aggregate function
    :param new_cols_out : boolean, whether output new column names (feature names), default=False

    :return             : df pd.DataFrame, input dataframe with features added
    """
    if not isinstance(value, str):
        raise NotImplementedError('Only support value to be string format (column name)')

    if isinstance(method, str):
        method = [method]

    method_options = ['nunique', 'count', 'mean', 'median', 'std', 'var', 'max', 'min', 'sum', 'skew', 'kurtosis']
    if any([True if m not in method_options else False for m in method]):
        raise AttributeError(f'Only support method in {method_options}.')

    if isinstance(cols, str):
        cols = [cols]

    new_cols = ['_'.join([*cols, value, m]) if m != 'count' else '_'.join([*cols, m]) for m in method]

    df_feats = pd.DataFrame(df.groupby(cols)[value].agg(method)).reset_index()
    df_feats.columns = cols + new_cols
    df = df.merge(df_feats, on=cols, how='left')
    if new_cols_out:
        return df, new_cols
    else:
        return df


def add_group_entropy(df, group, subgroup, value):
    """
    Refer from @author: owenzhang @address: https://github.com/owenzhang
    """
    if isinstance(group, str):
        group = [group]

    if isinstance(subgroup, list):
        full_group = group
        full_group.extend(subgroup)
    else:
        full_group = [*group, subgroup]
    
    gp_1 = df.groupby(full_group)[value].count().reset_index()
    gp_1.columns = full_group + ['subgroup_cnt']
    
    gp_2 = df.groupby(group)[value].count().reset_index()
    gp_2.columns = [group, 'cnt']
    
    gp_3 = gp_2.merge(gp_1, on=group, how='left')
    
    gp_3['entropy'] = -np.log(gp_3['subgroup_cnt'] / gp_3['cnt']) * gp_3['subgroup_cnt'] / gp_3['cnt']
    gp_3['entropy'].fillna(0, inplace=True)

    gp_4 = gp_3.groupby(group)['entropy'].sum().reset_index()
    gp_4.columns = ['_'.join([c, 'entropy']) for c in full_group]

    df = df.merge(gp_4, on=group, how='left')
    return df


def add_group_cumcount(df, cols):
    """ Extract sequential id in a group """
    if isinstance(cols, str):
        cols = [cols]
    df['_'.join([*cols, 'cumcount'])] = df.groupby(cols).cumcount() + 1
    return df


def add_group_value_count(df, cols, value):
    """ Extract value_counts() in each group """
    df_value_count = df.groupby(cols)[value].size().unstack().fillna(0.0).reset_index()
    df_value_count.columns = [cols[0]] + ['_'.join(c.split()).upper() + '_count' for c in df_value_count.columns.tolist()[1:]]
    df = df.merge(df_value_count, on=cols[0], how='left')
    return df


def merge_groupby_feat(df, agg_config, calc_diff=True):
    """

    Extract multiple groupby features
    
    :param df           : pd.DataFrame, input dataframe
    :param agg_func_map : dict, map from aggregation function to feature extraction function
    :param agg_config   : list, collection of categorical features to groupby and numerical features to aggregate
    :param calc_diff    : boolean, whether or not to calculate the difference between raw feature and agg feature
    
    :return df          : pd.DataFrame, input dataframe with extracted features added

    Examples
    --------
    >>> agg_config = [["ORGANIZATION_TYPE"], [("AMT_INCOME_TOTAL", "mean"),
    ...                                       ("AMT_INCOME_TOTAL", "median")])]
    
    """
    for agg_pair in agg_config:
        for agg_feat in agg_pair[1]:
            # agg_pair[0]: categorical features, agg_feat[0]: numerical features, agg_feat[1]: agg function
            if calc_diff:
                df, new_cols = add_group_stats(df, cols=agg_pair[0], value=agg_feat[0], method=agg_feat[1], new_cols_out=True)
                for c in new_cols:
                    df[f'{agg_feat[0]}_{c}_diff'] = df[agg_feat[0]] - df[c]
            else:
                df = add_group_stats(df, cols=agg_pair[0], value=agg_feat[0], method=agg_feat[1], new_cols_out=False)
    return df
