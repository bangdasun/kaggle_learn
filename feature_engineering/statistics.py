

import numpy as np
import pandas as pd


def add_group_nunique(df, cols, cname, value):
    """ @auther: plantsgo @address: https://www.kaggle.com/plantsgo
    """
    df_nunique = pd.DataFrame(df.groupby(cols)[value].nunique()).reset_index()
    df_nunique.columns = cols + [cname]
    df = df.merge(df_nunique, on=cols, how='left')
    return df


def add_group_count(df, cols, cname, value):
    """ @auther: plantsgo @address: https://www.kaggle.com/plantsgo
    """
    df_count = pd.DataFrame(df.groupby(cols)[value].count()).reset_index()
    df_count.columns = cols + [cname]
    df = df.merge(df_count, on=cols, how='left')
    return df


def add_group_mean(df, cols, cname, value):
    """ @auther: plantsgo @address: https://www.kaggle.com/plantsgo
    """
    df_count = pd.DataFrame(df.groupby(cols)[value].mean()).reset_index()
    df_count.columns = cols + [cname]
    df = df.merge(df_count, on=cols, how='left')
    return df


def add_group_median(df, cols, cname, value):
    """ @auther: plantsgo @address: https://www.kaggle.com/plantsgo
    """
    df_count = pd.DataFrame(df.groupby(cols)[value].median()).reset_index()
    df_count.columns = cols + [cname]
    df = df.merge(df_count, on=cols, how='left')
    return df


def add_group_std(df, cols, cname, value):
    """ @auther: plantsgo @address: https://www.kaggle.com/plantsgo
    """
    df_count = pd.DataFrame(df.groupby(cols)[value].std()).reset_index()
    df_count.columns = cols + [cname]
    df = df.merge(df_count, on=cols, how='left')
    return df


def add_group_max(df, cols, cname, value):
    """ @auther: plantsgo @address: https://www.kaggle.com/plantsgo
    """
    df_count = pd.DataFrame(df.groupby(cols)[value].max()).reset_index()
    df_count.columns = cols + [cname]
    df = df.merge(df_count, on=cols, how='left')
    return df


def add_group_min(df, cols, cname, value):
    """ @auther: plantsgo @address: https://www.kaggle.com/plantsgo
    """
    df_count = pd.DataFrame(df.groupby(cols)[value].min()).reset_index()
    df_count.columns = cols + [cname]
    df = df.merge(df_count, on=cols, how='left')
    return df


def add_group_entropy(df, group, subgroup, cname, value):
    """ refer from @author: owenzhang @address: https://github.com/owenzhang
    """
    if isinstance(subgroup, list):
        full_group = [group]
        full_group.extend(subgroup)
    else:
        full_group = [group, subgroup]
    
    gp_1 = df.groupby(full_group)[value].count().reset_index()
    gp_1.columns = full_group + ['subgroup_cnt']
    
    gp_2 = df.groupby(group)[value].count().reset_index()
    gp_2.columns = [group, 'cnt']
    
    gp_3 = gp_2.merge(gp_1, on=group, how='left')
    
    gp_3['entropy'] = -np.log(gp_3['subgroup_cnt'] / gp_3['cnt']) * gp_3['subgroup_cnt'] / gp_3['cnt']
    gp_3['entropy'].fillna(0, inplace=True)
    gp_4 = gp_3.groupby(group)['entropy'].sum().reset_index()
    gp_4.columns = [group, cname]
    df = df.merge(gp_4, on=group, how='left')
    return df


def add_group_cumcount(df, cols, cname):
	""" Extract sequential id in a group """
    df[cname] = df.groupby(cols).cumcount() + 1
    return df