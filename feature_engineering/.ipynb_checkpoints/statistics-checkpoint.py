
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
    df_mean = pd.DataFrame(df.groupby(cols)[value].mean()).reset_index()
    df_mean.columns = cols + [cname]
    df = df.merge(df_mean, on=cols, how='left')
    return df


def add_group_median(df, cols, cname, value):
    """ @auther: plantsgo @address: https://www.kaggle.com/plantsgo
    """
    df_median = pd.DataFrame(df.groupby(cols)[value].median()).reset_index()
    df_median.columns = cols + [cname]
    df = df.merge(df_median, on=cols, how='left')
    return df


def add_group_std(df, cols, cname, value):
    """ @auther: plantsgo @address: https://www.kaggle.com/plantsgo
    """
    df_std = pd.DataFrame(df.groupby(cols)[value].std()).reset_index()
    df_std.columns = cols + [cname]
    df = df.merge(df_std, on=cols, how='left')
    return df


def add_group_max(df, cols, cname, value):
    """ @auther: plantsgo @address: https://www.kaggle.com/plantsgo
    """
    df_max = pd.DataFrame(df.groupby(cols)[value].max()).reset_index()
    df_max.columns = cols + [cname]
    df = df.merge(df_max, on=cols, how='left')
    return df


def add_group_min(df, cols, cname, value):
    """ @auther: plantsgo @address: https://www.kaggle.com/plantsgo
    """
    df_min = pd.DataFrame(df.groupby(cols)[value].min()).reset_index()
    df_min.columns = cols + [cname]
    df = df.merge(df_min, on=cols, how='left')
    return df


def add_group_sum(df, cols, cname, value):
    """ @auther: plantsgo @address: https://www.kaggle.com/plantsgo
    """
    df_sum = pd.DataFrame(df.groupby(cols)[value].sum()).reset_index()
    df_sum.columns = cols + [cname]
    df = df.merge(df_sum, on=cols, how='left')
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


def add_group_value_count(df, cols, value):
    """ Extract value_counts() in each group """
    df_value_count = df.groupby(cols)[value].size().unstack().fillna(0.0).reset_index()
    df_value_count.columns = [cols[0]] + ['_'.join(c.split()).upper() + '_CNT' for c in df_value_count.columns.tolist()[1:]]
    df = df.merge(df_value_count, on=cols[0], how='left')
    return df


def merge_groupby_feat(df, agg_func_map, agg_config, calc_diff=True):
    """ Extract multiple groupby features
    
    Parameters
    ----------
    df          : pd.DataFrame, input dataframe
    agg_func_map: dictionary, map from aggregation function to feature extraction function
    agg_config  : list, collection of categorical features to groupby and numerical features to aggregate
    calc_diff   : boolean, whether or not to calculate the difference between raw feature and agg feature
    
    Returns
    -------
    df          : pd.DataFrame, input dataframe with extracted features added

    Examples
    --------
    >>> agg_func_map = {"mean": add_group_mean, "median": add_group_median}
    >>> agg_config = [["ORGANIZATION_TYPE"], [("AMT_INCOME_TOTAL", "mean"),
    ...                                       ("AMT_INCOME_TOTAL", "median")])]
    
    """
    new_colnames = []
    for agg_pair in agg_config:
        for agg_feat in agg_pair[1]:
            new_col = "_".join([agg_feat[0], "_".join(agg_pair[0]), agg_feat[1]])
            new_colnames.append(new_col)

            # agg_pair[0]: categorical features, agg_feat[0]: numerical features, agg_feat[1]: agg function
            df = agg_func_map[agg_feat[1]](df, cols=agg_pair[0], cname=new_col, value=agg_feat[0])
            if calc_diff:
                df["{}_{}_diff".format(agg_feat[0], new_col)] = df[agg_feat[0]] - df[new_col]
    return df