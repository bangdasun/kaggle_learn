
import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(importance_df, figsize=(12, 30)):
    """
    Plot feature importance

    :param importance_df: pd.DataFrame, contains 2 columns: features, importance
    :param figsize      : tuple, figure size
    :return:
    """
    assert 'features' in importance_df.columns and 'importance' in importance_df.columns
    importance_df = importance_df.set_index('features')
    importance_df = importance_df.sort_values('importance')
    importance_df['importance'].plot(kind='barh', figsize=figsize)
