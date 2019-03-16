
import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(importance_df, importance_type='gain', figsize=(12, 30)):
    """ Plot lightgbm feature importance """
    importance_df = importance_df.set_index('features')
    importance_df = importance_df.sort_values('importance_{}'.format(importance_type))
    importance_df['importance_{}'.format(importance_type)].plot(kind='barh', figsize=figsize)
