#! /usr/bin/env python3
import sys, os

THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prepare_dataset import build_datset, expand_df_labels, set_df_labels, normalize_dataset


def extract_correlate(df, output_dir = None):
    # print(df.columns)
    df_corr = df.corr()

    if output_dir is not None:
        axis_corr = sns.heatmap(
            df_corr,
            vmin=-1, 
            vmax=1, 
            center=0,
            cmap="coolwarm",
            # cmap=sns.diverging_palette(50, 500, n=500, as_cmap=True),
            cbar_kws={"shrink": .5},
            # square=True,
            # annot=True,
            # fmt=".1f",
            linewidth=.5,
        )
        plt.title('Correlation Matrix Heatmap')
        axis_corr.tick_params(labelsize=5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,'correlation.png'))
        plt.close('all')
    else :
        return df_corr

def extract_statistics(df, output_dir = None):
    stats = {
        'index' : df.columns, **{
        key : df.aggregate(key)
            for key in ['min', 'max', 'mean', 'std']
    }}
    stats = pd.DataFrame(stats).set_index('index')

    if output_dir is not None:
        stats.to_csv(os.path.join(output_dir,'statistics.csv'))
    else :
        print('Absolute statistics')
        print(stats)

    df = normalize_dataset(df)
    stats = {
        'index' : df.columns, **{
        key : df.aggregate(key)
            for key in ['min', 'max', 'mean', 'std']
    }}
    stats = pd.DataFrame(stats).set_index('index')

    if output_dir is not None:
        stats.to_csv(os.path.join(output_dir,'normalized_statistics.csv'))
    else :
        print('Normalized statistics')
        print(stats)
    
if __name__ == '__main__':
    # Download latest version
    df = set_df_labels(build_datset())
    extract_correlate(df, os.path.join(THIS_DIR,'dataset'))
    
    df = expand_df_labels(build_datset())
    extract_statistics(df, os.path.join(THIS_DIR,'dataset'))
    