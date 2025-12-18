#! /usr/bin/env python3
import sys, os

THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prepare_dataset import get_prepared_dataset, set_df_labels, normalize_dataset, clean_dataset, build_dataset

def extract_correlate(df, output_dir=None):
    df_numeric = df.select_dtypes(include=['number'])
    df_corr = df_numeric.corr()

    if output_dir is None:
        return df_corr

    # --- Dynamic sizing based on number of variables ---
    n = df_corr.shape[0]
    fig_height = max(6, 0.35 * n)
    fig_width = max(6, 0.35 * n)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        df_corr,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="coolwarm",
        ax=ax
    )
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(df_corr.index, fontsize=6)

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(df_corr.columns, rotation=90, fontsize=6)

    ax.set_title('Correlation Matrix Heatmap', pad=12)

    plt.savefig(
        os.path.join(output_dir, 'correlation.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

def get_corellate():
    if 'df_corr' not in globals():
        path = os.path.join(THIS_DIR,'dataset','confusion_matrix.csv')
        if not os.path.exists(path):
            df_corr = extract_correlate(set_df_labels(clean_dataset(build_dataset())))
            df_corr.to_csv(path, index_label='Labels')
        else:
            df_corr = pd.read_csv(path, index_col='Labels')
        globals()['df_corr'] = df_corr
    
    return globals()['df_corr']

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
    df_raw = build_dataset()
    df_cleaned = clean_dataset(df_raw)
    df = set_df_labels(df_cleaned)
    extract_correlate(df, os.path.join(THIS_DIR,'dataset'))
    
    get_corellate()
    get_corellate()
    del globals()['df_corr']
    print('get_corellate', get_corellate())
    
    df = get_prepared_dataset(treat_quality_as_categorical=False)
    extract_statistics(df, os.path.join(THIS_DIR,'dataset'))
    