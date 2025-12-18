from typing import Literal
import sys, os

THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

import pandas as pd
import numpy as np
import json

__dataset_dir = os.path.join(THIS_DIR,'dataset/')

def create_labels(
    df : pd.DataFrame,
    label_enumeration : Literal['linear', 'exponential'] = 'linear',
    force = False, 
    ) -> dict:
    json_path = os.path.join(__dataset_dir,'labels.json')
    if force or not os.path.exists(json_path):
        label_dict = {}
        
        for col in df.dtypes[df.dtypes == 'object'].index:
            if col == 'Date':
                continue
            
            label_dict[col] = {}
            
            for idx, val in enumerate(df[col].sort_values().unique(), start=(label_enumeration == 'linear')):
                label_dict[col][val] = 2 ** idx if label_enumeration == 'exponential' else idx
                
        with open(os.path.join(__dataset_dir,'labels.json'), 'w') as fw:
            json.dump(label_dict, fw, indent=4)
            
    with open(os.path.join(__dataset_dir,'labels.json'), 'r') as fr:
        label_dict = json.load(fr)
        
    return label_dict
    
def set_df_labels(
    df: pd.DataFrame, 
    label_dict :dict = None,
    ):
    if label_dict is None:
        label_dict = create_labels(df)
        
    for col, labels in label_dict.items():
        df[col] = df[col].apply(
            lambda row : labels[row if isinstance(row, str) else 'Unknown']
        )
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].apply(
            lambda row : row.day_of_year
        )
    return df[df.columns.sort_values()]

def expand_df_labels(
    df: pd.DataFrame, 
    label_dict :dict = None,
    ):
    if label_dict is None:
        label_dict = create_labels(df)
        
    for col, labels in label_dict.items():
        for label in labels.keys():
            if label != 'NaN' :
                df[f'{col}_Is_{label}'] = df[col].apply(
                    lambda row : int(row == label)
                )
            else :
                df[f'{col}_Is_Unknown'] = df[col].apply(
                    lambda row : str(row).lower() == 'nan'
                )
            
        df.drop(labels=col, axis=1, inplace=True)
        
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].apply(
            lambda row : row.day_of_year
        )
    return df[df.columns.sort_values()]  

def build_dataset(force = False):
    dataset_path = os.path.join(__dataset_dir,'Ship_Performance_Dataset.csv')
    if force or not os.path.exists(dataset_path):
        os.environ['KAGGLE_CONFIG_DIR'] = TOP_DIR
        import kagglehub
        import kaggle as kg
        
        kagglehub.whoami()
        kg.api.dataset_download_files("arunsara/disease-prediction-using-ml", path=__dataset_dir, unzip=True, force=force)
        
    return get_dataset()

def get_dataset():
    dataset_path = os.path.join(__dataset_dir,'Ship_Performance_Dataset.csv')
    df = pd.read_csv(dataset_path)
    for col in df.dtypes[df.dtypes == 'object'].index:
        if col == 'Date':
            continue
        idx = df[col].isna()
        df.loc[idx, [col]] = 'Unknown'
    return df

def normalize_dataset(
    df : pd.DataFrame,
    reverse  = False
):
    df = df.copy()
    label_path = os.path.join(__dataset_dir, 'normalize.json')
    if not os.path.exists(os.path.join(__dataset_dir, 'statistics.csv')):
        import extract_statistics
        
    stats = pd.read_csv(os.path.join(__dataset_dir, 'statistics.csv'), index_col='index')
    # print(stats)
    
    if os.path.exists(label_path):
        with open(label_path, 'r') as fr:
            label_dict = json.load(fr)
            
        great_values = label_dict['great_values']
        big_values   = label_dict['big_values']
        mid_values   = label_dict['mid_values']
        low_values   = label_dict['low_values']
    else :
        
        great_values = stats[stats['mean'] > 5e4].index.tolist()
        # print(great_values)
        big_values = stats[stats['max'] > 100].index
        big_values = big_values[np.isin(big_values, great_values, invert=True)].tolist()
        # print(big_values)
        mid_values = stats[stats['max'] > 15].index
        mid_values = mid_values[np.isin(mid_values, [*great_values,*big_values], invert=True)].tolist()
        # print(mid_values)
        low_values = stats[stats['max'] > 2].index
        low_values = low_values[np.isin(low_values, [*great_values,*big_values,*mid_values], invert=True)].tolist()
        # print(low_values)
        
        label_dict = {
            'great_values' : great_values,
            'big_values'   : big_values,
            'mid_values'   : mid_values,
            'low_values'   : low_values,
        }
        with open(label_path, 'w') as fw:
            json.dump(label_dict, fw, indent=2)
    
    if reverse :
        df[great_values] = 10 ** (df[great_values].values + 5)
        df[big_values]  *= stats.loc[big_values, 'max']
        df[mid_values]  *= 100
        df[low_values]  *= 10
    else :
        df[great_values] = np.log10(df[great_values].values) - 5
        df[big_values]  /= stats.loc[big_values, 'max']
        df[mid_values]  /= 100
        df[low_values]  /= 10
    
    return df

if __name__ == '__main__':
    # Download latest version
    print('FIX THIS')
    exit()
    df = build_dataset()

    label_dict = create_labels(df, force=True)
        
    for key, val in label_dict.items():
        print(key)
        for key, _val in val.items():
            print('  ', _val, key)
            
    df_set = set_df_labels(df.copy(),label_dict)
    print('Set df labels')
    print(df_set)
    print(df)
    
    for col in label_dict.keys():
        print(col, np.unique_counts(df[col].apply(str)))
    
    df_expand = expand_df_labels(df.copy(),label_dict)
    print('Expand df labels')
    print(df_expand)
    
    print(normalize_dataset(df_expand))