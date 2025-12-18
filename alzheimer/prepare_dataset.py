#!/usr/bin/env python3
from typing import Literal
import sys, os

THIS_DIR = os.path.dirname(__file__)
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

import pandas as pd
import numpy as np
import json

from kan_utils.dataset import group

__dataset_dir = os.path.join(THIS_DIR,'dataset/')

def create_labels(
    df : pd.DataFrame,
    label_enumeration : Literal['linear', 'exponential'] = 'linear',
    force = False, 
    ) -> dict:
    # print(df.dtypes)
    json_path = os.path.join(__dataset_dir,'labels.json')
    if force or not os.path.exists(json_path):
        label_dict = {}
        
        for col in df.dtypes[df.dtypes == 'category'].index:
            if col == 'MMSE':
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
            lambda row : labels[str(row)]
        )
        if len(labels) < 2:
            df = df.drop(columns=col)
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
            if str(label).lower() == 'nan' :
                df[f'{col}_Is_Unknown'] = (pd.isna(df[col]) | (df[col] == 'NaN')).astype('int8')
            else :
                df[f'{col}_Is_{label}'] = (df[col] == label).astype('int8')
        
        assert df[[f'{col}_Is_{"Unknown" if str(label).lower() == 'nan' else label}' for label in labels]].values.sum() == len(df), str(df[[f'{col}_Is_{"Unknown" if str(label).lower() == 'nan' else label}' for label in labels]])
            
        df.drop(labels=col, axis=1, inplace=True)
        
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].apply(
            lambda row : row.day_of_year
        )
    return df[df.columns.sort_values()]  

def _exe(file, input_dir) :
    import subprocess
    local_dir = os.path.splitext(file)[0].split('_')[:-1]
    target_dir = os.path.join(__dataset_dir, 'slices', *local_dir)
    os.makedirs(target_dir, exist_ok=True)
    return subprocess.run(
        ' '.join((
            'med2image',
            '-I', input_dir,
            '-i', file,
            # '-i', os.path.join(input_dir, file),
            '-d', target_dir,
            '-t jpg',
            '-o sample',
            '--reslice',
            '--verbosity 0'
        )),
        shell = True,
        text = True,
    )
    
def build_dataset(force = False):
    dataset_path = os.path.join(__dataset_dir,'slices')
    if force or not os.path.exists(dataset_path):
        os.environ['KAGGLE_CONFIG_DIR'] = TOP_DIR
        import kagglehub
        import kaggle as kg
        from tqdm import tqdm
        import multiprocessing as mp
        
        # Download Data
        kagglehub.whoami()
        # kg.api.dataset_download_files("ninadaithal/oasis-1-shinohara", path=__dataset_dir, unzip=True, force=force)
        
        input_dir = os.path.join(__dataset_dir, 'oasis', 'OASIS')
        
        with mp.Pool(mp.cpu_count()) as pool:
            with tqdm(os.listdir(input_dir)) as pbar:
                    
                for file in pbar:
                    pbar.set_postfix({'Current file' : file})
                    result = pool.apply_async(_exe, (file, input_dir))
                    
                result = result.get()
                assert result is None, f"Result is {result}"
        
    return get_dataset()

def get_dataset():
    dataset_path = os.path.join(__dataset_dir,'oasis_cross-sectional.csv')
    df = pd.read_csv(dataset_path, index_col='ID')
    for col in list(df.dtypes[df.dtypes == 'object'].index) + ['Educ', 'SES', 'CDR', 'MMSE']:
        df[col] = df[col].apply(
            lambda row: 'NaN' if pd.isna(row) else str(int(row) if isinstance(row, float) else row)
        ).astype('category')
        
    input_dir = os.path.join(__dataset_dir, 'oasis', 'OASIS')
    img_paths = os.listdir(input_dir)
    df['Path'] = [pth for pth in img_paths for idx in df.index if idx in pth]
    df['Path'] = [
        os.path.join(__dataset_dir, 'slices', *os.path.splitext(file)[0].split('_')[:-1])
        for file in df['Path']
    ]
    df = df.drop(columns='Hand')
    return create_groups(df)

def create_groups(df : pd.DataFrame, inplace=False):
    df_local = df.copy()
    
    # Level 1: Age -- Groups [<=25, <=45, <= 65, >65]
    age_bins = [0,25,45,65,100]
    age_labels = ['18-25','26-45','46-65','65+']
    df_local['Age_Group'] = pd.cut(df_local['Age'], bins=age_bins, labels=age_labels)
    
    # Level 2: Number of scans -- Groups [1, 2] (Auto)
    df_local['Num_Scans'] = [str(_)[:str(_).find('_MR')] for _ in df_local.index]
    num_scan_labels = df_local['Num_Scans'].value_counts()
    df_local['Num_Scans'] = df_local['Num_Scans'].apply(lambda row: num_scan_labels[row])
    
    # Level 4: Dominant Hand -- Groups [Right, Left] (Auto)
    # -- Drop because there are only right handed patients
    # hand_labels = df['Hand'].unique()
    # print(hand_labels)
    if 'Hand' in df.columns:
        df.drop(columns=['Hand'], inplace=True)
    
    if inplace:
        df.assign(df_local.to_dict())
    else :
        return df_local
    
def make_groups(df) :
    if 'Age_Group' not in df.columns:
        df = create_groups(df)
        
    # Level 1: Age -- Groups [<=25, <=45, <= 65, >65]
    age_labels = df['Age_Group'].unique()
    
    # Level 2: Number of scans -- Groups [1, 2] (Auto)
    num_scan_labels = df['Num_Scans'].unique()
    
    # Level 3: CDR -- Groups [0, 0.5, 1, 2] (Auto)
    cdr_labels = df['CDR'].unique()
    # print(cdr_labels)
    
    # Level 5: Sex -- Groups [Male, Female] (Auto)
    sex_labels = df['M/F'].unique()
    # print(sex_labels)
    
    # Level 6: Education -- Groups [1, 2, 3, 4, 5, N/A] (Auto)
    edu_labels = df['Educ'].unique()
    # edu_labels.sort()
    # print(edu_labels)
    
    # Level 7: Socio-Economic Status -- Groups [1, 2, 3, 4, 5, N/A] (Auto)
    ses_labels = df['SES'].unique()
    # ses_labels.sort()
    # print(ses_labels)
    
    return group(
        df, {
            'Num_Scans' : num_scan_labels,
            'CDR'       : cdr_labels,
            'M/F'       : sex_labels,
            'Age_Group' : age_labels,
            'Educ'      : edu_labels,
            'SES'       : ses_labels,
        },
    )
    
def __save_groups(groups):
    groups_path = os.path.join(__dataset_dir, 'groups.json')
    with open(groups_path, 'w') as fw:
        json.dump(groups, fw, indent=2)
    
def get_groups(regenerate = False):
    groups_path = os.path.join(__dataset_dir, 'groups.json')
    
    if os.path.exists(groups_path) and not regenerate:
        with open(groups_path, 'r') as fr:
            groups = json.load(fr)
    else :
        groups = make_groups(set_df_labels(build_dataset()))
        __save_groups(groups)
    return groups

def normalize_dataset(
    df : pd.DataFrame,
    reverse  = False
):
    df = df.copy()
    df['MMSE'] = df['MMSE'].astype(float)
    label_path = os.path.join(__dataset_dir, 'normalize.json') 
    
    if not os.path.exists(os.path.join(__dataset_dir, 'statistics.csv')):
        from extract_statistics import extract_statistics
        extract_statistics(df.drop(columns=['Path']), __dataset_dir)
        
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
    df = build_dataset()
    
    if os.path.exists(os.path.join(__dataset_dir, 'groups.json')):
        os.remove(os.path.join(__dataset_dir, 'groups.json'))
    groups = get_groups()
    
    # print(groups)
    
    print('Columns')
    print(df.columns)

    label_dict = create_labels(df, force=True)
        
    for key, val in label_dict.items():
        print(key)
        for key, _val in val.items():
            print('  ', _val, key)

    df_set = set_df_labels(df.copy(),label_dict)
    print('Set df labels')
    print(df_set)
    print(df)
    
    print(df.describe())
    
    # print('FIX THIS')
    # exit()
    
    for col in label_dict.keys():
        print(df[col].value_counts(sort=True, ascending=True))
    
    df_expand = expand_df_labels(df.copy(),label_dict)
    print('Expand df labels')
    print(df_expand)
    
    print(normalize_dataset(df_expand))