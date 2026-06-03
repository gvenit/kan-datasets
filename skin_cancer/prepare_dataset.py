#!/usr/bin/env python3   
from typing import Literal
import sys, os

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
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
    json_path = os.path.join(__dataset_dir,'labels.json')
    if force or not os.path.exists(json_path):
        label_dict = {}
        
        for col in df.dtypes[df.dtypes == 'category'].index:
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
        
    return df[df.columns.sort_values()]  

def build_dataset(force = False):
    dataset_path = os.path.join(__dataset_dir,'GroundTruth.csv')
    if force or not os.path.exists(dataset_path):
        os.environ['KAGGLE_CONFIG_DIR'] = TOP_DIR
        import kagglehub
        import kaggle as kg
        
        kagglehub.whoami()
        kg.api.dataset_download_files("surajghuwalewala/ham1000-segmentation-and-classification", path=__dataset_dir, unzip=True, force=force)
        
    return get_dataset()

def get_dataset():
    dataset_path = os.path.join(__dataset_dir,'GroundTruth.csv')
    df = pd.read_csv(dataset_path)
    df = df.rename({
        'image' : 'ID', **{
    }}, axis = 1) # .set_index('ID')
    
    cols = df.columns[df.columns != 'ID']
    assert (df[cols].sum(axis=1) == 1).all()
    df['Lesion'] = [cols[col] for col in df[cols].values.argmax(axis=1)]
    df['Lesion'] = df['Lesion'].astype('category')
    df = df.drop(columns=cols)
    
    # Add image and mask paths
    pth = os.path.join(__dataset_dir,'images','{id}.jpg')
    df['Image'] = df['ID'].transform(lambda x: pth.format(id=x))
    pth = os.path.join(__dataset_dir,'masks','{id}_segmentation.png')
    df['Mask']  = df['ID'].transform(lambda x: pth.format(id=x))
    return df.set_index('ID')

def make_groups(df) :
    return group(
        df, 
        labels = ['Lesion',]
    )
    
def __save_groups(groups):
    groups_path = os.path.join(__dataset_dir, 'groups.json')
    with open(groups_path, 'w') as fw:
        json.dump(groups, fw, indent=2)
    
def __nested_pop(groups, keys):
    if len(keys) > 1 and keys[0] in groups.keys():
        __nested_pop(groups[keys[0]], keys[1:])
        if len(groups[keys[0]]) == 0:
            groups.pop(keys[0])
    elif len(keys) == 1 and keys[0] in groups.keys():
        groups.pop(keys[0])
    
def exclude_groups(groups, exclude : list[list[str]]=None):
    if exclude is not None:
        for keys in exclude:
            __nested_pop(groups, keys)
    return groups
    
def get_groups(regenerate = False, exclude=None):
    groups_path = os.path.join(__dataset_dir, 'groups.json')
    
    if os.path.exists(groups_path) and not regenerate:
        with open(groups_path, 'r') as fr:
            groups = json.load(fr)
    else :
        groups = make_groups(set_df_labels(build_dataset()).reset_index())
        # print(groups)
        __save_groups(groups)
    return exclude_groups(groups, exclude)

def normalize_dataset(
    df : pd.DataFrame,
    reverse  = False
):
    return df

if __name__ == '__main__':
    # Download latest version
    # print('FIX THIS')
    # exit()
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
    
    print(get_groups().keys())
    print(get_groups(exclude=[['Lesion (6)']]).keys())
    