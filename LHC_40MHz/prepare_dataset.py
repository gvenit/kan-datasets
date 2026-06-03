#! /usr/bin/env python3   
from typing import Literal
import sys, os
import h5py
import tables

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TOP_DIR = os.path.dirname(THIS_DIR)
sys.path.append(TOP_DIR)

import pandas as pd
import numpy as np
import json

from kan_utils.dataset import group

__dataset_dir = os.path.join(THIS_DIR,'dataset')

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
            vals = df[col].sort_values().unique()
            for idx, val in enumerate(vals, start=(label_enumeration == 'linear')):
                label_dict[col][val] = 2 ** idx if label_enumeration == 'exponential' else \
                                       idx      if len(vals) > 2 else \
                                       bool(idx-1)
                
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
        if len(labels.keys()) > 2:
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
        else :
            uc = np.unique_counts(df[col].apply(str))
            label = uc.values[uc.counts.argmin()]
            df[col] = (df[col].values == label).astype('float')
            
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].apply(
            lambda row : row.day_of_year
        )
    return df[df.columns.sort_values()]  

def get_features_labels(file_name, remove_mass_pt_window=True):
    # load file
    h5file = tables.open_file(file_name, 'r')
    
    # Get all available datasets from the h5 file
    available_keys = h5file.root._f_list_nodes()
    
    njets = h5file.root[available_keys[0]].shape[0]

    # allocate arrays based on available data
    feature_array = np.zeros((njets, len(available_keys)))
    label_array = np.zeros((njets, len(available_keys)))

    # load all arrays
    for (i, key) in enumerate(available_keys):
        feature_array[:, i] = getattr(h5file.root, key)[:]

    # remove samples outside mass/pT window
    if remove_mass_pt_window:
        mask = (feature_array[:, 0] > 40) & (feature_array[:, 0] < 200) & \
               (feature_array[:, 1] > 300) & (feature_array[:, 1] < 2000)
        feature_array = feature_array[mask]
        label_array = label_array[mask]

    feature_array = feature_array[np.sum(feature_array, axis=1) == 1]
    label_array = label_array[np.sum(feature_array, axis=1) == 1]

    h5file.close()
    return feature_array, label_array

def build_dataset(force = False):
    dataset_path = os.path.join(__dataset_dir,'background_for_training.h5')
    if force or not os.path.exists(dataset_path):
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        os.system(' '.join([
            'wget',
            'https://zenodo.org/records/5046389/files/background_for_training.h5?download=1',
            '-O' ,
            dataset_path
        ]))
    return get_dataset()

def get_dataset():
    dataset_path = os.path.join(__dataset_dir,'background_for_training.h5')
    with h5py.File(dataset_path, 'r') as file:
        full_data = file['Particles'][:]
        full_data = np.reshape(full_data, (full_data.shape[0], -1))
        keys = [str(_).lstrip("b'").rstrip("'") for _ in file['Particles_Names']]
    
    classes = {'None' : 0, 'MET' : 1, 'Ele' : 2, 'Mu' : 3, 'Jet' : 4}
    rclasses = {val : key for key, val in classes.items()}
    cols = [ 
        f'{ckey}_{key}' 
            for ckey in [
                f'{cls_key}_{_iter}' 
                    for cls_key, len_iter in zip(classes.keys(), [0,1,4,4,10]) 
                    for _iter in range(len_iter)
            ]
            for key in keys
    ]
    df = pd.DataFrame(data = full_data, columns=cols)
    
    
    classes = [col for col in df.columns if col.endswith('Class')]
    df[classes] = np.vectorize(
        lambda x: rclasses[x]
    )(
        df[classes].values.astype('int8')
    )
    df[classes] = df[classes].astype('category')
    
    # feature_array, label_array = get_features_labels(dataset_path, remove_mass_pt_window=False)
    # df = pd.concat([
    #     pd.DataFrame(columns = features, data=feature_array),
    #     pd.DataFrame(columns = labels, data=label_array),
    # ], join='outer', axis=1)
    
    # df['Label'] = [['QCD-QCD', 'H*-BB'][_.argmax()] for _ in df[labels].values]
    # df['Label'] = df['Label'].astype('category')
    # df = df.drop(columns=labels)
    return df

def make_groups(df) :
    return group(
        df, 
        labels = ['Label']
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
        groups = make_groups(set_df_labels(build_dataset()).reset_index())
        # print(groups)
        __save_groups(groups)
    return groups

def normalize_dataset(
    df : pd.DataFrame,
    reverse  = False
):
    df = df.copy()
    label_path = os.path.join(__dataset_dir, 'normalize.json')
    if not os.path.exists(os.path.join(__dataset_dir, 'statistics.csv')):
        import extract_statistics
        extract_statistics.extract_statistics(df, __dataset_dir)
        
    stats = pd.read_csv(os.path.join(__dataset_dir, 'statistics.csv'), index_col='index')
    # print(stats)
    
    _min, _max = stats.loc[df.columns, ['min', 'max']].values.T
    if reverse :
        df[df.columns] =  df[df.columns].values * (_max-_min)  + _min
    else :
        df[df.columns] = (df[df.columns].values - _min) / (_max-_min)
    
    return df

# def normalize_dataset(
#     df : pd.DataFrame,
#     reverse  = False
# ):
#     df = df.copy()
#     label_path = os.path.join(__dataset_dir, 'normalize.json')
#     if not os.path.exists(os.path.join(__dataset_dir, 'statistics.csv')):
#         import extract_statistics
#         extract_statistics.extract_statistics(df, __dataset_dir)
        
#     stats = pd.read_csv(os.path.join(__dataset_dir, 'statistics.csv'), index_col='index')
#     # print(stats)
    
#     if os.path.exists(label_path):
#         with open(label_path, 'r') as fr:
#             label_dict = json.load(fr)
            
#         great_values = [_ for _ in label_dict['great_values'] if _ in df.columns]
#         big_values   = [_ for _ in label_dict['big_values'] if _ in df.columns]
#         mid_values   = [_ for _ in label_dict['mid_values'] if _ in df.columns]
#         low_values   = [_ for _ in label_dict['low_values'] if _ in df.columns]
        
#         # check  = np.isin(great_values + big_values + mid_values + low_values, df.columns).all()
#         # if not check:
#         #     os.remove(label_path) 
        
#     if not os.path.exists(label_path):
#         great_values = stats[stats['mean'] > 5e4].index.tolist()
#         # print(great_values)
#         big_values = stats[stats['max'] > 100].index
#         big_values = big_values[np.isin(big_values, great_values, invert=True)].tolist()
#         # print(big_values)
#         mid_values = stats[stats['max'] > 15].index
#         mid_values = mid_values[np.isin(mid_values, [*great_values,*big_values], invert=True)].tolist()
#         # print(mid_values)
#         low_values = stats[stats['max'] > 2].index
#         low_values = low_values[np.isin(low_values, [*great_values,*big_values,*mid_values], invert=True)].tolist()
#         # print(low_values)
        
#         label_dict = {
#             'great_values' : great_values,
#             'big_values'   : big_values,
#             'mid_values'   : mid_values,
#             'low_values'   : low_values,
#         }
#         with open(label_path, 'w') as fw:
#             json.dump(label_dict, fw, indent=2)
    
#     if reverse :
#         df[great_values] = 10 ** (df[great_values].values + 5)
#         df[big_values]  *= stats.loc[big_values, 'max']
#         df[mid_values]  *= 100
#         df[low_values]  *= 10
#     else :
#         df[great_values] = np.log10(df[great_values].values) - 5
#         df[big_values]  /= stats.loc[big_values, 'max']
#         df[mid_values]  /= 100
#         df[low_values]  /= 10
    
#     # if reverse :
#     #     df[great_values] = (df[great_values].values +0.5) * (10 ** np.ceil(np.log10(stats.loc[great_values, 'max'].values)))[None,:]
#     #     df[big_values]   = (df[big_values].values   +0.5) * (10 ** np.ceil(np.log10(stats.loc[big_values,   'max'].values)))[None,:]
#     #     df[mid_values]  *= 100
#     #     df[low_values]  *= 10
#     # else :
#     #     df[great_values] = df[great_values].values / (10 ** np.ceil(np.log10(stats.loc[great_values, 'max'].values)))[None,:] - 0.5
#     #     df[big_values]   = df[big_values].values   / (10 ** np.ceil(np.log10(stats.loc[big_values,   'max'].values)))[None,:] - 0.5
#     #     df[mid_values]  /= 100
#     #     df[low_values]  /= 10
    
#     return df

if __name__ == '__main__':
    # Download latest version
    df = build_dataset()
    print('Columns: ', df.columns.tolist())

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
    
    ndf = normalize_dataset(df_expand)
    print(ndf)
    print(ndf['Label'])
    
    exit()
    