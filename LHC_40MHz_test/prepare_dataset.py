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
    df: pd.DataFrame,
    label_enumeration: Literal['linear', 'exponential'] = 'linear',
    force: bool = False, 
    ) -> dict:
    """Create label mappings for categorical columns."""
    json_path = os.path.join(__dataset_dir, 'labels.json')
    
    if force or not os.path.exists(json_path):
        label_dict = {}
        
        for col in df.dtypes[df.dtypes == 'category'].index:
            if col == 'MMSE':
                continue
            label_dict[col] = {}
            vals = df[col].sort_values().unique()
            
            for idx, val in enumerate(vals):
                if label_enumeration == 'exponential':
                    label_dict[col][val] = 2 ** idx
                if label_enumeration == 'linear':
                    label_dict[col][val] = idx
                else:
                    raise ValueError(f'Unsupported label enumeration method: {label_enumeration}')
                
        os.makedirs(__dataset_dir, exist_ok=True)
        with open(json_path, 'w') as fw:
            json.dump(label_dict, fw, indent=4)
            
    with open(json_path, 'r') as fr:
        label_dict = json.load(fr)
        
    return label_dict
    
def set_df_labels(
    df: pd.DataFrame, 
    label_dict: dict = None,
    ):
    """Apply label mappings to categorical columns."""
    if label_dict is None:
        label_dict = create_labels(df)
        
    for col, labels in label_dict.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(labels)
                
    # Handle date columns if present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].dt.dayofyear
        
    return df[sorted(df.columns)]

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
    """Build and cache the processed dataset."""
    dataset_path = os.path.join(__dataset_dir,'background_for_training.h5')
    processed_path = os.path.join(__dataset_dir, 'processed_dataset.pkl')
    
    # Check if processed dataset exists and is newer than raw data
    if (force or 
        not os.path.exists(processed_path) or 
        (os.path.exists(dataset_path) and 
         os.path.getmtime(processed_path) < os.path.getmtime(dataset_path))):
        
        # Download raw data if needed
        if force or not os.path.exists(dataset_path):
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
            os.system(' '.join([
                'wget',
                'https://zenodo.org/records/5046389/files/background_for_training.h5?download=1',
                '-O' ,
                dataset_path
            ]))
        
        # Process and cache the dataset
        print("Processing dataset from raw data... (this may take a few minutes)", file=sys.stderr)
        df = get_dataset()
        
        # Save processed dataset
        os.makedirs(__dataset_dir, exist_ok=True)
        df.to_pickle(processed_path)
        print(f"Cached processed dataset to {processed_path}", file=sys.stderr)
        return df
    else:
        # Load cached processed dataset
        print(f"Loading cached dataset from {processed_path}", file=sys.stderr)
        return pd.read_pickle(processed_path)

def get_dataset():
    """Load and process the LHC dataset."""
    dataset_path = os.path.join(__dataset_dir, 'background_for_training.h5')
    
    # Load raw 3D data: (samples, 19 particles, 4 features)
    with h5py.File(dataset_path, 'r') as file:
        full_data = file['Particles'][:]  # Shape: (4M, 19, 4)
        full_data = np.reshape(full_data, (full_data.shape[0], -1))  # Flatten to: (4M, 76)
        keys = [str(_).lstrip("b'").rstrip("'") for _ in file['Particles_Names']]  # ['Pt', 'Eta', 'Phi', 'Class']
    
    # Generate column names for flattened data
    # Structure: MET_0_Pt, MET_0_Eta, MET_0_Phi, MET_0_Class, Ele_0_Pt, ..., Jet_9_Class
    # Particle slots: MET(1), Ele(4), Mu(4), Jet(10) = 19 total particles
    cols = [ 
        f'{particle_slot}_{feature}' 
            for particle_slot in [
                f'{particle_type}_{slot_num}' 
                    for particle_type, num_slots in zip(['MET', 'Ele', 'Mu', 'Jet'], [1, 4, 4, 10])
                    for slot_num in range(num_slots)
            ]
            for feature in keys  # For each particle slot, add all 4 features
    ]
    
    df = pd.DataFrame(data=full_data, columns=cols)
    
    # Create Label column using vectorized operations for speed
    # Physics priority: Jet (rarest) > Muon > Electron (most common detectable)
    # Raw class values: 0=Empty, 1=MET, 2=Electron, 3=Muon, 4=Jet
    
    # Extract class data for each particle type as 2D arrays for vectorized processing  
    ele_classes = df[[f'Ele_{i}_Class' for i in range(4)]].values    # Shape: (4M, 4)
    mu_classes = df[[f'Mu_{i}_Class' for i in range(4)]].values     # Shape: (4M, 4)
    jet_classes = df[[f'Jet_{i}_Class' for i in range(10)]].values  # Shape: (4M, 10)
    
    # Check if ANY particle of each type exists in each collision event
    # .any(axis=1) checks across all slots for each sample
    has_ele = (ele_classes == 2).any(axis=1)   # Boolean array: (4M,)
    has_mu = (mu_classes == 3).any(axis=1)     # Boolean array: (4M,)  
    has_jet = (jet_classes == 4).any(axis=1)   # Boolean array: (4M,)
    # BUG: are they mutually exclusive?
    # Assign labels based on physics priority (higher priority overwrites lower)
    # Default to Electron (most common), then override with rarer particles
    label = np.zeros(len(df), dtype='int8')     # Default: all Electron (label=0)
    label[has_mu] = 2                           # Muon events get label=2
    label[has_jet] = 1                          # Jet events get label=1 (highest priority)
    
    df['Label'] = label
    df = df.astype('float32')  
    df['Label'] = df['Label'].astype('int8')  
    
    return df

def make_groups(df):
    """Create groups for dataset organization.""" 
    return group(df, labels=['Label'])
    
def save_groups(groups):
    """Save groups to JSON file."""
    groups_path = os.path.join(__dataset_dir, 'groups.json')
    os.makedirs(__dataset_dir, exist_ok=True)
    with open(groups_path, 'w') as fw:
        json.dump(groups, fw, indent=2)
    
def get_groups(regenerate=False):
    """Get or generate data groups."""
    groups_path = os.path.join(__dataset_dir, 'groups.json')
    
    if os.path.exists(groups_path) and not regenerate:
        with open(groups_path, 'r') as fr:
            groups = json.load(fr)
    else:
        df = build_dataset()
        df_labeled = set_df_labels(df).reset_index(drop=True)
        groups = make_groups(df_labeled)
        save_groups(groups)
    
    return groups

def normalize_dataset(df: pd.DataFrame, reverse=False):
    """Normalize dataset using min-max scaling."""
    df = df.copy()
    
    # Convert all columns to float to avoid dtype compatibility issues
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].astype('float64')
    
    # Generate statistics if they don't exist
    if not os.path.exists(os.path.join(__dataset_dir, 'statistics.csv')):
        import extract_statistics
        extract_statistics.extract_statistics(df, __dataset_dir)
        
    stats = pd.read_csv(os.path.join(__dataset_dir, 'statistics.csv'), index_col='index')
    
    # Get min and max values for each column
    _min, _max = stats.loc[df.columns, ['min', 'max']].values.T
    
    # Handle constant columns (where max == min) to avoid division by zero
    range_values = _max - _min
    constant_mask = range_values == 0
    
    if reverse:
        # Reverse normalization: x = x_norm * (max - min) + min
        # For non-constant columns
        if not constant_mask.all():
            df.loc[:, ~constant_mask] = (df.loc[:, ~constant_mask].values * 
                                       range_values[~constant_mask] + 
                                       _min[~constant_mask])
        # Constant columns remain unchanged in reverse normalization
    else:
        # Forward normalization: x_norm = (x - min) / (max - min)
        # For non-constant columns
        if not constant_mask.all():
            df.loc[:, ~constant_mask] = ((df.loc[:, ~constant_mask].values - 
                                        _min[~constant_mask]) / 
                                       range_values[~constant_mask])
        # For constant columns, set to 0 (since (x-min)/(max-min) = 0/0, we define as 0)
        if constant_mask.any():
            df.loc[:, constant_mask] = 0.0
    
    return df

if __name__ == '__main__':
    print("Building LHC dataset...")
    
    # Load the raw dataset
    df = build_dataset()
    print(f'Dataset loaded with {df.shape[0]} samples and {df.shape[1]} features')
    print('Columns:', df.columns.tolist())

    # Create label mappings
    label_dict = create_labels(df, force=True)
    print('\nLabel mappings created:')
    for key, val in label_dict.items():
        print(f'{key}:')
        for label, numeric_val in val.items():
            print(f'  {label} -> {numeric_val}')
            
    # Apply label mappings
    df_labeled = set_df_labels(df.copy(), label_dict)
    print(f'\nLabeled dataset shape: {df_labeled.shape}')
    
    # Check label distributions
    for col in label_dict.keys():
        if col in df_labeled.columns:
            unique_vals = np.unique(df_labeled[col].values, return_counts=True)
            print(f'{col} distribution: {dict(zip(unique_vals[0], unique_vals[1]))}')
    
    # Normalize the dataset
    normalized_df = normalize_dataset(df_labeled)
    print(f'\nNormalized dataset created with shape: {normalized_df.shape}')
    
    print("\nDataset preparation complete!")