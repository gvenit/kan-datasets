import torch
from torch.utils.data import Dataset, random_split
import pandas as pd
import numpy as np

class DataFrameToDataset (Dataset): 
    def __init__(self, df : pd.DataFrame, input_cols, output_cols, return_key = False):
        super().__init__()
        
        self.index = df.index.tolist()
        self.df = df.reset_index()
        self.input_cols, self.output_cols = input_cols, output_cols
        
        self.i_input_cols  = [self.df.columns.tolist().index(_) for _ in self.input_cols]
        self.i_output_cols = [self.df.columns.tolist().index(_) for _ in self.output_cols]
        
        self._return_key = return_key
        
    def __len__(self) :
        return len(self.df)
    
    def return_key(self, return_key = True):
        self._return_key = return_key
    
    def __getitem__(self, index):
        key = self.index[index]
        data = self.df.iloc[index, self.i_input_cols].values.tolist()
        targ = self.df.iloc[index, self.i_output_cols].values.tolist()
        # print(data, targ)
        if self._return_key:
            return torch.tensor(data).float(), torch.tensor(targ).float(), key
        return torch.tensor(data).float(), torch.tensor(targ).float()
    
        # if self._return_key:
        #     return torch.tensor(data, dtype = torch.float32), torch.tensor(targ, dtype = torch.float32), key
        # return torch.tensor(data, dtype = torch.float32), torch.tensor(targ, dtype = torch.float32)
    
def group(df : pd.DataFrame, label_dict = {}, indices = None):
    '''A grouper for datasets with categorical data.
    
    Args
    ----
    df: DataFrame
        The target dataframe
    label_dict: dict[str, list]
        The labels to apply the group
    indices: list, Optional
    
    Returns
    -------
    dict
    
    >>> group(df, {'Group_0' : [label_0_0, label_0_1, ...]',...})
    
    '''
    if indices is None:
        indices = df.reset_index().index.to_list()
        
        # Reverse dictionary key order
        dict_label = {}
        while  len(label_dict) > 0:
            key, val = label_dict.popitem()
            dict_label[key] = val
            
        # print(label_dict, dict_label)
        # print(dict_label)
        label_dict = dict_label
        # print(label_dict)
        
    if len(label_dict) < 1 or df.empty or len(indices) < 1:
        return indices
    
    key, labels = label_dict.popitem()
    # print(key, labels)
    df = df.iloc[indices]
    subgroup = {
        f'{key} ({label})' : group(
            df,
            label_dict.copy(),(
                df[df[key].isna()] if isinstance(label, float) and np.isnan(label) 
                    else df[df[key] == label]
            ).index.to_list()
        )
            for label in labels 
    }
    for key in list(subgroup.keys()):
        if len(subgroup[key]) == 0:
            subgroup.pop(key)
    
    return subgroup
    
def split_dataset(splits, full_dataset, seed = None):
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return random_split(full_dataset, splits, generator=generator)
    
def smart_split_indices(splits: list[float], full_dataset, groups:dict[str, list] | dict[str, dict], seed = None):
    '''A method for randomly splitting indices for grouped data.
    
    Args
    ----
    splits : list
        A list containing weights with respect to the target size of the subsets.
    full_dataset : Dataset
        The target dataset object to be split.
    groups : dict
        A dictionary containing groups of indexes in the following form:
            >>> groups = {
            ...     'Group_0 (label_0_0)' : {
            ...         'Group_1 (label_1_0)' : {
            ...             ... : {
            ...                 'Group_N (label_N_0)' :[
            ...                     idx_N_0_A,
            ...                     idx_N_0_B,
            ...                     ...
            ...                 ], ...,
            ...                 'Group_N (label_N_M)' :[
            ...                     idx_N_M_A,
            ...                     idx_N_M_B,
            ...                     ...
            ...                 ]
            ...             }
            ...         }
            ...     }
            ... }  
    seed : int, Optional
        If specified, the seed to use for a deterministic split.
        
    Returns
    -------
    list[Subset]
        A list with the split dataset
    '''
    sets = [[] for _ in splits]
    for key, val in groups.items():
        if isinstance(val, dict):
            subsets = smart_split_indices(splits, None, val, seed)
        else :
            subsets = split_dataset(splits, val, seed)
            subsets = [[val[_] for _ in subset.indices] for subset in subsets]
                
        for _iter, subset in enumerate(subsets):
            sets[_iter].extend(subset)
    
    if full_dataset is not None:     
        sets = [
            torch.utils.data.Subset(full_dataset, idx)
                for idx in sets
        ]
    return sets

if __name__ == '__main__':
    df = pd.DataFrame({
        i : ((torch.tensor(range(100)) / 100.) ** i).tolist()
            for i in range(5)
    })
    input_cols  = df.columns[:len(df.columns)-1]
    output_cols = df.columns[len(df.columns)-2:]
    
    dataset = DataFrameToDataset(
        df          = df,
        input_cols  = input_cols,
        output_cols = output_cols,
    )
    print('Dataset lenght', len(dataset))
    
    splits = [0.8, 0.2]
    datasets = split_dataset(splits, dataset, 42)
    
    for i, (split, dataset_i) in enumerate(zip(splits, datasets), start=1):
        print('Split', i, '-- Percentage =', split, '-- Counts =', len(dataset_i))