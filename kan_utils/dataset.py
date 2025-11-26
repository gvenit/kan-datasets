import torch
from torch.utils.data import Dataset, random_split
import pandas as pd

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
    
def split_dataset(splits, full_dataset, seed = None):
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
        
    return random_split(full_dataset, splits)
    
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