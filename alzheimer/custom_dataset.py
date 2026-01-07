from typing import Literal
import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import pandas as pd
import nibabel as nib
import albumentations as A
import os

global_pool = torch.multiprocessing.get_context().Pool(processes=torch.multiprocessing.cpu_count())

class AlzheimerDataset (Dataset): 
    def __init__(
        self, 
        df : pd.DataFrame, 
        input_cols, 
        output_cols, 
        input_img_dims,
        output_img_dims,
        return_key = False, 
        path_col = 'Path',
        orientation : Literal['x','y','z','fixed','random'] = 'fixed',
    ):
        super().__init__()
        
        self.index = df.index.tolist()
        self.df = df.reset_index()
        self.input_cols, self.output_cols = input_cols, output_cols
        
        self.i_input_cols  = [self.df.columns.tolist().index(_) for _ in self.input_cols]
        self.i_output_cols = [self.df.columns.tolist().index(_) for _ in self.output_cols]
        
        self._return_key = return_key
        self.i_path_col  = self.df.columns.tolist().index(path_col)
        
        self.pth_isin_in  = self.i_path_col in self.i_input_cols
        self.pth_isin_out = self.i_path_col in self.i_output_cols
        
        if self.pth_isin_in:
            self.i_input_cols.remove(self.i_path_col)
            
        if self.pth_isin_out:
            self.i_output_cols.remove(self.i_path_col)
            
        self.input_img_dims  = input_img_dims
        self.output_img_dims = output_img_dims
        self.orientation     = orientation
            
        self.transform = A.Compose([
                A.Normalize(normalization='min_max'),
                A.RandomCrop3D(self.input_img_dims),
                A.CubicSymmetry(),
                A.CoarseDropout3D(p=0.5),
            ], telemetry=False
        )
    
    def get_keys(self, index):
        return [self.index[idx] for idx in index]
    
    def __get_orientation(self):
        if self.orientation in ('x','y','z',):
            in_axis = out_axis = self.orientation
        else :
            if self.orientation in ('fixed',):
                in_axis = out_axis = ['x','y','z'][torch.randint(0,3,(1,))]
            if self.orientation in ('random',):
                in_axis  = ['x','y','z'][torch.randint(0,3,(1,))]
                out_axis = ['x','y','z'][torch.randint(0,3,(1,))]
                
        return in_axis, out_axis
        
    def __len__(self) :
        return len(self.df)
    
    def return_key(self, return_key = True):
        self._return_key = return_key
    
    @classmethod
    def __get_image(cls, pth):
        img = nib.load(pth)
        return np.asarray(img.get_fdata(), dtype=img.header.get_data_dtype())
    
    def __get_image_pack(self, pth, in_axis, out_axis):
        img = self.__get_image(pth)
        img = self.transform(volume = img,)['volume']
        if self.pth_isin_in:
            in_img = torch.as_tensor(img).float()
        else :
            in_img = None
            
        if self.pth_isin_out:
            out_img = torch.as_tensor(img).float()
        else :
            out_img = None
            
        return in_img, out_img
    
    def __getitem__(self, index):
        data, targ, *key = self.__getitems__([index,])
        
        if isinstance(data, tuple):
            data = (
                _.squeeze(0) for _ in data
            )
        else :
            data = data.squeeze(0)
            
        if isinstance(targ, tuple):
            targ = (
                _.squeeze(0) for _ in targ
            )
        else :
            targ = targ.squeeze(0)
        key = (
            _[0] for _ in key
        )
        return data, targ, *key
    
    def __getitems__(self, index):
        if len(self.i_input_cols) > 0:
            data = self.df.iloc[index, self.i_input_cols].values.tolist()
            
        if len(self.i_output_cols) > 0:
            targ = self.df.iloc[index, self.i_output_cols].values.tolist()
        # print(data, targ)
        
        if self.pth_isin_in or self.pth_isin_out:
            pth = self.df.iloc[index, self.i_path_col]
            
            in_axis, out_axis = self.__get_orientation()
            
            collect = [self.__get_image_pack(pth_i, in_axis, out_axis) for pth_i in pth]
            
            if self.pth_isin_in:
                in_img = torch.stack([
                    collect_i[0] for collect_i in collect
                ])
            if self.pth_isin_out:
                out_img = torch.stack([
                    collect_i[1] for collect_i in collect
                ])
        
        output_args = ()
        
        out_data = ()
        if self.pth_isin_in:
            out_data += in_img.float(), 
        
        if len(self.i_input_cols) > 0:
            out_data +=  torch.tensor(data).float(),
            
        if len(out_data) == 1:
            output_args += out_data
        else :
            output_args += out_data,
            
        out_targ = ()
        if self.pth_isin_out:
            out_targ += out_img.float(), 
        
        if len(self.i_input_cols) > 0:
            out_targ +=  torch.tensor(targ).float(),
            
        if len(out_targ) == 1:
            output_args += out_targ
        else :
            output_args += out_targ,
            
        if self._return_key:
            output_args +=  torch.as_tensor(index),
            
        return output_args
    
# if __name__ == '__main__':
#     df = pd.DataFrame({
#         i : ((torch.tensor(range(100)) / 100.) ** i).tolist()
#             for i in range(5)
#     })
#     input_cols  = df.columns[:len(df.columns)-1]
#     output_cols = df.columns[len(df.columns)-2:]
    
#     dataset = DataFrameToDataset(
#         df          = df,
#         input_cols  = input_cols,
#         output_cols = output_cols,
#     )
#     print('Dataset lenght', len(dataset))
    
#     splits = [0.8, 0.2]
#     datasets = split_dataset(splits, dataset, 42)
    
#     for i, (split, dataset_i) in enumerate(zip(splits, datasets), start=1):
#         print('Split', i, '-- Percentage =', split, '-- Counts =', len(dataset_i))