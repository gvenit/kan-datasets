from typing import Literal
import torch
from torch.utils.data import Dataset, random_split
import numpy as np
import pandas as pd
import nibabel as nib
import albumentations as A
import os

global_pool = torch.multiprocessing.get_context().Pool(processes=torch.multiprocessing.cpu_count())

class SwapAxes3D (A.Transform3D):
    def __init__(
        self, 
        p = 1, 
        orientation : Literal['x','y','z','random'] = 'random' 
    ):
        super().__init__(p)
        self.orientation = {
            'x' : lambda : 0,
            'y' : lambda : 1,
            'z' : lambda : 2,
            'random' : lambda : self.random_generator.integers(0,3)
        }[orientation]
        
        self.transforms = {
            0 : lambda volume: np.swapaxes(volume, 0, 2),
            1 : lambda volume: np.swapaxes(volume, 0, 1),
            2 : lambda volume: volume,
        }
        
    def apply_to_volume(self, volume, *args, **params,) :
        return self.transforms[self.orientation()](volume)
        
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
        orientation : Literal['x','y','z','same','random'] = 'same',
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
            ], telemetry=False
        )
        if self.orientation == 'same':
            self.transform = A.Compose([
                    self.transform,
                    A.CubicSymmetry(),
                    A.CoarseDropout3D(p=0.5),
                ], telemetry=False
            )
        elif self.orientation in ('x','y','z', 'random'):
            self.in_transform = A.Compose([
                    A.CubicSymmetry(),
                    A.CoarseDropout3D(p=0.5),
                ], telemetry=False
            )
            if self.orientation == 'random':
                self.out_transform = A.Compose([
                    A.CubicSymmetry(),
                ], telemetry=False
            )
            else :
                self.out_transform = A.Compose([
                        SwapAxes3D(orientation=self.orientation),
                    ], telemetry=False
                )
                
    
    def get_keys(self, index):
        return [self.index[idx] for idx in index]
    
    def __len__(self) :
        return len(self.df)
    
    def return_key(self, return_key = True):
        self._return_key = return_key
    
    @classmethod
    def __get_image(cls, pth):
        img = nib.load(pth)
        return np.asarray(img.get_fdata(), dtype=img.header.get_data_dtype())
    
    def __get_image_pack(self, pth):
        img = self.__get_image(pth)
        img = self.transform(volume = img,)['volume']
        
        if self.orientation == 'same':
            if self.pth_isin_in:
                in_img = torch.as_tensor(img).float()
            else :
                in_img = None
                
            if self.pth_isin_out:
                out_img = torch.as_tensor(img).float()
            else :
                out_img = None
        else :
            if self.pth_isin_in:
                in_img = self.in_transform(volume = img,)['volume']
                in_img = torch.as_tensor(in_img).float()
            else :
                in_img = None
                
            if self.pth_isin_out:
                out_img = self.out_transform(volume = img,)['volume']
                out_img = torch.as_tensor(out_img).float()
            else :
                out_img = None
                
        return in_img, out_img
    
    def __getitem__(self, index):
        return self.__getitems__([index,])[0]
    
    def __getitems__(self, index):
        if len(self.i_input_cols) > 0:
            data = self.df.iloc[index, self.i_input_cols].values.tolist()
            
        if len(self.i_output_cols) > 0:
            targ = self.df.iloc[index, self.i_output_cols].values.tolist()
        # print(data, targ)
        
        if self.pth_isin_in or self.pth_isin_out:
            pth = self.df.iloc[index, self.i_path_col]
            
            collect = [self.__get_image_pack(pth_i) for pth_i in pth]
            
            if self.pth_isin_in:
                in_img = torch.stack([
                    collect_i[0] for collect_i in collect
                ])
            if self.pth_isin_out:
                out_img = torch.stack([
                    collect_i[1] for collect_i in collect
                ])
        
        output_args = []
        for _iter, idx in enumerate(index):
            out_data = ()
            if self.pth_isin_in:
                out_data += in_img[_iter].float(), 
            
            if len(self.i_input_cols) > 0:
                out_data +=  torch.tensor(data[_iter]).float(),
                
            if len(out_data) == 1:
                out_data = out_data[0]
            
            out_targ = ()
            if self.pth_isin_out:
                out_targ += out_img[_iter].float(), 
            
            if len(self.i_input_cols) > 0:
                out_targ +=  torch.tensor(targ[_iter]).float(),
                
            if len(out_targ) == 1:
                out_targ = out_targ[0]
            
            if self._return_key:
                output_args.append((out_data, out_targ, idx))
            else :
                output_args.append((out_data, out_targ))
        
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