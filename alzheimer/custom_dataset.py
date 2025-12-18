from typing import Literal
import torch
from torch.utils.data import Dataset, random_split
import pandas as pd
import cv2
import albumentations as A
import os

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
        flags = (cv2.IMREAD_UNCHANGED, ),
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
        self.flags           = flags
        self.orientation     = orientation
            
        self.input_transform = A.Compose([
                A.Resize(*self.input_img_dims),
                A.ToTensorV2(),
            ], telemetry=False
        )
        self.output_transform = A.Compose([
                A.Resize(*self.output_img_dims),
                A.ToTensorV2(),
            ], telemetry=False
        )
        
    def __len__(self) :
        return len(self.df)
    
    def return_key(self, return_key = True):
        self._return_key = return_key
    
    def __get_input_image(self, pth):
        with cv2.imread(pth, *self.flags) as fr:
            if self.pth_isin_in:
                img = self.input_transform(fr)
            else :
                img = None
        return img
    
    def __get_output_image(self, pth):
        with cv2.imread(pth, *self.flags) as fr:
            if self.pth_isin_out:
                img = self.output_transform(fr)
            else :
                img = None
        return img
    
    def __get_image_pack(self, pth):
        if self.orientation in ('x','y','z',):
            in_axis = out_axis = self.orientation
        else :
            if self.orientation in ('fixed',):
                in_axis = out_axis = ['x','y','z'][torch.randint(0,3,(1,))]
            if self.orientation in ('random',):
                in_axis  = ['x','y','z'][torch.randint(0,3,(1,))]
                out_axis = ['x','y','z'][torch.randint(0,3,(1,))]
                
        if self.pth_isin_in:
            in_pth = os.listdir(os.path.join(pth, in_axis))
            in_img = torch.stack([
                self.__get_input_image(in_pth_i)
                    for in_pth_i in in_pth
            ])
        else :
            in_img = None
            
        if self.pth_isin_out:
            out_pth = os.listdir(os.path.join(pth, out_axis))
            out_img = torch.stack([
                self.__get_output_image(out_pth_i)
                    for out_pth_i in out_pth
            ])
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
        key  = [self.index[idx] for idx in index]
        data = self.df.iloc[index, self.i_input_cols].values.tolist()
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
        
        output_args = ()
        
        if self.pth_isin_in:
            output_args += (torch.tensor(data).float(), in_img.float()),
        else :
            output_args +=  torch.tensor(data).float(),
            
        if self.pth_isin_out:
            output_args += (torch.tensor(targ).float(), out_img.float()),
        else :
            output_args +=  torch.tensor(targ).float(),
            
        if self._return_key:
            output_args +=  key,
            
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