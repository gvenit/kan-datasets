from typing import Literal, Mapping
import torch
from prepare_dataset import expand_df_labels, get_dataset
from torch.utils.data import Dataset, random_split
import pandas as pd
import albumentations as A
import cv2

get_extra_transforms = lambda probability, height=None, width=None : [
    A.HorizontalFlip(p=probability),
    A.VerticalFlip(p=probability),
    A.ShiftScaleRotate(scale_limit=(-0.2,0),rotate_limit=(-90,90), p=probability),
    *([] if None in (height, width) else [A.Resize(height, width)]),
    A.RGBShift(p=probability),
    A.RandomSunFlare(p=probability),
    A.RandomBrightnessContrast(p=probability),
    A.HueSaturationValue(p=probability),
    A.ColorJitter(p=probability),
    # A.Perspective(p=probability),
    A.MotionBlur(p=probability),
    A.ChannelShuffle(p=probability),
    A.ChannelDropout(p=probability),
]
get_basic_transforms = lambda probability, height=None, width=None : [
    *([] if None in (height, width) else [A.Resize(height, width)]),
]

class SkinCancerDataset (Dataset):
    __cv2_flags = {
        'image' : cv2.IMREAD_COLOR_RGB,
    }
    def __init__(
        self,
        df              : pd.DataFrame,
        input_cols      : list[str],
        output_cols     : dict[str, list[str]] | list[list[str]],
        input_img_dims  = None,
        path_cols       : dict[Literal['image'],list[str]] = {},
        extra_transforms: list[A.BasicTransform] = [],
        flatten         = False,
        return_key      = False,
        return_type     : Literal['list','dict'] = 'list',
        seed            : int = None,
    ):
        super(SkinCancerDataset, self).__init__()
        self.index  = pd.Series(df.copy().index)
        self.df     = df.copy().reset_index(drop=True)
        col_list    = self.df.columns.tolist()
        self.input_cols  = input_cols
        self.output_cols = output_cols
        self.return_key(return_key)
        self.return_type(return_type)
        
        # Identify columns that are paths
        path_cols = path_cols.copy()
        path_cols_list   = []
        for key, cols in path_cols.items():
            cols = [col_list.index(_) for _ in cols]
            path_cols_list.extend(cols)
            path_cols[key] = cols
            
        diff = set(path_cols_list).difference(path_cols_list)
        assert len(diff) == 0, "Multiple keys found for path columns:" + ', '.join(diff)
        
        self.path_cols = path_cols
        
        gather_cols = set()
        self.path_dict = {
            key : set() for key in path_cols.keys()
        }
        
        # Identify & separate input column paths
        self.i_input_cols, self.i_input_pths = \
            self.filter_path_columns(col_list, path_cols_list, self.input_cols)
            
        # Gather columns
        gather_cols.update(self.i_input_cols)
        gather_cols.update(self.i_input_pths)
        
        # Gather paths
        for key, val in self.group_paths(self.i_input_pths).items():
            self.path_dict[key].update(val)
        self.input_img_dims  = input_img_dims
        
        if not isinstance(output_cols, Mapping):
            self.output_cols = {
                f'head.{_iter}' : cols
                    for _iter, cols in enumerate(cols)
            }
            
        self.i_output_cols = {}
        self.i_output_pths = {}
        for key, cols in self.output_cols.items():
            cols, pths = \
                self.filter_path_columns(col_list, path_cols_list, cols)
                
            gather_cols.update(cols)
            gather_cols.update(pths)
            
            if len(cols) > 0:
                self.i_output_cols[key] = cols
            if len(pths) > 0:
                self.i_output_pths[key] = pths
                
                for key2, val in self.group_paths(self.i_output_pths[key]).items():
                    self.path_dict[key2].update(val)
                
        # self.output_img_dims = output_img_dims
        self.path_cols_list = path_cols_list
        self.path_type_dict = {
            f'{key}_{val}' : key
                for key, vals in self.path_dict.items()
                    for val in vals
                        if len(vals) > 0
        }
        self.path_cols_dict = {
            f'{key}_{val}' : val
                for key, vals in self.path_dict.items()
                    for val in vals
                        if len(vals) > 0
        }
        
        self.df = self.df[self.df.columns[list(gather_cols)]]
        self.df.columns = list(gather_cols)
        self.transform = A.Compose([
                # A.RandomResizedCrop(size=input_img_dims[-2:], scale=(0.08, 1.0), area_for_downscale="image_mask",),
                # A.CenterCrop(*input_img_dims[-2:], pad_if_needed = True),
                *([A.Resize(*input_img_dims[-2:])] if input_img_dims is not None else []),
                *extra_transforms,
                A.Normalize(normalization='min_max'),
                A.ToTensorV2(transpose_mask=True),
            ],
            telemetry           = False,
            additional_targets  = self.path_type_dict,
            seed                = seed
        )
        self.flatten = flatten
    
    def __get_dummy_img(self):
        if not hasattr(self,'_dummy_img'):
            keys = {
                id : self.df.loc[0, self.path_cols_dict[
                    [key for key, val in self.path_type_dict.items() if val == id][0]
                ]]
                    for id in ['image']
            }
            self._dummy_img = {
                key : cv2.imread(
                    pth,
                    self.__cv2_flags[key]
                ).astype('uint8')
                    for key, pth in keys.items()
            }
        return self._dummy_img
        
    def group_paths(self, paths):
        tmp = {
            key : [_ for _ in paths if _ in cols]
                for key, cols in self.path_cols.items()
        }
        return { 
            key : val for key, val in tmp.items()
                if len(val) > 0
        }
        
    @classmethod
    def filter_path_columns(cls, column_list, path_list, iterable):
        # Transform columns to column indices
        cols   = [column_list.index(_) for _ in iterable]
        # Identify input columns that are paths
        pths   = [_ for _ in path_list if _ in cols]
        # Filter out input path columns
        cols   = [_ for _ in cols if _ not in pths]
        return cols, pths
    
    def get_keys(self, index):
        return self.index.loc[list(index)].tolist()
        # return [self.index[idx] for idx in index]
    
    def __len__(self) :
        return len(self.df)
    
    def return_key(self, return_key = True):
        self._return_key = return_key
    
    def return_type(self, return_type : Literal['list','dict'] = 'list'):
        if return_type in ('list','dict'):
            self._return_type = return_type
        else :
            raise NotImplementedError(f'Return type "{return_type}" is not supported.')
    
    def __get_image(self, path_dict, num_idx):
        if torch.utils.data.get_worker_info() is not None:
            cv2.setNumThreads(0)

        out = ({
                key : cv2.imread(
                    pths[_iter],
                    self.__cv2_flags[self.path_type_dict[key]]
                ) for key, pths in path_dict.items()
            } for _iter in range(num_idx)
        )
        return ({
                key : val.astype('uint8')
                    for key, val in out_i.items()
            } for out_i in out
        )
    
    def __get_image_pack(self, pth, num_idx):
        img_dict = self.__get_image(pth, num_idx)
        img_dict = tuple(self.transform(**self.__get_dummy_img(),**_) for _ in img_dict)
        if len(self.i_input_pths) > 0 :
            in_img = tuple(
                tuple(
                    _img_dict[[
                            key for key in _img_dict.keys()
                                if key.endswith(f'_{col}')
                        ][0]
                    ].float() for col in self.i_input_pths
                ) for _img_dict in img_dict
            )
        else :
            in_img = None
            
        if len(self.i_output_pths) > 0 :
            out_img = tuple({
                    head : tuple(
                        _img_dict[[
                                key for key in _img_dict.keys()
                                    if key.endswith(f'_{col}')
                            ][0]
                        ].float() for col in pths
                    ) for head, pths in self.i_output_pths.items()
                } for _img_dict in img_dict
            )
        else :
            out_img = None
        
        return in_img, out_img
    
    def __getitem__(self, index):
        return self.__getitems__([index,])[0]
    
    def __getitems__(self, index):
        if len(self.i_input_cols) > 0:
            data = torch.tensor(self.df.loc[index, self.i_input_cols].values).float() #.tolist()
            
        targ = {}
        for head, i_output_cols in self.i_output_cols.items():
            if len(i_output_cols) > 0:
                try :
                    targ[head] = torch.tensor(self.df.loc[index, i_output_cols].values).float() #.tolist()
                except Exception as e:
                    print(self.df.loc[index, i_output_cols].values)
                    print(i_output_cols)
                    raise e
            # print(data, targ)
        
        if len(self.path_cols_dict) > 0:
            paths = {
                key : self.df.loc[index,val].tolist()
                    for key, val in self.path_cols_dict.items()
            }
            # print(self.get_keys(index))
            in_img, out_img = self.__get_image_pack(paths, len(index))
            # collect = [self.__get_image_pack(pth_i) for pth_i in pth]
            
        if self._return_key:
            keys = self.get_keys(index)
        
        output_args = []
        for _iter, idx in enumerate(index):
            out_data = ()
            if in_img is not None:
                img = [
                    _.unsqueeze(0) if len(_.shape) < 3 else _ 
                        for _ in in_img[_iter]
                ]
                if self.flatten :
                    img = [_.flatten() for _ in img]
                    
                if len(img) == 1:
                    img = img[0]
                out_data += img, 
            
            if len(self.i_input_cols) > 0:
                out_data +=  data[_iter],
                # out_data +=  torch.tensor(data[_iter]).float(),
                
            if len(out_data) == 1:
                out_data = out_data[0]
            
            out_targ = {}
            for head in self.output_cols:
                trg = ()
                if head in self.i_output_pths:
                    img = [
                        _.unsqueeze(0) if len(_.shape) < 3 else _ 
                            for _ in out_img[_iter][head]
                    ]
                    if self.flatten :
                        img = [_.flatten() for _ in img]
                        
                    if len(img) == 1:
                        img = img[0]
                    trg += img,
                
                if head in targ:
                    trg +=  targ[head][_iter],
                    # trg +=  torch.tensor(targ[head][_iter]).float(),
                    
                if len(trg) == 1:
                    trg = trg[0]

                out_targ[head] = trg
                
            if self._return_type == 'list':
                out_targ = tuple(out_targ.values())
                # Unwrap single-element tuples
                if len(out_targ) == 1:
                    out_targ = out_targ[0]
            
            if self._return_key:
                output_args.append((out_data, out_targ, keys[_iter]))
            else :
                output_args.append((out_data, out_targ))
        
        return output_args
