from typing import Iterable, Union
import torch
import torch.nn as nn
import numpy as np
  
class FlattenBatch:
    def __init__(self, data_dim):
        super(FlattenBatch,self).__init__()
        self.data_dim  = int(data_dim)
        
    @classmethod
    def __find_batch_size(cls, shape):
        a = 1
        for dim in shape:
            a *= dim
        return a
        
    def __flatten(self, x : Union[torch.Tensor, tuple[torch.Tensor]]):
        if isinstance(x, (tuple, list)) :
            for _iter, _ in enumerate(x):
                batch_shape = self.__find_batch_size(x[_iter].shape[:self.data_dim])
                data_shape = x[_iter].shape[self.data_dim:]
                x[_iter] = x[_iter].resize_(batch_shape, *data_shape)
        else :
            batch_shape = self.__find_batch_size(x.shape[:self.data_dim])
            data_shape = x.shape[self.data_dim:]
            x = x.resize_(batch_shape, *data_shape)

    def __call__(self, data : torch.Tensor, target : torch.Tensor, *args, **kwargs):
        self.__flatten(data)
        self.__flatten(target)
