from typing import Literal, overload
import torch
import numpy as np

from kan_utils.utils import expand_value
from kan_utils.models import SubBatch

class ImageSplitter(torch.nn.Module):
    def __init__(
        self,
        input_shape : tuple[int, int] = None,
        output_shape: tuple[int, int] = None,
        stride : int | tuple[int, int] = None,
        stride_percent : float | tuple[float, float] = 1.,
        num_strides : int | tuple[int, int] = None,
        output_dim  : int = 0,
        padding_val : Literal['constant', 'reflect', 'replicate', 'circular'] = 'replicate',
        keep_chn_dim : bool = False,
    ):
        super(ImageSplitter, self).__init__()
        self.output_dim  = output_dim
        self.num_dims, self.output_shape, self.stride = self.parse_shapes(
            input_shape     = input_shape,
            output_shape    = output_shape,
            stride          = stride,
            stride_percent  = stride_percent,
            num_strides     = num_strides,
        )
        self.input_shape = input_shape if input_shape is None else expand_value(input_shape, self.num_dims)
        self.keep_chn_dim = keep_chn_dim
        if padding_val in ('constant', 'reflect', 'replicate', 'circular'):
            self.padding_val = padding_val
        else :
            raise NotImplementedError(f'Strategy "{padding_val}" for padding is not implemented.')
        
        self.unfold = SubBatch(
            input_data_dim  = -self.num_dims-1,
            model           = torch.nn.Unfold(
                kernel_size = self.output_shape,
                stride      = self.stride,
            )  
        )
    
    @classmethod
    def get_num_dims(
        self,
        *args,
        ):
        tmp = filter(
            lambda x: hasattr(x, '__iter__'),
            args
        )
        return max(*[len(_) for _ in tmp])
        
    def reverse(self, input_shape = None):
        if self.input_shape is None:
            self.input_shape = input_shape
        return ImageMerger(self)
    
    def get_padding(self, input_shape) :
        padding = [(input_shape[_]-self.output_shape[_]) % self.stride[_] for _ in range(-self.num_dims,0)]
        padding = [(self.stride[_]-          padding[_]) % self.stride[_] for _ in range(-self.num_dims,0)]
        
        return tuple(padding)
        
    def parse_shapes(
        self,
        input_shape : tuple[int, int] = None,
        output_shape: tuple[int, int] = None,
        stride : int | tuple[int, int] = None,
        num_strides : int | tuple[int, int] = None,
        stride_percent : float | tuple[float, float] = 1.,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        num_dims = self.get_num_dims(
            input_shape,
            output_shape,
            stride,
            num_strides,
            stride_percent,
        )
        if None not in (output_shape, stride):
            output_shape = expand_value(output_shape, num_dims)
            stride       = expand_value(stride,       num_dims)
            
            stride = [int(_) for _ in stride]
            assert np.all([0 < stride[_] <= output_shape[_] for _ in range(num_dims)])
            
        elif None not in (output_shape, num_strides):
            # if not (hasattr(input_shape, '__iter__') and len(input_shape) == num_dims):
            #     raise TypeError(f"Expected 'input_shape' to have type 'tuple' of {num_dims} int elements'; got '{input_shape}'")
            input_shape  = expand_value(input_shape,  num_dims)
            output_shape = expand_value(output_shape, num_dims)
            num_strides  = expand_value(num_strides,  num_dims)
            
            assert np.all([num_strides[_] >= (input_shape[_] / output_shape[_]) for _ in range(num_strides)])
            stride  = [np.ceil((input_shape[_] - output_shape[_]) / (num_strides[_] - 1)).astype(int) for _ in range(num_strides)]
            
        elif None not in (output_shape, stride_percent):
            output_shape   = expand_value(output_shape,   num_dims)
            stride_percent = expand_value(stride_percent, num_dims)
            
            assert np.all([0 < stride_percent[_] <= 1. for _ in range(num_dims)])
            stride  = [int(round(stride_percent[_] * output_shape[_])) for _ in range(num_dims)]
            
        elif None not in (stride, num_strides):
            # if not (hasattr(input_shape, '__iter__') and len(input_shape) == num_dims):
            #     raise TypeError(f"Expected 'input_shape' to have type 'tuple' of {num_dims} int elements'; got '{input_shape}'")
            input_shape  = expand_value(input_shape,  num_dims)
            stride       = expand_value(stride,       num_dims)
            num_strides  = expand_value(num_strides,  num_dims)

            output_shape = [input_shape[_] - (num_strides[_]-1) * stride[_] for _ in range(num_dims)]
            assert np.all([0 < stride[_] <= output_shape[_]  for _ in range(num_dims)])
            assert np.all([num_strides[_] >= (input_shape[_] / output_shape[_])  for _ in range(num_dims)])
            
        elif None not in (stride, stride_percent):
            stride         = expand_value(stride,         num_dims)
            stride_percent = expand_value(stride_percent, num_dims)
            
            assert np.all([0 < stride_percent[_] <= 1.  for _ in range(num_dims)])
            output_shape = [int(round(stride[_] / stride_percent[_])) for _ in range(num_dims)]
        
        elif None not in (stride_percent, num_strides):
            if not (hasattr(input_shape, '__iter__') and len(input_shape) == 2):
                raise TypeError(f"Expected 'input_shape' to have type 'tuple[int, int]'; got '{input_shape}'")
            
            stride_percent = expand_value(stride_percent, num_dims)
            num_strides  = expand_value(num_strides,  num_dims)
            
            assert np.all([0 < stride_percent[_] <= 1.  for _ in range(num_dims)])
            
            stride = [np.ceil(input_shape[_] / (num_strides[_] + 1./stride_percent[_] - 1)).astype(int) for _ in range(num_dims)]
            output_shape = [stride[_] / stride_percent[_] for _ in range(num_dims)]
            
            assert np.all([0 < stride[_] <= output_shape[_]  for _ in range(num_dims)])
        
        else :
            raise ValueError("At least two of 'output_shape', 'stride', 'num_strides', 'stride_percent' must be specified ")
        return num_dims, tuple(output_shape), tuple(stride)
     
    def apply_padding(self, x):
        if self.padding_val not in ('constant', 'reflect', 'replicate', 'circular'):
            raise NotImplementedError(f'Strategy "{self.padding_val}" for padding is not implemented.')
    
        if self.input_shape is None:
            input_shape = list(x.shape[-self.num_dims:])
        else :
            assert self.input_shape == list(x.shape[-self.num_dims:]), f'Expected size {self.input_shape}; got {x.shape[-self.num_dims:]}'
            input_shape = self.input_shape
           
        padding = self.get_padding(input_shape)
        first_pad = lambda dim: padding[dim] // 2
        last_pad  = lambda dim: (padding[dim]+1) // 2
        
        return torch.nn.functional.pad(
            x,
            pad  = [(
                    first_pad(-(dim//2)-1) if dim % 2 == 0 else 
                    last_pad (-(dim//2)-1)
                ) for dim in range(2*self.num_dims)
            ],
            mode = self.padding_val,
        )
    
    def forward(self, x:torch.Tensor):
        x = self.apply_padding(x)
        x = self.unfold(x).reshape(
            *x.shape[:-self.num_dims], *self.output_shape, -1
        ).movedim(-1,self.output_dim)
        
        if not self.keep_chn_dim:
            x = x.movedim(-self.num_dims-1,self.output_dim)
            x = x.flatten(start_dim=self.output_dim, end_dim=self.output_dim+1)
            
        return x
      
class ImageMerger(torch.nn.Module):
    @overload
    def __init__(
        self,
        ImageSplitter : ImageSplitter
    ): ...
        
    @overload
    def __init__(
        self,
        input_shape : tuple[int, int] = None,
        output_shape: tuple[int, int] = None,
        stride : int | tuple[int, int] = None,
        stride_percent : float | tuple[float, float] = 1.,
        num_strides : int | tuple[int, int] = None,
        input_dim  : int = 0,
        keep_chn_dim : bool = False,
    ): ...
        
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(ImageMerger, self).__init__()
        
        if len(args) and isinstance(args[0], ImageSplitter) :
            self.splitter = args[0]
        elif 'ImageSplitter' in kwargs.keys():
            self.splitter = kwargs['ImageSplitter']
        else :
            kwargs = self.parse_args(*args, **kwargs)
            self.splitter = ImageSplitter(**kwargs)
            
        self.num_dims     = self.splitter.num_dims
        self.input_dim    = self.splitter.output_dim
        self.input_shape  = self.splitter.output_shape
        self.stride       = self.splitter.stride
        self.output_shape = self.splitter.input_shape
        self.keep_chn_dim = self.splitter.keep_chn_dim
        del self.splitter
        
        padded_shape = [h+p for h, p in zip(self.output_shape,self.get_padding())]
        self.fold = SubBatch(
            input_data_dim  =-self.num_dims,
            model           = torch.nn.Fold(
                output_size = padded_shape,
                kernel_size = self.input_shape,
                stride      = self.stride,
            )
        )
        if self.output_shape is None:
            raise ValueError(f"Output shape must be specified")
        
    @classmethod
    def parse_args(cls, *args, **kwargs):
        if 'output_shape' in kwargs.keys():
            input_shape = kwargs['output_shape']
        else :
            input_shape = None
        if 'input_shape' in kwargs.keys():
            output_shape = kwargs['input_shape']
        else :
            output_shape = None
        if 'input_dim' in kwargs.keys():
            output_dim = kwargs['input_dim']
        else :
            output_dim = None
            
        if len(args) >= 1:
            kwargs.update({'output_shape'   : args[0]})
        else :
            kwargs.update({'output_shape'   : output_shape})
            
        if len(args) >= 2:
            kwargs.update({'input_shape'    : args[1]})
        else :
            kwargs.update({'input_shape'    : input_shape})
            
        if len(args) >= 3:
            kwargs.update({'stride'         : args[2]})
        if len(args) >= 4:
            kwargs.update({'stride_percent' : args[3]})
        if len(args) >= 5:
            kwargs.update({'num_strides'    : args[4]})
        if len(args) >= 6:
            kwargs.update({'output_dim'     : args[5]})
        else :
            kwargs.update({'output_dim'     : output_dim})
        if len(args) >= 7:
            kwargs.update({'keep_chn_dim'   : args[6]})
            
        return {
            key:val for key,val in kwargs.items() 
                if key in (
                    'input_shape', 
                    'output_shape', 
                    'stride', 
                    'stride_percent', 
                    'num_strides', 
                    'output_dim', 
                    'padding_val', 
                    'keep_chn_dim',
                )
        }
            
    def get_padding(self) :
        padding = [(self.output_shape[_]-self.input_shape[_]) % self.stride[_] for _ in range(-self.num_dims,0)]
        padding = [(self.stride      [_]-         padding[_]) % self.stride[_] for _ in range(-self.num_dims,0)]
        return tuple(padding)
    
    def remove_padding(self, x):
        padding      = self.get_padding()
        padded_shape = x.shape[-self.num_dims:]
        first_pad = lambda dim: padding[dim] // 2
        last_pad  = lambda dim: padded_shape[dim] - (padding[dim]+1) // 2
        
        for dim in range(self.num_dims):
            x = x.movedim(-self.num_dims,-1)[..., first_pad(dim):last_pad(dim)]
        return x
        
    def forward(self, x: torch.Tensor) :
        assert x.shape[-self.num_dims:] == self.input_shape
        x = x.movedim(self.input_dim,-1)
        x = x.reshape(*x.shape[:-3-self.keep_chn_dim],-1,x.shape[-1])
        return self.remove_padding(self.fold(x))
        
class AbstractSelfAttention(torch.nn.Module):
    def __init__(
        self, 
        key_model       : torch.nn.Module,
        query_model     : torch.nn.Module,
        value_model     : torch.nn.Module,
        sequence_dim    : int | list[int] = 1,
        input_data_dim  =-1,
        output_data_dim =-1,
        dropout_rate    = 0.,
        reduction       : Literal['sum','mean','none'] = 'sum',
        keepdim         : bool = False,
    ):
        super(AbstractSelfAttention,self).__init__()
        if not hasattr(sequence_dim,'__iter__'):
            sequence_dim = [sequence_dim,]
            
        self.in_seq_dim = sequence_dim
        self.key_seq_dim = []
        self.out_seq_dim = []
        
        for s_dim in sequence_dim:
            if s_dim < 0:
                if input_data_dim < 0:
                    self.key_seq_dim.append(s_dim - input_data_dim - 1)
                else :
                    raise ValueError(f'Expected "sequence_dim","input_data_dim" to have the same sign; got {s_dim},{input_data_dim}')
            else :
                self.key_seq_dim.append(s_dim)
            
            if s_dim < 0:
                self.out_seq_dim.append(s_dim - input_data_dim + output_data_dim)
            else :
                self.out_seq_dim.append(s_dim)
        
        self.key_batch = SubBatch(
            input_data_dim  = input_data_dim,
            output_data_dim = -1,
            model           = key_model,
        )
        self.query_batch = SubBatch(
            input_data_dim  = input_data_dim,
            output_data_dim = -1,
            model           = query_model,
        )
        self.value_batch = SubBatch(
            input_data_dim  = input_data_dim,
            output_data_dim = output_data_dim,
            model           = value_model,
        )
        self.drop = torch.nn.Dropout(dropout_rate)
        self.softmax = torch.nn.ModuleList([
            torch.nn.Softmax(s_dim)
                for s_dim in self.key_seq_dim
        ])
        self.keepdim = keepdim
        
        if reduction not in ('sum','mean','none'):
            raise NotImplementedError(f'Reduction strategy "{reduction}" is not supported.')
        else :
            self.reduction = reduction
    
    @classmethod
    def reshape_tensor(cls, input : torch.Tensor, target):
        init_shape = list(input.shape)
        targ_shape = list(target.shape)
        
        for dim, dim_size in enumerate(targ_shape):
            if dim >= len(init_shape):
                init_shape.append(1)
            elif dim_size != init_shape[dim]:
                init_shape = init_shape[:dim] + [1,] + init_shape[dim:]

        return input.reshape(init_shape)

    def forward(self, x):
        # print('Abstract in', x.shape)
        k : torch.Tensor = self.key_batch(x)
        q : torch.Tensor = self.query_batch(x)
        v : torch.Tensor = self.value_batch(x)
        
        weights : torch.Tensor = (k * q).sum(dim=-1) / (k.shape[-1] ** 0.5)
        for softmax in self.softmax:
            weights = softmax(weights)
            
        # print('weights',weights.shape)
        # print('v', v.shape)
        data : torch.Tensor = self.reshape_tensor(weights,v) * v
        if self.reduction == 'sum':
            data = data.sum(dim=self.out_seq_dim, keepdim=self.keepdim)
            # return data.sum(dim=self.out_seq_dim)
        elif self.reduction == 'mean':
            data = data.mean(dim=self.out_seq_dim, keepdim=self.keepdim)
            # return data.mean(dim=self.out_seq_dim)
        # elif self.reduction == 'none':
            
        # print('Abstract out', data.shape)
        return data