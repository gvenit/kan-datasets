from typing import overload, Literal, Mapping, Callable, Iterable, Any, TypeVar, Generic
import torch
import torch.nn as nn

class SubBatch(nn.Module):
    '''
    A wrapper model that compresses multiple batch dimensions for models that expect a single batch dimension.
    After the model execution, batches are decompressed to their original batch dimensionality.

    :param input_data_dim: The position of the firt input data dimension.
    :type input_data_dim: int
    :param model: The model to wrap around
    :type model: Callable
    '''
    def __init__(self, input_data_dim, output_data_dim = None,  model = None):
        super(SubBatch,self).__init__()
        
        self.model = model
        self.input_data_dim  = int(input_data_dim)
        
    def extra_repr(self):
        return f"input_data_dim={self.input_data_dim}"
        
    def forward(self, x : torch.Tensor, *args, **kwargs):
        batch_shape = x.shape[:self.input_data_dim]
        x = x.reshape(-1, *x.shape[self.input_data_dim:])
        x = self.model(x, *args, **kwargs)
        
        if isinstance(x, tuple):
            x, *args = x
            x = x.reshape(*batch_shape, *x.shape[1:])
            x = x, *args
            
        else :
            x = x.reshape(*batch_shape, *x.shape[1:])
            
        return x

class Reshaper(nn.Module):
    '''
    A wrapper model that reshapes data dimensions.
    
    :param input_data_shape: The input data shape.
    :type input_data_shape: int
    :param output_data_shape: The output data (result) shape.
    :type output_data_shape: int
    '''
    def __init__(self, input_data_shape, output_data_shape):
        super(Reshaper,self).__init__()
        self.input_data_shape  = input_data_shape
        self.output_data_shape = output_data_shape
        
        assert len(torch.empty(self.input_data_shape).reshape(-1)) == len(torch.empty(self.output_data_shape).reshape(-1)),\
            f"Total number of elements missmatch; {len(torch.empty(self.input_data_shape).reshape(-1))} != {len(torch.empty(self.output_data_shape).reshape(-1))}"
        
    def extra_repr(self):
        return f"input_data_shape={self.input_data_shape}, output_data_shape={self.output_data_shape}"
        
    def forward(self, x : torch.Tensor):
        batch_shape = x.shape[:-len(self.input_data_shape)]
        x = x.reshape(*batch_shape, *self.output_data_shape)
        return x
    
class RangeTransform(nn.Module):
    '''
    A wrapper model that applies a range transformation to the input data.

    :param data_min: The minimum value of the input data.
    :type data_min: float
    :param data_max: The maximum value of the input data.
    :type data_max: float
    :param target_min: The minimum value of the target data.
    :type target_min: float
    :param target_max: The maximum value of the target data.
    :type target_max: float
    '''
    def __init__(self, data_min = 0, data_max = 1, target_min = 0, target_max = 1):
        super(RangeTransform,self).__init__()
        self.data_min       = data_min
        self.data_max     = data_max
        self.target_min     = target_min
        self.target_max     = target_max

    def to(self, device):
        self.data_min     = self.data_min.to(device)
        self.data_max   = self.data_max.to(device)
        self.target_min   = self.target_min.to(device)
        self.target_max = self.target_max.to(device)
        return super().to(device)

    def forward(self, x : torch.Tensor):
        x = (x - self.data_min) / (self.data_max - self.data_min)
        x = x * (self.target_max - self.target_min) + self.target_min
        return x
    
class Parameterizer(nn.Module):
    '''
    A wrapper model that converts a tensor into a learnable parameter.
    
    :param module: The module to wrap around
    :type module: Callable
    :param kwargs: The keyword arguments for the module. Each argument should be a tuple of (is_param: bool, value: Any)
    :type kwargs: dict[str, tuple[bool, Any]]
    '''
    def __init__(self, module, *args, **kwargs):
        super(Parameterizer,self).__init__()
        new_args = []
        for _iter, (is_param, *value) in enumerate(args):
            if is_param:
                param = nn.Parameter(torch.tensor(value))
                setattr(self, f"arg_{_iter}", param)
            else:
                setattr(self, f"arg_{_iter}", *value)
            new_args.append(getattr(self, f"arg_{_iter}"))
            
        new_kwargs = {}
        for key, (is_param, *value) in kwargs.items():
            if is_param:
                param = nn.Parameter(torch.tensor(value))
                setattr(self, key, param)
            else:
                setattr(self, key, *value)
            new_kwargs[key] = getattr(self, key)
        self.module = module(*new_args,**new_kwargs)
        
    def to(self, device):
        for key, value in self.__dict__.items():
            if isinstance(value, (torch.Tensor, nn.Parameter)):
                setattr(self, key, value.to(device))
        self.module = self.module.to(device)
        return super().to(device)
        
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
        
class MultiHead(nn.Module):
    '''
    Apply multiple modules to the same input.
    '''
    @overload
    def __init__(
        self , 
        *args : Iterable[Callable],
        return_type : Literal['dict', 'list'] = 'list',
    ):
        '''
        :param args: Iterable of models to apply
        :type args: Iterable[Callable]
        '''
        ...
        
    @overload
    def __init__(
        self , 
        args : Mapping[str, Callable],
        return_type : Literal['dict', 'list'] = 'dict',
        ):
        '''
        :param args: Mapping of models to apply
        '''
        ...
        
    def __init__(
        self, 
        *args,
        return_type : Literal['dict', 'list'] = 'list',
        ):
        super().__init__()
        if return_type in ('dict','list'):
            self.return_type = return_type
        else :
            raise NotImplementedError(f'Return type "{return_type}" is not supported.')
        
        if isinstance(args[0], Mapping):
            self.heads = nn.ModuleDict(args[0])
            
        else :
            self.heads = nn.ModuleList(args)
           
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        return getattr(self.heads, name)
       
    # def to(self, device):
    #     # print('MultiHead called device.')
    #     self.heads = self.heads.to(device)
    #     return super().to(device)
            
    def forward(self, *args, **kwargs):
        output = tuple(
            head(*args,**kwargs)
                for head in (
                    self.heads if isinstance(self.heads, nn.ModuleList) else 
                    self.heads.values() 
                )
        )
        if self.return_type == 'list' :
            return output
        elif self.return_type == 'dict':
            keys = self.heads.keys() if isinstance(self.heads, nn.ModuleDict) else [
                f'head.{_iter}' for _iter in range(len(self.heads))
            ]
            return {
                key : val for key, val in zip(keys, output)
            }
        else :
            raise NotImplementedError(f'Return type "{self.return_type}" is not supported.')
