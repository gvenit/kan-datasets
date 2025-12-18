import torch
import torch.nn as nn

class SubBatch(nn.Module):
    '''A wrapper model that compresses multiple batch dimensions for models that expect a single batch dimension.
    After the model execution, batches are decompressed to their original batch dimensionality.
    
    Args
    ----
    input_data_dim: int
        The position of the firt input data dimension.
    output_data_dim: int
        The position of the firt output data (result) dimension.
    model: Callable
        The model to wrap around
    '''
    def __init__(self, input_data_dim, output_data_dim, model):
        super(SubBatch,self).__init__()
        
        self.model = model
        self.input_data_dim = input_data_dim
        self.output_data_dim = output_data_dim
        
    def forward(self, x : torch.Tensor, *args, **kwargs):
        batch_shape = x.shape[:self.input_data_dim]
        x = self.model(x.reshape(-1, *x.shape[self.input_data_dim:]), *args, **kwargs)
        
        if isinstance(x, tuple):
            x, *args = x
            x = x.reshape(batch_shape, x.shape[self.output_data_dim:])
            x = x, *args
            
        else :
            x = x.reshape(batch_shape, x.shape[self.output_data_dim:])
            
        return x