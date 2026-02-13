import torch
import torch.nn as nn

class LambdaModule(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def extra_repr(self):
        return f'func={self.func}'
        
    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)
    
class RSWAFF(nn.Module):
    def __init__(self):
        super(RSWAFF, self).__init__()
        self.tanh = torch.nn.Tanh()
        
    def forward(self, x):
        return torch.ones_like(x) - self.tanh(x) ** 2