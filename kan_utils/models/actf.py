import torch.nn as nn

class LambdaModule(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)