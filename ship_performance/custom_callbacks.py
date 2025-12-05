from typing import Iterable
import torch
import numpy as np

class MaskInput :
    def __init__(
        self, 
        input : list = None,
        input_categories = None,
        max_probability = 0.25,
        x_shift = 0.,
        masked_value = -1,
    ):
        self.max_probability = max_probability
        self.x_shift = x_shift
        self.masked_value = torch.tensor(masked_value)
        
        self.input = input
        self.input_categories = input_categories
        
        if self.input_categories is None:
            self.input_categories = []
            
        if self.input is None:
            self.input_categories = []
            
        else :
            self._initialize_input(input)
    
    def _initialize_input(self, input) :
            self.input = {
                key : idx for idx, key in enumerate(self.input) 
            }
            self.input_categories = [[
                    self.input[key] for key in category
                ] for category in self.input_categories
            ]
            self.input_regression_type = self.input.values()
            
            for category in self.input_categories:
                self.input_regression_type = [
                    idx for idx in self.input_regression_type
                        if idx not in category
                ] 
            self._initialize_input = lambda input : None
            
    def __call__(
        self,
        data        : torch.Tensor, 
        epoch       : int,
        epochs      : int,
        dataloader  : Iterable,
        iteration   : int = None,
        device      = torch.device('cpu'),
        **kwargs
    ) :
        self._initialize_input(data.shape[-1])
        if iteration is None:
            iteration = len(dataloader)
            
        probability = self.max_probability / (1 + np.exp((epoch / epochs) * (iteration / len(dataloader)) - self.x_shift))
        
        mask = torch.rand(len(self.input_regression_type) + len(self.input_categories), device = device) < float(probability)
        
        mask_extended = torch.empty(len(self.input), device = device, dtype = mask.dtype)
        mask_extended[self.input_regression_type] = mask[:len(self.input_regression_type)]
        
        for offset, category in enumerate(self.input_categories):
            mask_extended[category] = mask[len(self.input_regression_type) + offset]
        
        data.masked_fill_(mask_extended, self.masked_value.to(data.dtype).to(device))