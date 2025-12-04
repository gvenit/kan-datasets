from typing import Literal
import torch

class MixedLoss(torch.nn.Module):
    '''A wrapper for wrapping criteria when the predicted and target values
    contain one-hot encoded data and regression-type data.
    '''
    def __init__(
        self, 
        output_cols : list, 
        categories : list[list[str]], 
        categoriesLoss = torch.nn.BCEWithLogitsLoss(), 
        regressionLoss = torch.nn.MSELoss(),
        reduction : Literal['sum','mean','random','none'] = 'sum'
    ):
        '''
        Parameters
        ----------
        output_cols : list
            The columns to expect in the predicted and target values in order of appearance. 
        categories : list[list[str]]
            The columns of `output_cols` grouped in categories. Columns not present in `categories`
            are treated as regression-type data.
        categoriesLoss = torch.nn.BCEWithLogitsLoss(), 
            The loss to use in categorized columns.
        regressionLoss = torch.nn.MSELoss()
            The loss to use in regression-type columns.
        '''
        super(MixedLoss,self).__init__()
        
        self.categoriesLoss = categoriesLoss
        self.regressionLoss = regressionLoss
        
        self.categoriesCols = [
            [output_cols.index(label) for label in group_i ]
                for group_i in categories
        ]
        self.regressionCols = []
        
        for group_i in categories:
            self.regressionCols.extend(group_i)
        
        self.regressionCols = [
            output_cols.index(label) 
                for label in output_cols 
                if label not in self.regressionCols
        ]
        self.reduction = reduction
        if self.reduction not in ('sum','mean','random','none'):
            raise ValueError(f'Unrecognised reduction type; got {self.reduction}')
        
    def forward(self, pred : torch.Tensor, targ : torch.Tensor):
        loss = torch.stack([
            self.regressionLoss(pred[:,self.regressionCols], targ[:,self.regressionCols]),
            *[
                self.categoriesLoss(pred[:,group_i], targ[:,group_i])
                    for group_i in self.categoriesCols
        ]])
        
        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'random':
            return loss[int( torch.rand((1,)) * len(loss))]
        elif self.reduction == 'none':
            return loss
        else :
            raise ValueError(f'Unrecognised reduction type; got {self.reduction}')
        
class TestLoss(torch.nn.Module):
    def __init__(self, basis = torch.nn.HuberLoss, epsilon = .1, reduction : Literal['none','sum','mean'] = 'mean'):
        super(TestLoss,self).__init__()
        # self.threshold_diff = threshold
        self.threshold_targ = 1e-3
        self.threshold_norm = 1e-2
        self.epsilon = epsilon
        self.reduction = reduction
        
        self.basis = basis(reduction='none')
        
    def forward(self, input, target):
        mask_targ = target.abs() < self.threshold_targ
        
        diff = input-target
        # mask_diff = diff.abs() < self.threshold_diff
        
        ndiff = (~mask_targ) * diff / (target.abs() + mask_targ)
        mask_norm = ((ndiff).abs() < self.threshold_norm)
        
        loss  = self.basis(diff, torch.zeros_like(diff))
        nloss = self.basis(ndiff, torch.zeros_like(diff))
        
        mask = mask_targ + mask_norm
        loss = loss * mask + nloss * (~mask)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else :
            return loss