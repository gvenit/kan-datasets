"""
FasterKAN: Radial Basis Function-based Kolmogorov-Arnold Networks

This module implements a version of RBF-KANs of Delis, called FasterKAN (Kolmogorov-Arnold Networks using 
Radial Basis Functions) with modifications for better training and hardware implementation.

Modifications compared to the original FasterKAN implementation:
- Dropout with scaling based on the number of grids
- Linear layers without bias for FPGA compatibility
- Gradient scaling for grid and inverse denominator parameters

Architecture:
-------------
FasterKAN consists of a sequence of layers, each containing:
1. A Radial Spline Function (RSF) using tanh-based RBF
2. A dropout layer with rate scaled based on grid count
3. A linear layer (without bias)

The RSF transformation computes:
    f(x) = sech²(σ·(x-μᵢ)) = 1 - tanh²(σ·(x-μᵢ))
where:
    - μᵢ are the grid points
    - σ is the inverse denominator (controlling basis function width)
    - sech² is the squared hyperbolic secant

Example Usage:
-------------
    model = FasterKAN(
        layers_hidden=[784, 100, 10],  # Input, hidden, output dimensions
        num_grids=10,                 # Grid points for RBFs
        grid_min=-3.0,                # Minimum grid value
        grid_max=3.0,                 # Maximum grid value
        inv_denominator=1.0           # Inverse denominator (σ)
    )
    
    # Forward pass
    outputs = model(inputs)

Components:
-------------
- RSWAFFunction: Autograd function for RBF computation
- RSF: Radial Spline Function module used as a wrapper for the RSWAFFunction
- FasterKANLayer: Single Layer combining RSF, dropout, and linear transformation
- FasterKAN: Main model class, that can consists of many different FasterKANLayers
"""
from warnings import warn
import torch
import torch.nn as nn
from typing import *
from torch.autograd import Function
from ..utils import expand_value

USE_BIAS_ON_LINEAR = False  # NOTE: Bias must be false to be able to implement on fpga

class RSWAFFunction(Function):
    """
    Autograd function for Radial Spline Wavelet Activation Function.
    
    Computes the derivative of tanh((x-grid)*inv_denominator) with respect to x,
    which is sech²((x-grid)*inv_denominator) = 1 - tanh²((x-grid)*inv_denominator).
    
    The backward pass:
    1. Scales gradients for grid and inv_denominator parameters by 10
    2. Allows selective training of grid and inv_denominator parameters
    """
    @staticmethod
    def forward(ctx, input, grid, inv_denominator):
        """
        Args:
            input (torch.Tensor): Input tensor [batch_size, input_dim]
            grid (torch.Tensor): Grid points [num_grids]
            inv_denominator (torch.Tensor): Inverse denominator
            
        Returns:
            torch.Tensor: sech²((x-grid)*inv_denominator) values
        """
        diff = (input[..., None] - grid)
        diff_mul = diff.mul(inv_denominator) 
        tanh_diff = torch.tanh(diff_mul)
        tanh_diff_deriviative = 1. - tanh_diff ** 2  # sech^2(x) = 1 - tanh^2(x)

        ctx.save_for_backward(inv_denominator, diff_mul, tanh_diff, tanh_diff_deriviative) # Save tensors for backward pass

        return tanh_diff_deriviative
    
    @staticmethod
    def backward(ctx, grad_output,train_grid: bool = True, train_inv_denominator: bool = True, gradient_boost=1):
        """
        Args:
            ctx: Context from forward pass
            grad_output (torch.Tensor): Gradient from downstream layers
            train_grid (bool): Whether to compute gradients for grid points
            train_inv_denominator (bool): Whether to compute gradients for inv_denominator
            
        Returns:
            tuple: Gradients for input, grid, and inv_denominator
        """
        inv_denominator, diff_mul, tanh_diff, tanh_diff_deriviative = ctx.saved_tensors
        grad_grid = grad_inv_denominator = None
        
        deriv = -2 * inv_denominator * tanh_diff * tanh_diff_deriviative * grad_output

        # Compute the backward pass for the input
        grad_input =  deriv.sum(dim=-1)
        ctx.train_grid = train_grid
        ctx.train_inv_denominator = train_inv_denominator

        # Compute the backward pass for grid
        if ctx.train_grid:
            grad_grid = - gradient_boost * deriv.sum(dim=-2) # NOTE: We boost the gradient by 10 to make it more significant

        # Compute the backward pass for inv_denominator        
        if ctx.train_inv_denominator:
            grad_inv_denominator = gradient_boost * (diff_mul * deriv).sum(0) # NOTE: We boost the gradient by 10 to make it more significant

            if inv_denominator.view(-1).size(0) == 1 :
                grad_inv_denominator = grad_inv_denominator.sum()
                
        return grad_input, grad_grid, grad_inv_denominator

class RSF(nn.Module):
    """
    Args:
        train_grid (bool): Whether to update grid points during training
        train_inv_denominator (bool): Whether to update inv_denominator during training
        grid_min (float): Minimum value for grid points
        grid_max (float): Maximum value for grid points
        num_grids (int): Number of grid points to use
        inv_denominator (float): Initial value for inverse denominator parameter
    
    Attributes:
        grid (nn.Parameter): Learnable grid points evenly spaced from grid_min to grid_max
        inv_denominator (nn.Parameter): Learnable inverse denominator controlling RBF width
    """
    def __init__(
        self,
        train_grid: bool,        
        train_inv_denominator: bool,
        grid_min: float,
        grid_max: float,
        num_grids: int,
        inv_denominator: float
    ):
        super(RSF,self).__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids).float()

        self.train_grid = train_grid
        self.train_inv_denominator = train_inv_denominator

        self.grid = torch.nn.Parameter(grid, requires_grad=train_grid)
        self.inv_denominator = torch.nn.Parameter(torch.tensor(inv_denominator).float(), requires_grad=train_inv_denominator)  # Cache the inverse of the denominator

    # def to(self, device, **kwagrs):
    #     self.grid = self.grid.to(device)
    #     self.inv_denominator = self.inv_denominator.to(device)
        
    #     super().to(device = device, **kwagrs)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Transformed tensor [batch_size, input_dim, num_grids]
        """
        return RSWAFFunction.apply(x, self.grid, self.inv_denominator) # returns tanh_diff_derivative

class RSFAuto(nn.Module):
    """
    Args:
        train_grid (bool): Whether to update grid points during training
        train_inv_denominator (bool): Whether to update inv_denominator during training
        grid_min (float): Minimum value for grid points
        grid_max (float): Maximum value for grid points
        num_grids (int): Number of grid points to use
        inv_denominator (float): Initial value for inverse denominator parameter
    
    Attributes:
        grid (nn.Parameter): Learnable grid points evenly spaced from grid_min to grid_max
        inv_denominator (nn.Parameter): Learnable inverse denominator controlling RBF width
    """
    def __init__(
        self,
        train_grid: bool,        
        train_inv_denominator: bool,
        grid_min: float,
        grid_max: float,
        num_grids: int,
        inv_denominator: float,
        mode : Literal['RSWAFF','tanh','tanh2','gaussian', 'sigmoid','square','triangle','sample'] = 'RSWAFF'
    ):
        super(RSFAuto,self).__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids).float()
        self.grid = torch.nn.Parameter(grid, requires_grad=train_grid)
        self.inv_denominator = torch.nn.Parameter(torch.tensor(inv_denominator).float(), requires_grad=train_inv_denominator)  # Cache the inverse of the denominator
        self.mode = mode
        if   self.mode == 'RSWAFF':
            self.rbf = lambda x : 1 - torch.nn.functional.tanh(x) ** 2
        elif self.mode == 'tanh':
            self.rbf = lambda x : torch.nn.functional.tanh(x)
        elif self.mode == 'tanh2':
            self.rbf = lambda x : torch.nn.functional.tanh(x) ** 2
        elif self.mode == 'gaussian':
            self.rbf = lambda x : torch.exp(-(x**2))
        elif self.mode == 'sigmoid':
            self.rbf = lambda x : torch.nn.functional.sigmoid(x)
        # elif self.mode == 'square':
        #     self.rbf = lambda x, threshold=0.5 : torch.where(x.abs() < threshold, 1., 0.)
        # elif self.mode == 'triangle':
        #     self.rbf = lambda x, threshold=0.5 : torch.where(x.abs() < threshold, -x, 0.)
        elif self.mode == 'sample':
            self.rbf = lambda x, guard=1e-8 : torch.sin(x+guard) / (x+guard)
        else :
            raise ValueError(f"Mode is not implemented; got '{self.mode}'")

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Transformed tensor [batch_size, input_dim, num_grids]
        """
        diff = (x[..., None] - self.grid).mul(self.inv_denominator) 
        return self.rbf(diff)

class FasterKANLayer(nn.Module):
    """
    A single layer in the FasterKAN architecture.
    
    The layer applies the following sequence:
    1. Transform inputs using Radial Spline Functions (RSF)
    2. Apply dropout with rate based on grid count (1-0.75^num_grids)
    3. Apply linear transformation to the outputs
    
    Args:
        train_grid (bool): Whether to update grid points during training
        train_inv_denominator (bool): Whether to update inv_denominator during training
        input_dim (int): Dimensionality of input features
        output_dim (int): Dimensionality of output features
        grid_min (float): Minimum value for grid points
        grid_max (float): Maximum value for grid points
        num_grids (int): Number of grid points to use
        inv_denominator (float): Initial value for inverse denominator parameter
    
    Attributes:
        rbf (RSF): Radial Spline Function module
        drop (nn.Dropout): Dropout layer with adaptive rate
        linear (nn.Linear): Linear transformation without bias
    """
    def __init__(
        self,
        train_grid: bool,        
        train_inv_denominator: bool,
        input_dim: int,
        output_dim: int,
        grid_min: float,
        grid_max: float,
        num_grids: int,
        inv_denominator: float,
        mode : Literal['RSWAFF','tanh','tanh2','gaussian', 'sigmoid','square','triangle','sample'] = 'RSWAFF',
    ) -> None:
        super(FasterKANLayer,self).__init__()

        self.rbf = RSFAuto(train_grid, train_inv_denominator,grid_min, grid_max, num_grids, inv_denominator, mode=mode)
        self.linear = nn.Linear(input_dim * num_grids, output_dim, bias=USE_BIAS_ON_LINEAR) 
        self.drop = nn.Dropout(0.5) # NOTE: Dropout rate increases with num_grids
        # self.drop = nn.Dropout(1-0.9**(num_grids)) # NOTE: Dropout rate increases with num_grids

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, output_dim]
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        spline_basis = self.rbf(x).view(batch_size, -1)
        spline_basis = self.drop(spline_basis)
        output = self.linear(spline_basis)
        return output

# @torch.compile
class FasterKAN(nn.Module):
    """
    FasterKAN: Radial Basis Function-based Kolmogorov-Arnold Network.
    This model stacks multiple FasterKANLayers to create a deep RBF-KAN architecture.
    
    Args:
        layers_hidden (List[int]): List of layer dimensions including input and output dimensions
            e.g., [784, 100, 10] for MNIST classification with one hidden layer
        num_grids (Union[int, List[int]]): Number of grid points for each layer
            If a single int is provided, it's used for all layers
        grid_min (float): Minimum value for grid points
        grid_max (float): Maximum value for grid points
        inv_denominator (float): Initial value for inverse denominator parameter
    
    Attributes:
        train_grid (bool): Whether grid points are being updated during training
        train_inv_denominator (bool): Whether inv_denominator is being updated during training
        layers (nn.ModuleList): List of FasterKANLayer modules
        is_eval (bool): Whether the model is in evaluation mode
    
    Example:
        ```python
        model = FasterKAN([784, 100, 10], num_grids=10, grid_min=-3.0, grid_max=3.0, inv_denominator=1.0)
        output = model(input_tensor)  # Shape: [batch_size, 10]
        ```
    """
    def __init__(
        self, layers_hidden: List[int], 
        num_grids: Union[int, List[int]],
        grid_min: float,
        grid_max: float,
        inv_denominator: float,
        mode : Literal['RSWAFF','tanh','tanh2','gaussian', 'sigmoid','square','triangle','sample'] = 'RSWAFF',
        residual : list[bool] = False
    ):
        super(FasterKAN, self).__init__()

        self.train_grid = True
        self.train_inv_denominator = True
        
        num_grids       = expand_value(num_grids,       len(layers_hidden)-1)
        grid_min        = expand_value(grid_min,        len(layers_hidden)-1)
        grid_max        = expand_value(grid_max,        len(layers_hidden)-1)
        inv_denominator = expand_value(inv_denominator, len(layers_hidden)-1)
        residual        = expand_value(residual,        len(layers_hidden)-1)
        
        self.residual   = []
        for _iter, residual_i in enumerate(residual):
            if residual_i :
                if layers_hidden[_iter] == layers_hidden[_iter+1] :
                    self.residual.append(True)
                else :
                    warn(f"Skipped residual connection at layer {_iter}; Number of features do not match ({layers_hidden[_iter]} != {layers_hidden[_iter+1]})")
                    self.residual.append(False)
            else :
                self.residual.append(False)
        
        self.layers = nn.ModuleList([
            FasterKANLayer(
                train_grid=self.train_grid,
                train_inv_denominator=self.train_inv_denominator,
                input_dim=in_dim, 
                output_dim=out_dim, 
                grid_min=grid_min_i,
                grid_max=grid_max_i,
                num_grids=num_grids_i,
                inv_denominator=inv_denominator_i,
                mode=mode
            ) for _iter, (
                num_grids_i, 
                in_dim, 
                out_dim, 
                grid_min_i, 
                grid_max_i, 
                inv_denominator_i, 
            ) in enumerate(zip(
                num_grids, 
                layers_hidden[:-1], 
                layers_hidden[1:],
                grid_min,
                grid_max,
                inv_denominator,
            ))
        ])

    def eval(self):
        """
        Set the model to evaluation mode, disabling grid and inv_denominator parameter updates.
        """
        self.is_eval = True
        self.train_grid = False
        self.train_inv_denominator = False
        super().eval()

    def train(self, mode=True):
        """
        Set the model to training mode, enabling updates to grid and inv_denominator parameters.
        
        Args:
            mode (bool): Whether to enable training mode (True) or evaluation mode (False)
        """
        self.is_eval = not mode
        self.train_grid = mode
        self.train_inv_denominator = mode
        super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, output_dim]
        """
        for layer, res in zip(self.layers, self.residual):
            if res:
                x = layer(x) + x
            else :
                x = layer(x)
        return x
    
# # @torch.compile
# class InterleavedFasterKAN(nn.Module):
#     """
#     InterleavedFasterKAN: Radial Basis Function-based Kolmogorov-Arnold Network.
#     This model stacks multiple FasterKANLayers to create a deep RBF-KAN architecture.
    
#     Args:
#         layers_hidden (List[int]): List of layer dimensions including input and output dimensions
#             e.g., [784, 100, 10] for MNIST classification with one hidden layer
#         num_grids (Union[int, List[int]]): Number of grid points for each layer
#             If a single int is provided, it's used for all layers
#         grid_min (float): Minimum value for grid points
#         grid_max (float): Maximum value for grid points
#         inv_denominator (float): Initial value for inverse denominator parameter
    
#     Attributes:
#         train_grid (bool): Whether grid points are being updated during training
#         train_inv_denominator (bool): Whether inv_denominator is being updated during training
#         layers (nn.ModuleList): List of FasterKANLayer modules
#         is_eval (bool): Whether the model is in evaluation mode
    
#     Example:
#         ```python
#         model = FasterKAN([784, 100, 10], num_grids=10, grid_min=-3.0, grid_max=3.0, inv_denominator=1.0)
#         output = model(input_tensor)  # Shape: [batch_size, 10]
#         ```
#     """
#     def __init__(
#         self, layers_hidden: List[int], 
#         num_grids: Union[int, List[int]],
#         grid_min: float,
#         grid_max: float,
#         inv_denominator: float
#     ):
#         super(InterleavedFasterKAN, self).__init__()

#         self.train_grid = True
#         self.train_inv_denominator = True
        
#         num_grids       = expand_value(num_grids,       len(layers_hidden)-1)
#         grid_min        = expand_value(grid_min,        len(layers_hidden)-1)
#         grid_max        = expand_value(grid_max,        len(layers_hidden)-1)
#         inv_denominator = expand_value(inv_denominator, len(layers_hidden)-1)

#         self.layers = nn.ModuleList([
#             FasterKANLayer(
#                 train_grid=self.train_grid,
#                 train_inv_denominator=self.train_inv_denominator,
#                 input_dim=in_dim, 
#                 output_dim=out_dim, 
#                 grid_min=grid_min_i,
#                 grid_max=grid_max_i,
#                 num_grids=num_grids_i,
#                 inv_denominator=inv_denominator_i
#             ) for _iter, (
#                 num_grids_i, 
#                 in_dim, 
#                 out_dim, 
#                 grid_min_i, 
#                 grid_max_i, 
#                 inv_denominator_i, 
#             ) in enumerate(zip(
#                 num_grids, 
#                 layers_hidden[:-1], 
#                 layers_hidden[1:],
#                 grid_min,
#                 grid_max,
#                 inv_denominator,
#             ))
#         ])

#     def eval(self):
#         """
#         Set the model to evaluation mode, disabling grid and inv_denominator parameter updates.
#         """
#         self.is_eval = True
#         self.train_grid = False
#         self.train_inv_denominator = False
#         super().eval()

#     def train(self, mode=True):
#         """
#         Set the model to training mode, enabling updates to grid and inv_denominator parameters.
        
#         Args:
#             mode (bool): Whether to enable training mode (True) or evaluation mode (False)
#         """
#         self.is_eval = not mode
#         self.train_grid = mode
#         self.train_inv_denominator = mode
#         super().train(mode)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x (torch.Tensor): Input tensor [batch_size, input_dim]
            
#         Returns:
#             torch.Tensor: Output tensor [batch_size, output_dim]
#         """
#         for layer in self.layers:
#             x = layer(x)
#         return x