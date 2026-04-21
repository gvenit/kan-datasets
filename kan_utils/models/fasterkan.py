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
        hidden_layers=[784, 100, 10],  # Input, hidden, output dimensions
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

class RadialBasisFunction(nn.Module):
    def __init__(self, mode):
        super(RadialBasisFunction, self).__init__()
        # Normalize mode to handle case variations
        mode_normalized = mode.upper() if isinstance(mode, str) else mode
        
        if   mode_normalized == 'RSWAFF':
            from ..models import RSWAFF
            self.rbf = RSWAFF()
            self.mode = 'RSWAFF'
        elif mode_normalized == 'PRELU':
            from ..models import PReLUGlobalParam
            self.rbf = PReLUGlobalParam()
            self.mode = 'PReLU'
        elif mode_normalized == 'TANH2':
            self.mode = mode
            self.rbf = lambda x : torch.nn.functional.tanh(x) ** 2
        elif mode_normalized == 'GAUSSIAN':
            self.mode = mode
            self.rbf = lambda x : torch.exp(-(x**2))
        elif mode_normalized == 'SAMPLE':
            self.mode = mode
            self.rbf = lambda x, guard=1e-8 : torch.sin(x+guard) / (x+guard)
        elif hasattr(mode, '__call__'):
            self.rbf = mode
            self.mode = str(mode)
        elif hasattr(torch.nn, mode):
            self.rbf = getattr(torch.nn, mode)()
            self.mode = mode
        else :
            raise ValueError(f"Mode is not implemented; got '{mode}'")
        
    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        if hasattr(self, 'mode'):
            return f"mode={self.mode}"
        return ''

    def forward(self, x):
        return self.rbf(x)

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
        :param ctx: Context to save information for backward pass
        :param input: Input tensor [batch_size, input_dim]
        :type input: torch.Tensor 
        :param grid: Grid points [num_grids]
        :type grid: torch.Tensor
        :param inv_denominator: Inverse denominator
        :type inv_denominator: torch.Tensor
        :return: sech²((x-grid)*inv_denominator) values
        :rtype: torch.Tensor 
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
        :param ctx: Context from forward pass
        :param grad_output: Gradient from downstream layers
        :type grad_output: torch.Tensor
        :param train_grid: Whether to compute gradients for grid points
        :type train_grid: bool
        :param train_inv_denominator: Whether to compute gradients for inv_denominator
        :type train_inv_denominator: bool
        :param gradient_boost: Factor to scale gradients for grid and inv_denominator
        :type gradient_boost: int
        :return: Gradients for input, grid, and inv_denominator
        :rtype: tuple(torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor])
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
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.scale = inv_denominator
        
        grid = torch.linspace(grid_min, grid_max, num_grids).float()

        self.train_grid = train_grid
        self.train_inv_denominator = train_inv_denominator

        self.grid = torch.nn.Parameter(grid, requires_grad=train_grid)
        self.inv_denominator = torch.nn.Parameter(torch.tensor(inv_denominator).float(), requires_grad=train_inv_denominator)  # Cache the inverse of the denominator

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"num_grids={self.num_grids}, grid_min={self.grid_min}, grid_max={self.grid_max}, inv_denominator={self.scale}"

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
    :param train_grid: Whether to update grid points during training
    :type train_grid: bool
    :param train_inv_denominator: Whether to update inv_denominator during training
    :type train_inv_denominator: bool
    :param grid_min: Minimum value for grid points
    :type grid_min: float
    :param grid_max: Maximum value for grid points
    :type grid_max: float
    :param num_grids: Number of grid points to use
    :type num_grids: int
    :param inv_denominator: Initial value for inverse denominator parameter
    :type inv_denominator: float
    :param mode: Type of radial basis function to use
    :type mode: Literal['RSWAFF','tanh','tanh2','gaussian', 'sigmoid','square','triangle','sample']
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
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.scale = inv_denominator
        
        grid = torch.linspace(grid_min, grid_max, num_grids).float()
        self.grid = torch.nn.Parameter(grid, requires_grad=train_grid)
        self.inv_denominator = torch.nn.Parameter(torch.tensor(inv_denominator).float(), requires_grad=train_inv_denominator)  # Cache the inverse of the denominator
        self.rbf = RadialBasisFunction(mode)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"num_grids={self.num_grids}, grid_min={self.grid_min}, grid_max={self.grid_max}, inv_denominator={self.scale}"

    def forward(self, x):
        """
        :param x: Input tensor [batch_size, input_dim]
        :type x: torch.Tensor
        :return: Transformed tensor [batch_size, input_dim, num_grids]
        :rtype: torch.Tensor
        """
        diff = (x[..., None] - self.grid).mul(self.inv_denominator) 
        return self.rbf(diff)


class RSFAutoV2(nn.Module):
    """
    Radial basis function layer with per‑input learnable parameters.

    :param input_dim: Number of input features
    :type input_dim: int
    :param train_grid: Whether to update grid points during training
    :type train_grid: bool
    :param train_inv_denominator: Whether to update inv_denominator during training
    :type train_inv_denominator: bool
    :param grid_min: Minimum value for grid points (same for all inputs)
    :type grid_min: float
    :param grid_max: Maximum value for grid points (same for all inputs)
    :type grid_max: float
    :param num_grids: Number of grid points per input
    :type num_grids: int
    :param inv_denominator: Initial value for the inverse denominator (same for all inputs)
    :type inv_denominator: float
    :param mode: Type of radial basis function
    :type mode: Literal['RSWAFF','tanh','tanh2','gaussian','sigmoid','square','triangle','sample']
    """
    def __init__(
        self,
        input_dim: int,
        train_grid: bool,
        train_inv_denominator: bool,
        grid_min: float,
        grid_max: float,
        num_grids: int,
        inv_denominator: float,
        mode: Literal['RSWAFF','tanh','tanh2','gaussian','sigmoid','square','triangle','sample'] = 'RSWAFF'
    ):
        super(RSFAutoV2, self).__init__()
        self.input_dim = input_dim
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        self.scale = inv_denominator

        # Create grid: one set of num_grids points per input dimension
        # Initialize by repeating the same linspace for each input
        base_grid = torch.linspace(grid_min, grid_max, num_grids).float()  # shape: (num_grids,)
        grid = base_grid.unsqueeze(0).expand(input_dim, -1).contiguous()   # shape: (input_dim, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=train_grid)

        # Inverse denominator: one scalar per input dimension
        inv_den = torch.full((input_dim,), inv_denominator, dtype=torch.float32)  # shape: (input_dim,)
        self.inv_denominator = nn.Parameter(inv_den, requires_grad=train_inv_denominator)

        # Radial basis function (applied elementwise)
        self.rbf = RadialBasisFunction(mode)

    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, num_grids={self.num_grids}, "
                f"grid_min={self.grid_min}, grid_max={self.grid_max}, "
                f"inv_denominator={self.scale}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape [batch_size, input_dim]
        :return: Transformed tensor of shape [batch_size, input_dim, num_grids]
        """
        # x: (batch, input_dim)
        # grid: (input_dim, num_grids) -> unsqueeze(0) to (1, input_dim, num_grids)
        # inv_denominator: (input_dim,) -> unsqueeze(0).unsqueeze(-1) to (1, input_dim, 1)
        diff = (x.unsqueeze(-1) - self.grid.unsqueeze(0)) * self.inv_denominator.unsqueeze(0).unsqueeze(-1)
        return self.rbf(diff)


class FasterKANLayerV2(nn.Module):
    """
    A single layer in the FasterKAN architecture using RSFAutoV2.
    
    Similar to FasterKANLayer but uses RSFAutoV2 which has per-input-dimension 
    learnable parameters (grid and inv_denominator) instead of shared ones.
    
    The layer applies the following sequence:
    1. Transform inputs using Radial Spline Functions with per-input parameters (RSFAutoV2)
    2. Apply dropout
    3. Apply linear transformation to the outputs

    :param train_grid: Whether to update grid points during training
    :type train_grid: bool
    :param train_inv_denominator: Whether to update inv_denominator during training
    :type train_inv_denominator: bool
    :param input_dim: Dimensionality of input features
    :type input_dim: int
    :param output_dim: Dimensionality of output features
    :type output_dim: int
    :param grid_min: Minimum value for grid points (same for all inputs)
    :type grid_min: float
    :param grid_max: Maximum value for grid points (same for all inputs)
    :type grid_max: float
    :param num_grids: Number of grid points per input
    :type num_grids: int
    :param inv_denominator: Initial value for inverse denominator (same for all inputs)
    :type inv_denominator: float
    :param mode: Type of radial basis function to use
    :type mode: Literal['RSWAFF','tanh','tanh2','gaussian', 'sigmoid','square','triangle','sample']
    :param dropout_rate: Dropout rate to use on RBF output; if None, defaults to 0.5
    :type dropout_rate: Optional[float]
    :param dropout_linear: Dropout rate to use on linear layer output; if None, no dropout
    :type dropout_linear: Optional[float]
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float,
        grid_max: float,
        num_grids: int,
        inv_denominator: float,
        mode: Literal['RSWAFF','tanh','tanh2','gaussian', 'sigmoid','square','triangle','sample'] = 'RSWAFF',
        dropout_rate: Optional[float] = None,
        dropout_linear: Optional[float] = None,
        train_grid: bool = True,        
        train_inv_denominator: bool = True,
        normalize_rbf: bool = False,
    ) -> None:
        super(FasterKANLayerV2, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize_rbf = normalize_rbf

        self.rbf = RSFAutoV2(
            input_dim=input_dim,
            train_grid=train_grid, 
            train_inv_denominator=train_inv_denominator,
            grid_min=grid_min, 
            grid_max=grid_max, 
            num_grids=num_grids, 
            inv_denominator=inv_denominator, 
            mode=mode
        )
        # Add normalization to stabilize RBF outputs before linear layer
        if normalize_rbf:
            self.rbf_norm = nn.LayerNorm(input_dim * num_grids)
        self.linear = nn.Linear(input_dim * num_grids, output_dim, bias=USE_BIAS_ON_LINEAR) 
        self.drop = nn.Dropout(0 if dropout_rate is None else dropout_rate)
        self.drop_linear = nn.Dropout(0 if dropout_linear is None else dropout_linear)

    def forward(self, x):
        """
        :param x: Input tensor [batch_size, input_dim]
        :type x: torch.Tensor
        :return: Output tensor [batch_size, output_dim]
        :rtype: torch.Tensor
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        spline_basis = self.rbf(x).view(batch_size, -1)
        if self.normalize_rbf:
            spline_basis = self.rbf_norm(spline_basis)  # Normalize before dropout and linear
        spline_basis = self.drop(spline_basis)
        output = self.linear(spline_basis)
        output = self.drop_linear(output)
        return output

    
class DynamicRSFAuto(nn.Module):
    """
    :param input_dim: Dimensionality of input features
    :type input_dim: int
    :param num_grids: Number of grid points to use
    :type num_grids: int
    :param mode: Type of radial basis function to use
    :type mode: Literal['RSWAFF','tanh','tanh2','gaussian', 'sigmoid','square','triangle','sample']
    Attributes:    
        params_linear (FasterKANLayer): KAN layer to compute grid and inv_denominator parameters
    """
    def __init__(
        self,
        input_dim: int,
        num_grids: int,
        mode : Literal['RSWAFF','tanh','tanh2','gaussian', 'sigmoid','square','triangle','sample'] = 'RSWAFF',
        dropout_rate: Optional[float] = None,
    ):
        super(DynamicRSFAuto,self).__init__()
        self.mode = mode
        self.params_linear = FasterKANLayer(
            input_dim               = input_dim,
            output_dim              = num_grids + 1,
            num_grids               = 1,
            grid_min                = 0,
            grid_max                = 0,
            inv_denominator         = 1.0,
            train_grid              = True,
            train_inv_denominator   = True,
            mode                    = mode,
            dropout_rate            = dropout_rate,
        )
        self.rbf = RadialBasisFunction(self.mode)

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"num_grids={self.params_linear.output_dim-1}"

    def forward(self, x):
        """
        :param x: Input tensor [batch_size, input_dim]
        :type x: torch.Tensor
        :return: Transformed tensor [batch_size, input_dim, num_grids]
        :rtype: torch.Tensor
        """
        params = self.params_linear(x).unsqueeze(-2)
        grid, scale = params.split([self.params_linear.output_dim - 1, 1], dim=-1)
        diff = (x[..., None] - grid).mul(scale) 
        return self.rbf(diff)

class FasterKANLayer(nn.Module):
    """
    A single layer in the FasterKAN architecture.
    
    The layer applies the following sequence:
    1. Transform inputs using Radial Spline Functions (RSF)
    2. Apply dropout with rate based on grid count (1-0.75^num_grids)
    3. Apply linear transformation to the outputs

    :param train_grid: Whether to update grid points during training
    :type train_grid: bool
    :param train_inv_denominator: Whether to update inv_denominator during training
    :type train_inv_denominator: bool
    :param input_dim: Dimensionality of input features
    :type input_dim: int
    :param output_dim: Dimensionality of output features
    :type output_dim: int
    :param grid_min: Minimum value for grid points
    :type grid_min: float
    :param grid_max: Maximum value for grid points
    :type grid_max: float
    :param num_grids: Number of grid points to use
    :type num_grids: int
    :param inv_denominator: Initial value for inverse denominator parameter
    :type inv_denominator: float
    :param mode: Type of radial basis function to use
    :type mode: Literal['RSWAFF','tanh','tanh2','gaussian', 'sigmoid','square','triangle','sample']
    :param dropout_rate: Dropout rate to use on RBF output; if None, defaults to 0.5
    :type dropout_rate: Optional[float]
    :param dropout_linear: Dropout rate to use on linear layer output; if None, no dropout
    :type dropout_linear: Optional[float]
    Attributes:
        rbf (RSF): Radial Spline Function module
        drop (nn.Dropout): Dropout layer with adaptive rate for RBF output
        drop_linear (nn.Dropout): Dropout layer for linear output
        linear (nn.Linear): Linear transformation without bias
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float,
        grid_max: float,
        num_grids: int,
        inv_denominator: float,
        mode : Literal['RSWAFF','tanh','tanh2','gaussian', 'sigmoid','square','triangle','sample'] = 'RSWAFF',
        dropout_rate: Optional[float] = None,
        dropout_linear: Optional[float] = None,
        train_grid: bool = True,        
        train_inv_denominator: bool = True,
        normalize_rbf: bool = False,
    ) -> None:
        super(FasterKANLayer,self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize_rbf = normalize_rbf

        self.rbf = RSFAuto(train_grid, train_inv_denominator,grid_min, grid_max, num_grids, inv_denominator, mode=mode)
        # Add normalization to stabilize RBF outputs before linear layer
        if normalize_rbf:
            self.rbf_norm = nn.LayerNorm(input_dim * num_grids)
        self.linear = nn.Linear(input_dim * num_grids, output_dim, bias=USE_BIAS_ON_LINEAR) 
        
        self.drop = nn.Dropout(0 if dropout_rate is None else float(dropout_rate))
        self.drop_linear = nn.Dropout(0 if dropout_linear is None else float(dropout_linear))
        # self.drop = nn.Dropout(0 if dropout_rate is None else dropout_rate)
        # self.drop_linear = nn.Dropout(0 if dropout_linear is None else dropout_linear) 

    def forward(self, x):
        """
        :param x: Input tensor [batch_size, input_dim]
        :type x: torch.Tensor
        :return: Output tensor [batch_size, output_dim]
        :rtype: torch.Tensor
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        spline_basis = self.rbf(x).view(batch_size, -1)
        if self.normalize_rbf:
            spline_basis = self.rbf_norm(spline_basis)  # Normalize before dropout and linear
        spline_basis = self.drop(spline_basis)
        output = self.linear(spline_basis)
        output = self.drop_linear(output)
        return output

class DynamicFasterKANLayer(nn.Module):
    """
    A single layer in the FasterKAN architecture.
    
    The layer applies the following sequence:
    1. Transform inputs using Radial Spline Functions (RSF)
    2. Apply dropout with rate based on grid count (1-0.75^num_grids)
    3. Apply linear transformation to the outputs

    :param input_dim: Dimensionality of input features
    :type input_dim: int
    :param output_dim: Dimensionality of output features
    :type output_dim: int
    :param num_grids: Number of grid points to use
    :type num_grids: int
    :param mode: Type of radial basis function to use
    :type mode: Literal['RSWAFF','tanh','tanh2','gaussian', 'sigmoid','square','triangle','sample']
    :param dropout_rate: Dropout rate to use on RBF output; if None, defaults to 0.5
    :type dropout_rate: Optional[float]
    :param dropout_linear: Dropout rate to use on linear layer output; if None, no dropout
    :type dropout_linear: Optional[float]
    Attributes:
        rbf (RSF): Radial Spline Function module
        drop (nn.Dropout): Dropout layer with adaptive rate
        drop_linear (nn.Dropout): Dropout layer for linear output
        linear (nn.Linear): Linear transformation without bias
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_grids: int,
        mode : Literal['RSWAFF','tanh','tanh2','gaussian', 'sigmoid','square','triangle','sample'] = 'RSWAFF',
        dropout_rate: Optional[float] = None,
        dropout_linear: Optional[float] = None,
        **kwargs
    ) -> None:
        super(DynamicFasterKANLayer,self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.rbf = DynamicRSFAuto(input_dim, num_grids, mode=mode, dropout_rate=dropout_rate)
        self.linear = nn.Linear(input_dim * num_grids, output_dim, bias=USE_BIAS_ON_LINEAR) 
        self.drop = nn.Dropout(0 if dropout_rate is None else dropout_rate)
        self.drop_linear = nn.Dropout(0 if dropout_linear is None else dropout_linear)

    def forward(self, x):
        """
        :param x: Input tensor [batch_size, input_dim]
        :type x: torch.Tensor
        :return: Output tensor [batch_size, output_dim]
        :rtype: torch.Tensor
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        spline_basis = self.rbf(x).view(batch_size, -1)
        spline_basis = self.drop(spline_basis)
        output = self.linear(spline_basis)
        output = self.drop_linear(output)
        return output

# @torch.compile
class FasterKAN(nn.Module):
    """
    FasterKAN: Radial Basis Function-based Kolmogorov-Arnold Network.
    This model stacks multiple FasterKANLayers to create a deep RBF-KAN architecture.
    
    :param hidden_layers: List of layer dimensions including input and output dimensions
        e.g., [784, 100, 10] for MNIST classification with one hidden layer
    :type hidden_layers: List[int]
    :param num_grids: Number of grid points for each layer
        If a single int is provided, it's used for all layers
    :type num_grids: Union[int, List[int]]
    :param grid_min: Minimum value for grid points
    :type grid_min: float
    :param grid_max: Maximum value for grid points
    :type grid_max: float
    :param inv_denominator: Initial value for inverse denominator parameter
    :type inv_denominator: float
    :param mode: Type of radial basis function to use
    :type mode: Literal['RSWAFF','tanh','tanh2','gaussian', 'sigmoid','square','triangle','sample']
    :param residual: List of booleans indicating whether to use residual connections for each layer
    :type residual: list[bool]
    :param dynamic: Whether to use dynamic grid and inv_denominator parameters
    :type dynamic: bool
    :param use_v2: Whether to use V2 variant with per-input-dimension learnable parameters
    :type use_v2: bool
    :param normalize: If `True`, add a normalization layer before every `FasterKAN` layer. Default is `True`
    :type normalize: bool
    :param dropout_rate: If specified, apply `torch.nn.Dropout` to the RBF-activated data with probability `dropout_rate`. Default is `None`.
    :type dropout_rate: float, Optional
    :param dropout_linear: If specified, apply `torch.nn.Dropout` to the linear layer output with probability `dropout_linear`. Default is `None`.
    :type dropout_linear: float, Optional
    
    Attributes:
        train_grid (bool): Whether grid points are being updated during training
        train_inv_denominator (bool): Whether inv_denominator is being updated during training
        layers (nn.ModuleList): List of FasterKANLayer modules
        is_eval (bool): Whether the model is in evaluation mode
        dropout_rate (float): The current dropout rate.
        dropout_linear (float): The current linear dropout rate.
    
    Example:
        ```python
        model = FasterKAN([784, 100, 10], num_grids=10, grid_min=-3.0, grid_max=3.0, inv_denominator=1.0)
        output = model(input_tensor)  # Shape: [batch_size, 10]
        ```
    """
    def __init__(
        self, hidden_layers: List[int], 
        num_grids: Union[int, List[int]],
        grid_min: float,
        grid_max: float,
        inv_denominator: float,
        mode : Callable | Literal['RSWAFF','tanh','tanh2','gaussian', 'sigmoid','square','triangle','sample'] = 'RSWAFF',
        residual : list[bool] = False,
        dynamic : bool = False,
        use_v2 : bool = False,
        normalize : bool = True,
        normalize_rbf : bool = False,
        dropout_rate: Optional[float] = None,
        dropout_linear: Optional[float] = None,
    ):
        super(FasterKAN, self).__init__()

        self.train_grid = True
        self.train_inv_denominator = True
        self.dropout_rate = dropout_rate
        self.dropout_linear = dropout_linear
        self.normalize_rbf = normalize_rbf
        
        num_grids       = expand_value(num_grids,       len(hidden_layers)-1)
        grid_min        = expand_value(grid_min,        len(hidden_layers)-1)
        grid_max        = expand_value(grid_max,        len(hidden_layers)-1)
        inv_denominator = expand_value(inv_denominator, len(hidden_layers)-1)
        residual        = expand_value(residual,        len(hidden_layers)-1)
        
        if dynamic :
            LayerClass = DynamicFasterKANLayer
        elif use_v2 :
            LayerClass = FasterKANLayerV2
        else :
            LayerClass = FasterKANLayer
        
        self.residual   = []
        for _iter, residual_i in enumerate(residual):
            if residual_i :
                if hidden_layers[_iter] == hidden_layers[_iter+1] :
                    self.residual.append(True)
                else :
                    warn(f"Skipped residual connection at layer {_iter}; Number of features do not match ({hidden_layers[_iter]} != {hidden_layers[_iter+1]})")
                    self.residual.append(False)
            else :
                self.residual.append(False)
        
        self.layers = nn.ModuleList([
            LayerClass(
                train_grid            = self.train_grid,
                train_inv_denominator = self.train_inv_denominator,
                input_dim             = in_dim, 
                output_dim            = out_dim, 
                grid_min              = grid_min_i,
                grid_max              = grid_max_i,
                num_grids             = num_grids_i,
                inv_denominator       = inv_denominator_i,
                mode                  = mode,
                dropout_rate          = self.dropout_rate,
                dropout_linear        = self.dropout_linear,
                normalize_rbf         = self.normalize_rbf,
            ) for _iter, (
                num_grids_i, 
                in_dim, 
                out_dim, 
                grid_min_i, 
                grid_max_i, 
                inv_denominator_i, 
            ) in enumerate(zip(
                num_grids, 
                hidden_layers[:-1], 
                hidden_layers[1:],
                grid_min,
                grid_max,
                inv_denominator,
            ))
        ])
        if normalize and len(hidden_layers) >= 2:
            self.normalize = nn.ModuleList([
                nn.LayerNorm(
                    in_dim
                ) for in_dim in hidden_layers[:-1]
            ])
        else :
            self.normalize = False

    def extra_repr(self) -> str:
        """
        Return the extra representation of the module.
        """
        return f"residual={tuple(self.residual)}"

    def eval(self):
        """
        Set the model to evaluation mode, disabling grid and inv_denominator parameter updates.
        """
        self.is_eval = True
        self.train_grid = False
        self.train_inv_denominator = False
        return super().eval()

    def train(self, mode=True):
        """
        Set the model to training mode, enabling updates to grid and inv_denominator parameters.
        
        Args:
            mode (bool): Whether to enable training mode (True) or evaluation mode (False)
        """
        self.is_eval = not mode
        self.train_grid = mode
        self.train_inv_denominator = mode
        return super().train(mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Output tensor [batch_size, output_dim]
        """
        for _iter, (layer, res) in enumerate(zip(self.layers, self.residual)):
            if self.normalize :
                x = self.normalize[_iter](x)
            if res:
                x = layer(x) + x
            else :
                x = layer(x)
        return x

@overload
def fuse_faster_kan(sequential : nn.Sequential) -> FasterKAN:
    ...
    
@overload
def fuse_faster_kan(*args : FasterKAN) -> FasterKAN:
    ...
    
def fuse_faster_kan(*args) -> FasterKAN:
    if isinstance(args[0], nn.Sequential):
        args = args[0].modules()
        
    model = FasterKAN(
        hidden_layers   = [0],
        num_grids       = [],
        grid_min        = [],
        grid_max        = [],
        inv_denominator = [],
        mode            ='RSWAFF',
        residual        = [],
        dynamic         = False,
        use_v2          = False,
        normalize       = False,
        normalize_rbf   = False,
    )
    model.normalize = nn.ModuleList([])
    for _iter, init_model in enumerate(args):
        if not isinstance(init_model, FasterKAN):
            warn(f'Model {_iter} is not FasterKAN (type is {type(init_model)}). Skipping...')
            continue
        
        model.layers.extend(init_model.layers)
        
        model.residual.extend(init_model.residual)
        
        if init_model.normalize:
            model.normalize.extend(init_model.normalize)
        else :
            model.normalize.extend([nn.Identity() for _ in init_model.layers])
    
    if len(model.layers) == 0:
        raise ValueError(f'No FasterKAN models were provided; got {[type(_) for _ in args]}')
    
    if sum(not isinstance(_, nn.Identity) for _ in model.normalize) == 0:
        del model.normalize 
        model.normalize = False
        
    return model