import math
import torch
import torch.nn as nn
import numpy as np
from inspect import signature
from typing import Tuple, Type, Union
class TaylorKANLayer(nn.Module):
  def __init__(self, input_dim, out_dim, degree, add_bias=True):
    super(TaylorKANLayer, self).__init__()
    self.input_dim = input_dim
    self.out_dim = out_dim
    self.degree = degree
    self.add_bias = add_bias

    # Initialize Taylor coefficients
    self.coeffs = nn.Parameter(torch.randn(out_dim, input_dim, degree) * 0.01)  
    if self.add_bias:
      self.bias = nn.Parameter(torch.zeros(1, out_dim)) 

  def forward(self, x):
    shape = x.shape
    outshape = shape[0:-1] + (self.out_dim,)
    x = torch.reshape(x, (-1, self.input_dim)) 

    x_expanded = x.unsqueeze(1).expand(-1, self.out_dim, -1)

    # Compute and accumulate each term of the Taylor expansion
    y = torch.zeros((x.shape[0], self.out_dim), device=x.device)  

    for i in range(self.degree):
      term = (x_expanded ** i) * self.coeffs[:, :, i]  
      y += term.sum(dim=-1) 

    if self.add_bias:
      y += self.bias  

    y = torch.reshape(y, outshape)
    return y

class TaylorKANConvNDLayer(nn.Module):
    def __init__(self,
                 conv_class: Type[nn.Module],
                 norm_class: Type[nn.Module],
                 input_dim: int,
                 output_dim: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 degree: int, # Number of Taylor terms (x^0 to x^(degree-1))
                 groups: int = 1,
                 padding: Union[int, Tuple[int, ...], str] = 0,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 ndim: int = 2,
                 base_activation: Type[nn.Module] = nn.GELU,
                 dropout: float = 0.0,
                 **norm_kwargs):
        super(TaylorKANConvNDLayer, self).__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')
        if degree < 1:
            # Need at least the x^0 term (constant)
            raise ValueError('degree must be at least 1')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.degree = degree
        self.groups = groups
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.ndim = ndim
        self.base_activation = base_activation() if base_activation is not None else nn.Identity()
        self.norm_kwargs = norm_kwargs

        self.input_dim_group = input_dim // groups
        self.output_dim_group = output_dim // groups
        self.poly_input_dim_group = self.input_dim_group * degree

        self.base_conv = nn.ModuleList([
            conv_class(
                self.input_dim_group,
                self.output_dim_group,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                groups=1,
                bias=False
            ) for _ in range(groups)
        ])

        self.poly_conv = nn.ModuleList([
            conv_class(
                self.poly_input_dim_group,
                self.output_dim_group,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                groups=1,
                bias=False 
            ) for _ in range(groups)
        ])

        valid_norm_args = signature(norm_class).parameters
        filtered_norm_kwargs = {k: v for k, v in norm_kwargs.items() if k in valid_norm_args}

        self.layer_norm = nn.ModuleList([norm_class(self.output_dim_group, **filtered_norm_kwargs) for _ in range(groups)])


        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(groups)])

        # Dropout layer
        self.dropout = None
        if dropout > 0:
            dropout_layer = getattr(nn, f'Dropout{ndim}d', None)
            if dropout_layer:
                self.dropout = dropout_layer(p=dropout)

        # Initialize weights
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        for conv_layer in self.poly_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def compute_taylor_basis(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        channels = x.shape[1]
        spatial_dims = x.shape[2:]

        # Apply tanh to bound the input features to (-1, 1)
        x_bounded = torch.tanh(x)

        taylor_basis = torch.zeros(batch_size, channels, self.degree, *spatial_dims,
                                   device=x.device, dtype=x.dtype)

        # Calculate powers using the bounded input x_bounded
        # x^0 = 1
        taylor_basis[:, :, 0, ...] = torch.ones_like(x) # Use original x just for shape/device info
        # x^1, x^2, ...
        if self.degree > 1:
           taylor_basis[:, :, 1, ...] = x_bounded
           for i in range(2, self.degree):
               # Use x_bounded for multiplication
               taylor_basis[:, :, i, ...] = taylor_basis[:, :, i - 1, ...].clone() * x_bounded

        poly_features = taylor_basis.contiguous().view(batch_size, channels * self.degree, *spatial_dims)
        return poly_features

    def forward_kan_taylor(self, x: torch.Tensor, group_index: int) -> torch.Tensor:
        # Base path
        base_output = self.base_conv[group_index](self.base_activation(x))
        # Polynomial (Taylor) path
        poly_features = self.compute_taylor_basis(x)
        poly_output = self.poly_conv[group_index](poly_features)

        # Combine paths
        combined_output = base_output + poly_output

        # Apply normalization and activation
        output = self.prelus[group_index](self.layer_norm[group_index](combined_output))
        
        # Apply dropout if configured
        if self.dropout is not None:
            output = self.dropout(output)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split_x = torch.split(x, self.input_dim_group, dim=1)
        output = [self.forward_kan_taylor(_x, group_ind) for group_ind, _x in enumerate(split_x)]
        y = torch.cat(output, dim=1)
        return y


class TaylorKANConv3DLayer(TaylorKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super().__init__(
            conv_class=nn.Conv3d, norm_class=norm_layer, input_dim=input_dim, output_dim=output_dim,
            kernel_size=kernel_size, degree=degree, groups=groups, padding=padding, stride=stride,
            dilation=dilation, ndim=3, base_activation=base_activation, dropout=dropout, **norm_kwargs
        )

class TaylorKANConv2DLayer(TaylorKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super().__init__(
            conv_class=nn.Conv2d, norm_class=norm_layer, input_dim=input_dim, output_dim=output_dim,
            kernel_size=kernel_size, degree=degree, groups=groups, padding=padding, stride=stride,
            dilation=dilation, ndim=2, base_activation=base_activation, dropout=dropout, **norm_kwargs
        )

class TaylorKANConv1DLayer(TaylorKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super().__init__(
            conv_class=nn.Conv1d, norm_class=norm_layer, input_dim=input_dim, output_dim=output_dim,
            kernel_size=kernel_size, degree=degree, groups=groups, padding=padding, stride=stride,
            dilation=dilation, ndim=1, base_activation=base_activation, dropout=dropout, **norm_kwargs
        )

