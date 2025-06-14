import math
import torch
import torch.nn as nn
import numpy as np
from inspect import signature
from typing import Tuple, Type, Union
class LaguerreKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, alpha):
        super(LaguerreKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.alpha = alpha  

        self.laguerre_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.laguerre_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = x.view(-1, self.input_dim)  
        x = torch.tanh(x) 

        laguerre = torch.zeros(x.size(0), self.input_dim, self.degree + 1, device=x.device)
        laguerre[:, :, 0] = 1  # L_0^alpha(x) = 1
        if self.degree > 0:
            laguerre[:, :, 1] = 1 + self.alpha - x  # L_1^alpha(x) = 1 + alpha - x

        for k in range(2, self.degree + 1):
            term1 = ((2 * (k-1) + 1 + self.alpha - x) * laguerre[:, :, k - 1].clone())
            term2 = (k - 1 + self.alpha) * laguerre[:, :, k - 2].clone()
            laguerre[:, :, k] = (term1 - term2) / (k)

        #laguerre = torch.tanh(laguerre)

        # Compute the Laguerre interpolation
        y = torch.einsum('bid,iod->bo', laguerre, self.laguerre_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)
        return y
class LaguerreKANConvNDLayer(nn.Module):
    def __init__(self,
                 conv_class: Type[nn.Module],
                 norm_class: Type[nn.Module],
                 input_dim: int,
                 output_dim: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 degree: int,
                 alpha: float, 
                 groups: int = 1,
                 padding: Union[int, Tuple[int, ...], str] = 0,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 ndim: int = 2,
                 base_activation: Type[nn.Module] = nn.GELU,
                 dropout: float = 0.0,
                 **norm_kwargs):
        super(LaguerreKANConvNDLayer, self).__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')
        if degree < 0:
            raise ValueError('degree must be non-negative')
        if alpha <= -1.0:
            raise ValueError('alpha must be greater than -1 for Laguerre polynomials')


        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.degree = degree
        self.alpha = alpha
        self.groups = groups
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.ndim = ndim
        self.base_activation = base_activation() if base_activation is not None else nn.Identity()
        self.norm_kwargs = norm_kwargs

        self.input_dim_group = input_dim // groups
        self.output_dim_group = output_dim // groups
        self.poly_input_dim_group = self.input_dim_group * (degree + 1)

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

    def compute_laguerre_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the generalized Laguerre polynomial basis functions L_0^alpha(x) to L_degree^alpha(x).
        L_0^alpha(x) = 1
        L_1^alpha(x) = 1 + alpha - x
        k * L_k^alpha(x) = (2k - 1 + alpha - x) * L_{k-1}^alpha(x) - (k - 1 + alpha) * L_{k-2}^alpha(x) for k >= 2
        """
        # Apply tanh normalization similar to original implementation for stability
        x_normalized = torch.tanh(x)

        batch_size = x_normalized.shape[0]
        channels = x_normalized.shape[1]
        spatial_dims = x_normalized.shape[2:]

        laguerre_basis = torch.zeros(batch_size, channels, self.degree + 1, *spatial_dims,
                                     device=x.device, dtype=x.dtype)

        # L_0^alpha(x) = 1
        laguerre_basis[:, :, 0, ...] = torch.ones_like(x_normalized)

        # L_1^alpha(x) = 1 + alpha - x
        if self.degree >= 1:
            laguerre_basis[:, :, 1, ...] = (1 + self.alpha) - x_normalized

        # Compute higher degree polynomials using recurrence
        for k in range(2, self.degree + 1):
            # k * L_k = (2*(k-1) + 1 + alpha - x) * L_{k-1} - (k - 1 + alpha) * L_{k-2}
            term1 = ((2 * (k - 1) + 1 + self.alpha - x_normalized) *
                     laguerre_basis[:, :, k - 1, ...].clone())
            term2 = ((k - 1 + self.alpha) *
                     laguerre_basis[:, :, k - 2, ...].clone())
            # Avoid division by zero (k starts at 2)
            laguerre_basis[:, :, k, ...] = (term1 - term2) / k

        poly_features = laguerre_basis.view(batch_size, channels * (self.degree + 1), *spatial_dims)
        return poly_features

    def forward_kan_laguerre(self, x: torch.Tensor, group_index: int) -> torch.Tensor:
        base_output = self.base_conv[group_index](self.base_activation(x))
        poly_features = self.compute_laguerre_basis(x)
        poly_output = self.poly_conv[group_index](poly_features)
        combined_output = base_output + poly_output
        output = self.prelus[group_index](self.layer_norm[group_index](combined_output))
        if self.dropout is not None:
            output = self.dropout(output)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Laguerre KAN convolutional layer."""
        split_x = torch.split(x, self.input_dim_group, dim=1)
        output = [self.forward_kan_laguerre(_x, group_ind) for group_ind, _x in enumerate(split_x)]
        y = torch.cat(output, dim=1)
        return y

class LaguerreKANConv3DLayer(LaguerreKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree, alpha, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super().__init__(
            conv_class=nn.Conv3d, norm_class=norm_layer, input_dim=input_dim, output_dim=output_dim,
            kernel_size=kernel_size, degree=degree, alpha=alpha, groups=groups, padding=padding, stride=stride,
            dilation=dilation, ndim=3, base_activation=base_activation, dropout=dropout, **norm_kwargs
        )

class LaguerreKANConv2DLayer(LaguerreKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree, alpha, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super().__init__(
            conv_class=nn.Conv2d, norm_class=norm_layer, input_dim=input_dim, output_dim=output_dim,
            kernel_size=kernel_size, degree=degree, alpha=alpha, groups=groups, padding=padding, stride=stride,
            dilation=dilation, ndim=2, base_activation=base_activation, dropout=dropout, **norm_kwargs
        )

class LaguerreKANConv1DLayer(LaguerreKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree, alpha, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super().__init__(
            conv_class=nn.Conv1d, norm_class=norm_layer, input_dim=input_dim, output_dim=output_dim,
            kernel_size=kernel_size, degree=degree, alpha=alpha, groups=groups, padding=padding, stride=stride,
            dilation=dilation, ndim=1, base_activation=base_activation, dropout=dropout, **norm_kwargs
        )

