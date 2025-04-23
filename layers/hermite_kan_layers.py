import math
import torch
import torch.nn as nn
import numpy as np
from inspect import signature
from typing import Tuple, Type, Union
class HermiteKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(HermiteKANLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim
        self.degree = degree

        # Initialize Hermite polynomial coefficients
        self.hermite_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.hermite_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.reshape(x, (-1, self.input_dim))
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        hermite = torch.ones(x.shape[0], self.input_dim, self.degree + 1, device=x.device)
        if self.degree > 0:
            hermite[:, :, 1] = 2 * x
        for i in range(2, self.degree + 1):
            hermite[:, :, i] = 2 * x * hermite[:, :, i - 1].clone() - 2 * (i - 1) * hermite[:, :, i - 2].clone()
        y = torch.einsum('bid,iod->bo', hermite, self.hermite_coeffs)
        y = y.view(-1, self.out_dim)
        return y
class HermiteKANConvNDLayer(nn.Module):
    def __init__(self,
                 conv_class: Type[nn.Module],
                 norm_class: Type[nn.Module],
                 input_dim: int,
                 output_dim: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 degree: int,
                 groups: int = 1,
                 padding: Union[int, Tuple[int, ...], str] = 0,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 ndim: int = 2,
                 base_activation: Type[nn.Module] = nn.GELU,
                 dropout: float = 0.0,
                 **norm_kwargs):
        super(HermiteKANConvNDLayer, self).__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')
        if degree < 0:
            raise ValueError('degree must be non-negative')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.degree = degree
        self.groups = groups
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.ndim = ndim
        self.base_activation = base_activation()
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

        self.dropout = None
        if dropout > 0:
            dropout_layer = getattr(nn, f'Dropout{ndim}d', None)
            if dropout_layer:
                self.dropout = dropout_layer(p=dropout)

        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        for conv_layer in self.poly_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def _compute_hermite_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Physicist's Hermite polynomial basis functions H_0(x) to H_degree(x).
        H_0(x) = 1
        H_1(x) = 2x
        H_n(x) = 2x * H_{n-1}(x) - 2(n-1) * H_{n-2}(x) for n >= 2
        """

        x_normalized = torch.tanh(x)

        batch_size = x_normalized.shape[0]
        channels = x_normalized.shape[1]
        spatial_dims = x_normalized.shape[2:]

        hermite_basis = torch.zeros(batch_size, channels, self.degree + 1, *spatial_dims,
                                    device=x.device, dtype=x.dtype)

        # H_0(x) = 1
        hermite_basis[:, :, 0, ...] = torch.ones_like(x_normalized)

        # H_1(x) = 2x
        if self.degree >= 1:
            hermite_basis[:, :, 1, ...] = 2 * x_normalized


        for i in range(2, self.degree + 1):
            term1 = 2 * x_normalized * hermite_basis[:, :, i - 1, ...].clone()
            term2 = 2 * (i - 1) * hermite_basis[:, :, i - 2, ...].clone()
            hermite_basis[:, :, i, ...] = term1 - term2

        poly_features = hermite_basis.view(batch_size, channels * (self.degree + 1), *spatial_dims)
        return poly_features

    def _forward_kan_group(self, x: torch.Tensor, group_index: int) -> torch.Tensor:
        base_output = self.base_conv[group_index](self.base_activation(x))
        poly_features = self._compute_hermite_basis(x)
        poly_output = self.poly_conv[group_index](poly_features)
        combined_output = base_output + poly_output
        output = self.prelus[group_index](self.layer_norm[group_index](combined_output))
        if self.dropout is not None:
            output = self.dropout(output)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split_x = torch.split(x, self.input_dim_group, dim=1)
        output = [self._forward_kan_group(_x, group_ind) for group_ind, _x in enumerate(split_x)]
        y = torch.cat(output, dim=1)
        return y


class HermiteKANConv3DLayer(HermiteKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super().__init__(
            conv_class=nn.Conv3d, norm_class=norm_layer, input_dim=input_dim, output_dim=output_dim,
            kernel_size=kernel_size, degree=degree, groups=groups, padding=padding, stride=stride,
            dilation=dilation, ndim=3, base_activation=base_activation, dropout=dropout, **norm_kwargs
        )

class HermiteKANConv2DLayer(HermiteKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super().__init__(
            conv_class=nn.Conv2d, norm_class=norm_layer, input_dim=input_dim, output_dim=output_dim,
            kernel_size=kernel_size, degree=degree, groups=groups, padding=padding, stride=stride,
            dilation=dilation, ndim=2, base_activation=base_activation, dropout=dropout, **norm_kwargs
        )

class HermiteKANConv1DLayer(HermiteKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super().__init__(
            conv_class=nn.Conv1d, norm_class=norm_layer, input_dim=input_dim, output_dim=output_dim,
            kernel_size=kernel_size, degree=degree, groups=groups, padding=padding, stride=stride,
            dilation=dilation, ndim=1, base_activation=base_activation, dropout=dropout, **norm_kwargs
        )
