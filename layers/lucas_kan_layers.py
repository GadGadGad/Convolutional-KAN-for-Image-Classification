import math
import torch
import torch.nn as nn
import numpy as np
from inspect import signature

from typing import Tuple, Type, Union
class LucasKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(LucasKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        # Initialize coefficients for the Lucas polynomials
        self.lucas_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.lucas_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Reshape to (batch_size, input_dim)
        x = torch.tanh(x)  # Normalize input x to [-1, 1] for stability in polynomial calculation

        # Initialize Lucas polynomial tensors
        lucas = torch.zeros(x.size(0), self.input_dim, self.degree + 1, device=x.device)
        lucas[:, :, 0] = 2  # L_0(x) = 2
        if self.degree > 0:
            lucas[:, :, 1] = x  # L_1(x) = x

        for i in range(2, self.degree + 1):
            # Compute Lucas polynomials using the recurrence relation
            lucas[:, :, i] = x * lucas[:, :, i - 1].clone() + lucas[:, :, i - 2].clone()

        # Normalize the polynomial outputs to prevent runaway values
        #lucas = torch.tanh(lucas)

        # Compute the Lucas interpolation
        y = torch.einsum('bid,iod->bo', lucas, self.lucas_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)
        return y
class LucasKANConvNDLayer(nn.Module):
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
                 base_activation: nn.Module = nn.GELU,
                 dropout: float = 0.0,
                 **norm_kwargs):
        """
        Initializes the LucasKANConvNDLayer.

        Args:
            conv_class: The PyTorch convolution class (e.g., nn.Conv2d).
            norm_class: The PyTorch normalization class (e.g., nn.InstanceNorm2d).
            input_dim: Number of input channels.
            output_dim: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            degree: The maximum degree of the Lucas polynomials.
            groups: Number of groups for grouped convolution. Input and output dims must be divisible by groups.
            padding: Padding added to the input.
            stride: Stride of the convolution.
            dilation: Dilation of the convolution.
            ndim: Number of spatial dimensions (1, 2, or 3).
            base_activation: Activation function applied to the input before the base convolution.
            dropout: Dropout probability. If > 0, applies appropriate DropoutNd.
            norm_kwargs: Additional keyword arguments for the normalization layer.
        """
        super(LucasKANConvNDLayer, self).__init__()
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

        self.dropout = None
        if dropout > 0:
            dropout_layer = getattr(nn, f'Dropout{ndim}d', None)
            if dropout_layer:
                self.dropout = dropout_layer(p=dropout)

        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        for conv_layer in self.poly_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def compute_lucas_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Lucas polynomial basis functions L_0(x) to L_degree(x).
        L_0(x) = 2
        L_1(x) = x
        L_n(x) = x * L_{n-1}(x) + L_{n-2}(x) for n >= 2
        """
        x_normalized = torch.tanh(x)

        batch_size = x_normalized.shape[0]
        channels = x_normalized.shape[1]
        spatial_dims = x_normalized.shape[2:]


        lucas_basis = torch.zeros(batch_size, channels, self.degree + 1, *spatial_dims,
                                  device=x.device, dtype=x.dtype)

        # L_0(x) = 2
        lucas_basis[:, :, 0, ...] = 2 * torch.ones_like(x_normalized)

        # L_1(x) = x
        if self.degree >= 1:
            lucas_basis[:, :, 1, ...] = x_normalized
        for i in range(2, self.degree + 1):
            lucas_basis[:, :, i, ...] = (x_normalized * lucas_basis[:, :, i - 1, ...].clone() +
                                         lucas_basis[:, :, i - 2, ...].clone())

        poly_features = lucas_basis.view(batch_size, channels * (self.degree + 1), *spatial_dims)
        return poly_features

    def forward_kan_lucas(self, x: torch.Tensor, group_index: int) -> torch.Tensor:
        """Applies the KAN logic for a single group."""
        base_output = self.base_conv[group_index](self.base_activation(x))

        poly_features = self.compute_lucas_basis(x)
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
        output = [self.forward_kan_lucas(_x, group_ind) for group_ind, _x in enumerate(split_x)]
        y = torch.cat(output, dim=1)
        return y


class LucasKANConv3DLayer(LucasKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super().__init__(
            conv_class=nn.Conv3d, norm_class=norm_layer, input_dim=input_dim, output_dim=output_dim,
            kernel_size=kernel_size, degree=degree, groups=groups, padding=padding, stride=stride,
            dilation=dilation, ndim=3, base_activation=base_activation, dropout=dropout, **norm_kwargs
        )

class LucasKANConv2DLayer(LucasKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super().__init__(
            conv_class=nn.Conv2d, norm_class=norm_layer, input_dim=input_dim, output_dim=output_dim,
            kernel_size=kernel_size, degree=degree, groups=groups, padding=padding, stride=stride,
            dilation=dilation, ndim=2, base_activation=base_activation, dropout=dropout, **norm_kwargs
        )

class LucasKANConv1DLayer(LucasKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super().__init__(
            conv_class=nn.Conv1d, norm_class=norm_layer, input_dim=input_dim, output_dim=output_dim,
            kernel_size=kernel_size, degree=degree, groups=groups, padding=padding, stride=stride,
            dilation=dilation, ndim=1, base_activation=base_activation, dropout=dropout, **norm_kwargs
        )

# Remove old LucasKANLayer class if no longer needed