import math
import torch
import torch.nn as nn
import numpy as np
from inspect import signature

from typing import Tuple, Type, Union
class FibonacciKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(FibonacciKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree

        # Initialize coefficients for the Fibonacci polynomials
        self.fib_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.fib_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Reshape to (batch_size, input_dim)
        x = torch.tanh(x)  # Normalize input x to [-1, 1] for stability in polynomial calculation

        # Initialize Fibonacci polynomial tensors
        fib = torch.zeros(x.size(0), self.input_dim, self.degree + 1, device=x.device)
        fib[:, :, 0] = 0  # F_0(x) = 0
        if self.degree > 0:
            fib[:, :, 1] = 1  # F_1(x) = 1

        for i in range(2, self.degree + 1):
            # Compute Fibonacci polynomials using the recurrence relation
            fib[:, :, i] = x * fib[:, :, i - 1].clone() + fib[:, :, i - 2].clone()

        # Normalize the polynomial outputs to prevent runaway values
        #fib = torch.tanh(fib)

        # Compute the Fibonacci interpolation
        y = torch.einsum('bid,iod->bo', fib, self.fib_coeffs)  # shape = (batch_size, output_dim)
        y = y.view(-1, self.output_dim)
        return y
        
class FibonacciKANConvNDLayer(nn.Module):
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
        super(FibonacciKANConvNDLayer, self).__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')
        if degree < 1:
            # Need at least F_0 and F_1
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
                groups=1, # Each group handles its own convolution
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
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            elif ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            elif ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        # Initialize weights
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        for conv_layer in self.poly_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def compute_fibonacci_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Fibonacci polynomial basis functions F_0(x) to F_degree(x).
        F_0(x) = 0
        F_1(x) = 1
        F_n(x) = x * F_{n-1}(x) + F_{n-2}(x) for n >= 2
        """
        # Apply tanh normalization similar to original implementation for stability
        x_normalized = torch.tanh(x)

        batch_size = x_normalized.shape[0]
        channels = x_normalized.shape[1]
        spatial_dims = x_normalized.shape[2:]

        # Shape: (batch, channels, degree+1, spatial_dims...)
        fib_basis = torch.zeros(batch_size, channels, self.degree + 1, *spatial_dims,
                                device=x.device, dtype=x.dtype)

        # F_0(x) = 0 (already initialized)
        # fib_basis[:, :, 0, ...] = 0

        # F_1(x) = 1
        if self.degree >= 1:
            fib_basis[:, :, 1, ...] = torch.ones_like(x_normalized)

        # Compute higher degree polynomials using recurrence
        for i in range(2, self.degree + 1):
            fib_basis[:, :, i, ...] = (x_normalized * fib_basis[:, :, i - 1, ...].clone() +
                                        fib_basis[:, :, i - 2, ...].clone())

        # Optional: Apply tanh normalization to polynomial outputs (like original)
        # fib_basis = torch.tanh(fib_basis)

        # Reshape for convolution: (batch, channels * (degree+1), spatial_dims...)
        poly_features = fib_basis.view(batch_size, channels * (self.degree + 1), *spatial_dims)
        return poly_features

    def forward_kan_group(self, x: torch.Tensor, group_index: int) -> torch.Tensor:
        """Applies the KAN logic for a single group."""
        # Base path
        base_output = self.base_conv[group_index](self.base_activation(x))

        # Polynomial path
        poly_features = self.compute_fibonacci_basis(x)
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
        """Forward pass through the Fibonacci KAN convolutional layer."""
        split_x = torch.split(x, self.input_dim_group, dim=1)
        output = []

        # Process each group
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kan_group(_x, group_ind)
            output.append(y) 

        # Concatenate results along the channel dimension
        y = torch.cat(output, dim=1)
        return y

class FibonacciKANConv3DLayer(FibonacciKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super(FibonacciKANConv3DLayer, self).__init__(
            conv_class=nn.Conv3d,
            norm_class=norm_layer,
            input_dim=input_dim,
            output_dim=output_dim,
            kernel_size=kernel_size,
            degree=degree,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ndim=3,
            base_activation=base_activation,
            dropout=dropout,
            **norm_kwargs
        )

class FibonacciKANConv2DLayer(FibonacciKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(FibonacciKANConv2DLayer, self).__init__(
            conv_class=nn.Conv2d,
            norm_class=norm_layer,
            input_dim=input_dim,
            output_dim=output_dim,
            kernel_size=kernel_size,
            degree=degree,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ndim=2,
            base_activation=base_activation,
            dropout=dropout,
            **norm_kwargs
        )

class FibonacciKANConv1DLayer(FibonacciKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super(FibonacciKANConv1DLayer, self).__init__(
            conv_class=nn.Conv1d,
            norm_class=norm_layer,
            input_dim=input_dim,
            output_dim=output_dim,
            kernel_size=kernel_size,
            degree=degree,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ndim=1,
            base_activation=base_activation,
            dropout=dropout,
            **norm_kwargs
        )
