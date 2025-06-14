import math
import torch
import torch.nn as nn
import numpy as np
from inspect import signature
from typing import Tuple, Type, Union
class GegenbauerKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, alpha_param):
        super(GegenbauerKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.alpha_param = alpha_param

        # Initialize Gegenbauer polynomial coefficients
        self.gegenbauer_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.gegenbauer_coeffs, mean=0.0, std=1/(input_dim * (degree + 1)))

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Reshape to (batch_size, input_dim)
        x = torch.tanh(x)  # Normalize x to [-1, 1]
        
        gegenbauer = torch.ones(x.shape[0], self.input_dim, self.degree + 1, device=x.device)
        if self.degree > 0:
            gegenbauer[:, :, 1] = 2 * self.alpha_param * x  # C_1^alpha(x) = 2*alpha*x

        for n in range(1, self.degree):
            term1 = 2 * (n + self.alpha_param) * x * gegenbauer[:, :, n].clone()
            term2 = (n + 2 * self.alpha_param - 1) * gegenbauer[:, :, n - 1].clone()
            gegenbauer[:, :, n + 1] = (term1 - term2) / (n + 1)  # Apply the recurrence relation

        y = torch.einsum('bid,iod->bo', gegenbauer, self.gegenbauer_coeffs)
        return y.view(-1, self.output_dim)
class GegenbauerKANConvNDLayer(nn.Module):
    def __init__(self,
                 conv_class: Type[nn.Module], 
                 norm_class: Type[nn.Module], 
                 input_dim: int,
                 output_dim: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 degree: int,
                 alpha_param: float, 
                 groups: int = 1,
                 padding: Union[int, Tuple[int, ...], str] = 0,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 ndim: int = 2,
                 base_activation: Type[nn.Module] = nn.GELU,
                 dropout: float = 0.0,
                 **norm_kwargs):
        super(GegenbauerKANConvNDLayer, self).__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')
        if degree < 0:
            raise ValueError('degree must be non-negative')
        if alpha_param <= -0.5:
            raise ValueError('alpha_param must be greater than -0.5')


        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.degree = degree
        self.alpha_param = alpha_param
        self.groups = groups
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.ndim = ndim
        self.base_activation = base_activation() if base_activation is not None else nn.Identity()
        self.norm_kwargs = norm_kwargs

        # Calculate dimensions per group
        self.input_dim_group = input_dim // groups
        self.output_dim_group = output_dim // groups
        # Input channels for the polynomial convolution path
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

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **filtered_norm_kwargs) for _ in range(groups)])


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

    def compute_gegenbauer_basis(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = torch.tanh(x)

        batch_size = x_normalized.shape[0]
        channels = x_normalized.shape[1]
        spatial_dims = x_normalized.shape[2:]

        gegen_basis = torch.zeros(batch_size, channels, self.degree + 1, *spatial_dims,
                                  device=x.device, dtype=x.dtype)

        gegen_basis[:, :, 0, ...] = torch.ones_like(x_normalized)

        if self.degree >= 1:
            gegen_basis[:, :, 1, ...] = 2 * self.alpha_param * x_normalized

        for n in range(1, self.degree):
            term1 = 2 * (n + self.alpha_param) * x_normalized * gegen_basis[:, :, n, ...].clone()
            term2 = (n + 2 * self.alpha_param - 1) * gegen_basis[:, :, n - 1, ...].clone()
            
            gegen_basis[:, :, n + 1, ...] = (term1 - term2) / (n + 1)


        poly_features = gegen_basis.view(batch_size, channels * (self.degree + 1), *spatial_dims)
        return poly_features

    def forward_kan_group(self, x: torch.Tensor, group_index: int) -> torch.Tensor:
        base_output = self.base_conv[group_index](self.base_activation(x))

        poly_features = self.compute_gegenbauer_basis(x)
        poly_output = self.poly_conv[group_index](poly_features)

        combined_output = base_output + poly_output

        output = self.prelus[group_index](self.layer_norm[group_index](combined_output))

        if self.dropout is not None:
            output = self.dropout(output)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        split_x = torch.split(x, self.input_dim_group, dim=1)
        output = []

        # Process each group
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kan_group(_x, group_ind)
            output.append(y) 
            
        y = torch.cat(output, dim=1)
        return y

class GegenbauerKANConv3DLayer(GegenbauerKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree, alpha_param, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super(GegenbauerKANConv3DLayer, self).__init__(
            conv_class=nn.Conv3d,
            norm_class=norm_layer,
            input_dim=input_dim,
            output_dim=output_dim,
            kernel_size=kernel_size,
            degree=degree,
            alpha_param=alpha_param,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ndim=3,
            base_activation=base_activation,
            dropout=dropout,
            **norm_kwargs
        )

class GegenbauerKANConv2DLayer(GegenbauerKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree, alpha_param, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(GegenbauerKANConv2DLayer, self).__init__(
            conv_class=nn.Conv2d,
            norm_class=norm_layer,
            input_dim=input_dim,
            output_dim=output_dim,
            kernel_size=kernel_size,
            degree=degree,
            alpha_param=alpha_param,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ndim=2,
            base_activation=base_activation,
            dropout=dropout,
            **norm_kwargs
        )

class GegenbauerKANConv1DLayer(GegenbauerKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree, alpha_param, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super(GegenbauerKANConv1DLayer, self).__init__(
            conv_class=nn.Conv1d,
            norm_class=norm_layer,
            input_dim=input_dim,
            output_dim=output_dim,
            kernel_size=kernel_size,
            degree=degree,
            alpha_param=alpha_param,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            ndim=1,
            base_activation=base_activation,
            dropout=dropout,
            **norm_kwargs
        )
