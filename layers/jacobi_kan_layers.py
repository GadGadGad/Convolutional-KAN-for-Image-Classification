# based on this implementation: https://github.com/SpaceLearner/JacobiKAN/blob/main/JacobiKANLayer.py

from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv3d, conv2d, conv1d
from inspect import signature
class JacobiKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, a=1.0, b=1.0, base_activation=nn.GELU):
        super(JacobiKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.a = a
        self.b = b
        self.degree = degree

        self.base_activation = base_activation() if base_activation is not None else nn.Identity()
        self.norm = nn.LayerNorm(output_dim, dtype=torch.float32)

        self.base_weights = nn.Parameter(
            torch.zeros(output_dim, input_dim, dtype=torch.float32)
        )

        self.jacobi_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))

        nn.init.normal_(self.jacobi_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        nn.init.xavier_uniform_(self.base_weights)

    def forward(self, x):
        x = torch.reshape(x, (-1, self.inputdim))  # shape = (batch_size, inputdim)

        basis = F.linear(self.base_activation(x), self.base_weights)

        # Since Jacobian polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # Initialize Jacobian polynomial tensors
        jacobi = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device)
        if self.degree > 0:  ## degree = 0: jacobi[:, :, 0] = 1 (already initialized) ; degree = 1: jacobi[:, :, 1] = x ; d
            jacobi[:, :, 1] = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2
        for i in range(2, self.degree + 1):
            theta_k = (2 * i + self.a + self.b) * (2 * i + self.a + self.b - 1) / (2 * i * (i + self.a + self.b))
            theta_k1 = (2 * i + self.a + self.b - 1) * (self.a * self.a - self.b * self.b) / (
                    2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            theta_k2 = (i + self.a - 1) * (i + self.b - 1) * (2 * i + self.a + self.b) / (
                    i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            jacobi[:, :, i] = (theta_k * x + theta_k1) * jacobi[:, :, i - 1].clone() - theta_k2 * jacobi[:, :,
                                                                                                  i - 2].clone()
        # Compute the Jacobian interpolation
        y = torch.einsum('bid,iod->bo', jacobi, self.jacobi_coeffs)  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)

        y = self.base_activation(self.norm(y + basis))
        return y
class JacobiKANConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, conv_w_fun, input_dim, output_dim, degree, kernel_size,
                 base_activation: nn.Module = nn.SiLU, a: float = 1.0, b: float = 1.0,
                 groups=1, padding=0, stride=1, dilation=1, dropout: float = 0.0,
                 ndim: int = 2, **norm_kwargs):
        super(JacobiKANConvNDLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.base_activation = base_activation() if base_activation is not None else nn.Identity()
        self.conv_w_fun = conv_w_fun
        self.ndim = ndim
        self.dropout = None
        self.norm_kwargs = norm_kwargs
        self.a = a
        self.b = b
        if dropout > 0:
            if self.ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if self.ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if self.ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        if self.groups <= 0:
            raise ValueError('groups must be a positive integer')
        if self.input_dim % self.groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if self.output_dim % self.groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.base_conv = nn.ModuleList([conv_class(self.input_dim // self.groups,
                                                   self.output_dim // self.groups,
                                                   self.kernel_size,
                                                   self.stride,
                                                   self.padding,
                                                   self.dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(self.groups)])

        valid_norm_args = signature(norm_class).parameters
        filtered_norm_kwargs = {k: v for k, v in norm_kwargs.items() if k in valid_norm_args}

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **filtered_norm_kwargs) for _ in range(groups)])

        poly_shape = (self.groups, self.output_dim // self.groups, (self.input_dim // self.groups) * (self.degree + 1)) + tuple(
            kernel_size for _ in range(self.ndim))

        self.poly_weights = nn.Parameter(torch.randn(*poly_shape))

        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        nn.init.normal_(self.poly_weights, mean=0.0, std=1 / (self.input_dim * (self.degree + 1) * self.kernel_size ** self.ndim))

    @lru_cache(maxsize=128)
    def compute_jacobi_polynomials(self, x, order):
        # Base case polynomials P0 and P1
        P0 = x.new_ones(x.shape)  # P0 = 1 for all x
        if order == 0:
            return P0.unsqueeze(-1)
        P1 = ((self.a - self.b) + (self.a + self.b + 2) * x) / 2
        jacobi_polys = [P0, P1]

        # Compute higher order polynomials using recurrence
        for i in range(2, order + 1):
            theta_k = (2 * i + self.a + self.b) * (2 * i + self.a + self.b - 1) / (2 * i * (i + self.a + self.b))
            theta_k1 = (2 * i + self.a + self.b - 1) * (self.a * self.a - self.b * self.b) / (
                    2 * i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            theta_k2 = (i + self.a - 1) * (i + self.b - 1) * (2 * i + self.a + self.b) / (
                    i * (i + self.a + self.b) * (2 * i + self.a + self.b - 2))
            pn = (theta_k * x + theta_k1) * jacobi_polys[i - 1].clone() - theta_k2 * jacobi_polys[i - 2].clone()
            jacobi_polys.append(pn)

        return torch.concatenate(jacobi_polys, dim=1)

    def forward_kaj(self, x, group_index):
        # Apply base activation to input and then linear transform with base weights
        base_output = self.base_conv[group_index](x)

        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x_normalized = torch.tanh(x)

        # Compute Legendre polynomials for the normalized x
        jacobi_basis = self.compute_jacobi_polynomials(x_normalized, self.degree)

        if self.dropout is not None:
            jacobi_basis = self.dropout(jacobi_basis)

        # Reshape legendre_basis to match the expected input dimensions for linear transformation
        # Compute polynomial output using polynomial weights
        poly_output = self.conv_w_fun(jacobi_basis, self.poly_weights[group_index],
                                      stride=self.stride, dilation=self.dilation,
                                      padding=self.padding, groups=1)

        # poly_output = poly_output.view(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], self.output_dim // self.groups)
        # Combine base and polynomial outputs, normalize, and activate
        x = base_output + poly_output
        if isinstance(self.layer_norm[group_index], nn.LayerNorm):
            orig_shape = x.shape
            x = self.layer_norm[group_index](x.view(orig_shape[0], -1)).view(orig_shape)
        else:
            x = self.layer_norm[group_index](x)
        x = self.base_activation(x)

        return x

    def forward(self, x):

        split_x = torch.split(x, self.input_dim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kaj(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class JacobiKANConv3DLayer(JacobiKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, base_activation=nn.GELU, a=1.0, b=1.0, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super(JacobiKANConv3DLayer, self).__init__(conv_class=nn.Conv3d, norm_class=norm_layer, conv_w_fun=conv3d,
                                              input_dim=input_dim, output_dim=output_dim,
                                              degree=degree, kernel_size=kernel_size,
                                              base_activation=base_activation, a=a, b=b,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=3, dropout=dropout, **norm_kwargs)


class JacobiKANConv2DLayer(JacobiKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, base_activation=nn.GELU, a=1.0, b=1.0, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(JacobiKANConv2DLayer, self).__init__(conv_class=nn.Conv2d, norm_class=norm_layer, conv_w_fun=conv2d,
                                              input_dim=input_dim, output_dim=output_dim,
                                              degree=degree, kernel_size=kernel_size,
                                              base_activation=base_activation, a=a, b=b,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=2, dropout=dropout, **norm_kwargs)

class JacobiKANConv1DLayer(JacobiKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, base_activation=nn.GELU, a=1.0, b=1.0, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super(JacobiKANConv1DLayer, self).__init__(conv_class=nn.Conv1d, norm_class=norm_layer, conv_w_fun=conv1d,
                                              input_dim=input_dim, output_dim=output_dim,
                                              degree=degree, kernel_size=kernel_size,
                                              base_activation=base_activation, a=a, b=b,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=1, dropout=dropout, **norm_kwargs)