import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import signature
class ChebyKANLayer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 degree: int):
        super(ChebyKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.epsilon = 1e-7
        
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))
    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.input_dim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, input_dim, self.degree + 1)
        # Apply acos
        x = torch.acos(torch.clamp(x, -1 + self.epsilon, 1 - self.epsilon))
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = torch.cos(x)
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.output_dim)
        return y
class ChebyKANConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_layer, input_dim, output_dim, degree, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, dropout=0.0, **norm_kwargs):
        super(ChebyKANConvNDLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.dropout = None
        self.norm_kwargs = norm_kwargs
        self.epsilon = 1e-7
        
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

        valid_norm_args = signature(norm_layer).parameters
        filtered_norm_kwargs = {k: v for k, v in norm_kwargs.items() if k in valid_norm_args}

        self.layer_norm = nn.ModuleList([norm_layer(output_dim // groups, **filtered_norm_kwargs) for _ in range(groups)])

        self.poly_conv = nn.ModuleList([conv_class((self.degree + 1) * self.input_dim // self.groups,
                                                   self.output_dim // self.groups,
                                                   self.kernel_size,
                                                   self.stride,
                                                   self.padding,
                                                   self.dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(self.groups)])
        arange_buffer_size = (1, 1, -1,) + tuple(1 for _ in range(self.ndim))
        self.register_buffer("arange", torch.arange(0, self.degree + 1, 1).view(*arange_buffer_size))
        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.poly_conv:
            nn.init.normal_(conv_layer.weight, mean=0.0, std=1 / (self.input_dim * (self.degree + 1) * self.kernel_size ** self.ndim))
            nn.init.kaiming_normal_(conv_layer.weight, mode='fan_in', nonlinearity='relu')
    def forward_ChebyKAN(self, x, group_index):

        x = torch.tanh(x).unsqueeze(2)
        x = torch.acos(torch.clamp(x, -1 + self.epsilon, 1 - self.epsilon))
        x = (x * self.arange).flatten(1, 2)
        x = torch.cos(x)
        x = self.poly_conv[group_index](x)
        x = self.layer_norm[group_index](x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def forward(self, x):

        split_x = torch.split(x, self.input_dim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_ChebyKAN(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class ChebyKANConv3DLayer(ChebyKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout=0.0, norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super(ChebyKANConv3DLayer, self).__init__(nn.Conv3d, norm_layer,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=3, dropout=dropout, **norm_kwargs)


class ChebyKANConv2DLayer(ChebyKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout=0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(ChebyKANConv2DLayer, self).__init__(nn.Conv2d, norm_layer,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=2, dropout=dropout, **norm_kwargs)


class ChebyKANConv1DLayer(ChebyKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout=0.0, norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super(ChebyKANConv1DLayer, self).__init__(nn.Conv1d, norm_layer,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=1, dropout=dropout, **norm_kwargs)