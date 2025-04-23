# Based on this: https://github.com/Khochawongwat/GRAMKAN/blob/main/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv3d, conv2d, conv1d
from inspect import signature
from typing import *
class ReLUKANLayer(nn.Module):

    def __init__(self,
                 input_size: int,
                 g: int,
                 k: int,
                 output_size: int,
                 train_ab: bool = True):
        super().__init__()
        self.g, self.k, self.r = g, k, 4 * g * g / ((k + 1) * (k + 1))
        self.input_size, self.output_size = input_size, output_size
        # modification here
        phase_low = torch.arange(-k, g) / g
        phase_high = phase_low + (k + 1) / g
        # modification here
        self.phase_low = nn.Parameter(phase_low[None, :].expand(input_size, -1),
                                      requires_grad=train_ab)
        # modification here, and: `phase_height` to `phase_high`
        self.phase_high = nn.Parameter(phase_high[None, :].expand(input_size, -1),
                                       requires_grad=train_ab)
        self.equal_size_conv = nn.Conv2d(1, output_size, (g + k, input_size))

    def forward(self, x):
        x = x[..., None]
        x1 = torch.relu(x - self.phase_low)
        x2 = torch.relu(self.phase_high - x)
        x = x1 * x2 * self.r
        x = x * x
        x = x.reshape((len(x), 1, self.g + self.k, self.input_size))
        x = self.equal_size_conv(x)
        x = x.reshape((len(x), self.output_size))
        return x
class ReLUConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, conv_w_fun, input_dim, output_dim, kernel_size, g: int = 5, k: int = 3, base_activation: Type[nn.Module] = nn.SiLU,
                 groups=1, padding=0, stride=1, dilation=1, dropout: float = 0.0, ndim: int = 2., train_ab: bool = True,
                 **norm_kwargs):
        super(ReLUConvNDLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.g = g
        self.k = k
        self.r = 4 * g * g / ((k + 1) * (k + 1))
        self.train_ab = train_ab
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
        self.p_dropout = dropout
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

        self.relukan_conv = nn.ModuleList([conv_class((self.g + self.k) * self.input_dim // self.groups,
                                                      self.output_dim // self.groups,
                                                      self.kernel_size,
                                                      self.stride,
                                                      self.padding,
                                                      self.dilation,
                                                      groups=1,
                                                      bias=False) for _ in range(self.groups)])

        phase_low = torch.arange(-self.k, self.g) / self.g
        phase_high = phase_low + (self.k + 1) / self.g

        phase_dims = (1, self.input_dim // self.groups, self.k + self.g) + (1, ) * self.ndim

        self.phase_low = nn.Parameter((phase_low[None, :].expand(self.input_dim // self.groups, -1)).view(*phase_dims),
                                      requires_grad=self.train_ab)

        self.phase_high = nn.Parameter((phase_high[None, :].expand(self.input_dim // self.groups, -1)).view(*phase_dims),
                                       requires_grad=self.train_ab)

        valid_norm_args = signature(norm_class).parameters
        filtered_norm_kwargs = {k: v for k, v in norm_kwargs.items() if k in valid_norm_args}

        self.layer_norm = nn.ModuleList([norm_class(output_dim // groups, **filtered_norm_kwargs) for _ in range(groups)])

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        for conv_layer in self.relukan_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def forward_relukan(self, x, group_index):

        if self.dropout:
            x = self.dropout(x)
        # Apply base activation to input and then linear transform with base weights
        basis = self.base_conv[group_index](self.base_activation(x))

        x = x.unsqueeze(dim=2)
        x1 = torch.relu(x - self.phase_low)
        x2 = torch.relu(self.phase_high - x)
        x = x1 * x2 * self.r
        x = x * x
        x = torch.flatten(x, 1, 2)

        y = self.relukan_conv[group_index](x)

        y = self.base_activation(self.layer_norm[group_index](y + basis))

        return y

    def forward(self, x):

        split_x = torch.split(x, self.input_dim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_relukan(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class ReLUKANConv3DLayer(ReLUConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, base_activation=nn.SiLU, g=5, k=3, train_ab=True, groups=1, padding=0, stride=1,
                 dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super(ReLUKANConv3DLayer, self).__init__(nn.Conv3d, norm_layer, conv3d,
                                                 input_dim, output_dim,
                                                 kernel_size, g=g, k=k, train_ab=train_ab,
                                                 base_activation=base_activation,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 ndim=3, dropout=dropout, **norm_kwargs)


class ReLUKANConv2DLayer(ReLUConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, base_activation=nn.SiLU, g=5, k=3, train_ab=True, groups=1, padding=0, stride=1,
                 dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(ReLUKANConv2DLayer, self).__init__(nn.Conv2d, norm_layer, conv2d,
                                                 input_dim, output_dim,
                                                 kernel_size, g=g, k=k, train_ab=train_ab,
                                                 base_activation=base_activation,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 ndim=2, dropout=dropout, **norm_kwargs)


class ReLUKANConv1DLayer(ReLUConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, base_activation=nn.SiLU, g=5, k=3, train_ab=True, groups=1, padding=0, stride=1,
                 dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super(ReLUKANConv1DLayer, self).__init__(nn.Conv1d, norm_layer, conv1d,
                                                 input_dim, output_dim,
                                                 kernel_size, g=g, k=k, train_ab=train_ab,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 ndim=1, dropout=dropout, **norm_kwargs)