from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import conv3d, conv2d, conv1d
from inspect import signature
class LegendreKANLayer(nn.Module):
    def __init__(self,
                 input_features: int,
                 output_features: int,
                 degree: int=3,
                 base_activation: nn.Module=nn.SiLU,
                 ):
        super(LegendreKANLayer, self).__init__()
        
        self.polynomial_order = degree
        self.base_activation = base_activation() if base_activation is not None else nn.Identity()
        
        self.base_weight = nn.Parameter(torch.rand(output_features, input_features))
        self.poly_weight = nn.Parameter(torch.rand(output_features, input_features * (degree + 1)))
        
        self.layer_norm = nn.LayerNorm(output_features)
        
        nn.init.kaiming_uniform_(self.base_weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.poly_weight, nonlinearity='linear')

    @lru_cache(maxsize=128)
    def compute_legendre_polynomials(self,
                                     x: torch.Tensor,
                                     order: int=3):
        P0 = x.new_ones(x.shape)
        if order == 0:
            return P0.unsqueeze(-1)
        P1 = x
        legendre_polys = [P0, P1]
        for n in range(1, order):
            Pn = ((2.0 * n + 1.0) * x * legendre_polys[-1] - n * legendre_polys[-2]) / (n + 1.0)
            legendre_polys.append(Pn)

        return torch.stack(legendre_polys, dim=-1)
    def forward(self,
                x: torch.Tensor):
        batch_size = x.shape[0]
        base_output = F.linear(self.base_activation(x), self.base_weight)
        x_normalized = 2 * (x - x.min()) / (x.max() - x.min()) - 1
        legendre_basis = self.compute_legendre_polynomials(x_normalized, self.polynomial_order).view(batch_size, -1)
        poly_output = F.linear(legendre_basis, self.poly_weight)
        x = self.base_activation(self.layer_norm(base_output + poly_output))
        
        return x
class LegendreKANConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, conv_w_fun, input_dim, output_dim, degree, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1, dropout: float = 0.0,
                 ndim: int = 2, **norm_kwargs):
        super(LegendreKANConvNDLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.base_activation = nn.SiLU()
        self.conv_w_fun = conv_w_fun
        self.ndim = ndim
        self.dropout = None
        self.norm_kwargs = norm_kwargs
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
            self.kernel_size for _ in range(self.ndim))

        self.poly_weights = nn.Parameter(torch.randn(*poly_shape))

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        nn.init.kaiming_uniform_(self.poly_weights, nonlinearity='linear')

    @lru_cache(maxsize=128)  # Cache to avoid recomputation of Legendre polynomials
    def compute_legendre_polynomials(self, x, order):
        # Base case polynomials P0 and P1
        P0 = x.new_ones(x.shape)  # P0 = 1 for all x
        if order == 0:
            return P0.unsqueeze(-1)
        P1 = x  # P1 = x
        legendre_polys = [P0, P1]

        # Compute higher order polynomials using recurrence
        for n in range(1, order):
            Pn = ((2.0 * n + 1.0) * x * legendre_polys[-1] - n * legendre_polys[-2]) / (n + 1.0)
            legendre_polys.append(Pn)

        return torch.concatenate(legendre_polys, dim=1)

    def forward_kal(self, x, group_index):
        base_output = self.base_conv[group_index](x)

        # Normalize x to the range [-1, 1] for stable Legendre polynomial computation
        x_normalized = 2 * (x - x.min()) / (x.max() - x.min()) - 1 if x.shape[0] > 0 else x

        if self.dropout is not None:
            x_normalized = self.dropout(x_normalized)

        # Compute Legendre polynomials for the normalized x
        legendre_basis = self.compute_legendre_polynomials(x_normalized, self.degree)

        poly_output = self.conv_w_fun(legendre_basis, self.poly_weights[group_index],
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

        # x = self.base_conv(x)
        split_x = torch.split(x, self.input_dim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kal(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class LegendreKANConv3DLayer(LegendreKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super(LegendreKANConv3DLayer, self).__init__(nn.Conv3d, norm_layer, conv3d,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=3, dropout=dropout, **norm_kwargs)


class LegendreKANConv2DLayer(LegendreKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(LegendreKANConv2DLayer, self).__init__(nn.Conv2d, norm_layer, conv2d,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=2, dropout=dropout, **norm_kwargs)


class LegendreKANConv1DLayer(LegendreKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super(LegendreKANConv1DLayer, self).__init__(nn.Conv1d, norm_layer, conv1d,
                                              input_dim, output_dim,
                                              degree, kernel_size,
                                              groups=groups, padding=padding, stride=stride, dilation=dilation,
                                              ndim=1, dropout=dropout, **norm_kwargs)