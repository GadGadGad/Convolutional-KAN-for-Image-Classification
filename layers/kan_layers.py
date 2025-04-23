from inspect import signature
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.regularization import L1, L2 

from typing import *
class KANLayer(nn.Module):
    def __init__(self, input_features, output_features, grid_size=5, spline_order=3, base_activation=nn.GELU,
                 grid_range=[-1, 1]):
        super(KANLayer, self).__init__()

        self.input_features = input_features
        self.output_features = output_features

        # The number of points in the grid for the spline interpolation.
        self.grid_size = grid_size
        # The order of the spline used in the interpolation.
        self.spline_order = spline_order
        # Activation function used for the initial transformation of the input.
        self.base_activation = base_activation() if base_activation is not None else nn.Identity()
        # The range of values over which the grid for spline interpolation is defined.
        self.grid_range = grid_range

        # Initialize the base weights with random values for the linear transformation.
        self.base_weight = nn.Parameter(torch.randn(output_features, input_features))
        # Initialize the spline weights with random values for the spline transformation.
        self.spline_weight = nn.Parameter(torch.randn(output_features, input_features, grid_size + spline_order))
        # Add a layer normalization for stabilizing the output of this layer.
        self.layer_norm = nn.LayerNorm(output_features)
        # Add a PReLU activation for this layer to provide a learnable non-linearity.
        self.prelu = nn.PReLU()

        # Compute the grid values based on the specified range and grid size.
        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32
        ).expand(input_features, -1).contiguous()


        # Initialize the weights using Kaiming uniform distribution for better initial values.
        nn.init.kaiming_uniform_(self.base_weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.spline_weight, nonlinearity='linear')

    def forward(self, x):
        # Process each layer using the defined base weights, spline weights, norms, and activations.
        grid = self.grid.to(x.device)
        # Move the input tensor to the device where the weights are located.

        # Perform the base linear transformation followed by the activation function.
        activated_x = self.base_activation(x)
        base_output = F.linear(activated_x, self.base_weight)

        x_uns = x.unsqueeze(-1)  # Expand dimensions for spline operations.

        # Compute the basis for the spline using intervals and input values.
        # Ensure grid dimensions match expectations for broadcasting
        grid_expanded = grid.unsqueeze(0).expand(x_uns.shape[0], -1, -1) # Expand grid for batch size

        bases = ((x_uns >= grid_expanded[:, :, :-1]) & (x_uns < grid_expanded[:, :, 1:])).to(x.dtype)

        # Compute the spline basis over multiple orders.
        for k in range(1, self.spline_order + 1):
            left_intervals = grid_expanded[:, :, :-(k + 1)]
            right_intervals = grid_expanded[:, :, k:-1]
            # Ensure grid[:, k + 1:] and grid[:, 1:(-k)] have compatible shapes for broadcasting
            grid_right_shift = grid_expanded[:, :, k + 1:]
            grid_left_shift = grid_expanded[:, :, 1:(-k)]

            delta_right = right_intervals - left_intervals
            delta_denom = grid_right_shift - grid_left_shift

            # Handle potential division by zero if grid points are identical
            delta_right = torch.where(delta_right == 0, torch.ones_like(delta_right), delta_right)
            delta_denom = torch.where(delta_denom == 0, torch.ones_like(delta_denom), delta_denom)


            term1 = (x_uns - left_intervals) / delta_right * bases[:, :, :-1]
            term2 = (grid_right_shift - x_uns) / delta_denom * bases[:, :, 1:]

            # Ensure term1 and term2 shapes match before addition
            if term1.shape[-1] != term2.shape[-1]:
                 # This might happen if slicing logic needs adjustment based on k
                 min_len = min(term1.shape[-1], term2.shape[-1])
                 bases = term1[..., :min_len] + term2[..., :min_len]
            else:
                 bases = term1 + term2

        bases = bases.contiguous()
        batch_size = x.size(0)
        num_bases = bases.shape[-1] 

        expected_num_bases = self.grid_size + self.spline_order
        if num_bases != expected_num_bases:
            if self.spline_weight.shape[-1] == expected_num_bases:
                if num_bases < expected_num_bases:
                    pad_size = expected_num_bases - num_bases
                    bases = F.pad(bases, (0, pad_size))
                else:
                    bases = bases[..., :expected_num_bases]


        bases_reshaped = bases.view(batch_size, self.input_features * num_bases)
        spline_weight_reshaped = self.spline_weight.view(self.output_features, self.input_features * num_bases)
        spline_output = F.linear(bases_reshaped, spline_weight_reshaped)

        combined_output = base_output + spline_output
        norm_output = self.layer_norm(combined_output)
        x = self.prelu(norm_output)

        return x

class KANConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, spline_order, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0,
                 **norm_kwargs):
        super(KANConvNDLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation() if base_activation is not None else nn.Identity()
        self.grid_range = grid_range
        self.norm_kwargs = norm_kwargs

        self.dropout = None
        if dropout > 0:
            dropout_layer_class = None
            if ndim == 1:
                dropout_layer_class = nn.Dropout1d
            elif ndim == 2:
                dropout_layer_class = nn.Dropout2d
            elif ndim == 3:
                dropout_layer_class = nn.Dropout3d

            self.dropout = dropout_layer_class(p=dropout)

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        # Calculate dimensions per group
        self.input_dim_group = input_dim // groups
        self.output_dim_group = output_dim // groups

        self.base_conv = nn.ModuleList([conv_class(self.input_dim_group,
                                                   self.output_dim_group,
                                                   self.kernel_size,
                                                   self.stride,
                                                   self.padding,
                                                   self.dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(groups)])

        # Spline conv input dimension = num_bases * input_dim_group
        spline_input_dim_group = (grid_size + spline_order) * self.input_dim_group
        self.spline_conv = nn.ModuleList([conv_class(spline_input_dim_group,
                                                     self.output_dim_group,
                                                     kernel_size,
                                                     stride,
                                                     padding,
                                                     dilation,
                                                     groups=1, 
                                                     bias=False) for _ in range(groups)])
        valid_norm_args = signature(norm_class).parameters
        filtered_norm_kwargs = {k: v for k, v in norm_kwargs.items() if k in valid_norm_args}

        self.layer_norm = nn.ModuleList([norm_class(self.output_dim_group, **filtered_norm_kwargs) for _ in range(groups)])
        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(groups)])

        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32
        )

        for i, conv_layer in enumerate(self.base_conv):
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')
        for i, conv_layer in enumerate(self.spline_conv):
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def forward_kan(self, x, group_index):
        # Apply base activation to input and then linear transform with base weights
        activated_x = self.base_activation(x)
        base_output = self.base_conv[group_index](activated_x)


        x_uns = x.unsqueeze(-1)  # Expand dimensions for spline operations. B, C, [D, H, W], 1
        grid_view_shape = [1] * (self.ndim + 1) + [-1] # e.g., [1, 1, 1, 1, -1] for 3D
        target_shape = list(x.shape[1:]) + [self.grid.shape[0]] # e.g., [C, D, H, W, num_grid_points+1]
        grid = self.grid.view(*grid_view_shape).expand(target_shape).contiguous().to(x.device)

        # Initial bases: B, C, [D, H, W], num_bases_init
        bases = ((x_uns >= grid[..., :-1]) & (x_uns < grid[..., 1:])).to(x.dtype)

        # Compute the spline basis over multiple orders.
        for k in range(1, self.spline_order + 1):
            left_intervals = grid[..., :-(k + 1)]
            right_intervals = grid[..., k:-1]
            grid_right_shift = grid[..., k + 1:]
            grid_left_shift = grid[..., 1:(-k)]

            delta_right = right_intervals - left_intervals
            delta_denom = grid_right_shift - grid_left_shift

            # Handle potential division by zero
            delta_right = torch.where(delta_right == 0, torch.ones_like(delta_right), delta_right)
            delta_denom = torch.where(delta_denom == 0, torch.ones_like(delta_denom), delta_denom)

            term1 = (x_uns - left_intervals) / delta_right * bases[..., :-1]
            term2 = (grid_right_shift - x_uns) / delta_denom * bases[..., 1:]

             # Check shapes before adding
            if term1.shape[-1] != term2.shape[-1]:
                min_len = min(term1.shape[-1], term2.shape[-1])
                bases = term1[..., :min_len] + term2[..., :min_len]
            else:
                bases = term1 + term2


        bases = bases.contiguous()
        bases = bases.moveaxis(-1, 2).flatten(1, 2) # B, C*num_bases, [D,H,W]
        expected_spline_conv_input_dim = (self.grid_size + self.spline_order) * self.input_dim_group
        spline_output = self.spline_conv[group_index](bases)

        combined_output = base_output + spline_output
        norm_output = self.layer_norm[group_index](combined_output)
        x = self.prelus[group_index](norm_output)

        if self.dropout is not None:
            x = self.dropout(x)
        return x

    def forward(self, x):
        split_x = torch.split(x, self.input_dim_group, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_kan(_x, group_ind)
            output.append(y) 

        y = torch.cat(output, dim=1)

        return y


class KANConv3DLayer(KANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0, norm_layer=nn.InstanceNorm3d,
                 **norm_kwargs):
        super(KANConv3DLayer, self).__init__(nn.Conv3d, norm_layer,
                                             input_dim, output_dim,
                                             spline_order, kernel_size,
                                             groups=groups, padding=padding, stride=stride, dilation=dilation,
                                             ndim=3,
                                             grid_size=grid_size, base_activation=base_activation,
                                             grid_range=grid_range, dropout=dropout, **norm_kwargs)


class KANConv2DLayer(KANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0, norm_layer=nn.InstanceNorm2d,
                 **norm_kwargs):
        super(KANConv2DLayer, self).__init__(nn.Conv2d, norm_layer,
                                             input_dim, output_dim,
                                             spline_order, kernel_size,
                                             groups=groups, padding=padding, stride=stride, dilation=dilation,
                                             ndim=2,
                                             grid_size=grid_size, base_activation=base_activation,
                                             grid_range=grid_range, dropout=dropout, **norm_kwargs)


class KANConv1DLayer(KANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, spline_order=3, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=5, base_activation=nn.GELU, grid_range=[-1, 1], dropout=0.0, norm_layer=nn.InstanceNorm1d,
                 **norm_kwargs):
        super(KANConv1DLayer, self).__init__(nn.Conv1d, norm_layer,
                                             input_dim, output_dim,
                                             spline_order, kernel_size,
                                             groups=groups, padding=padding, stride=stride, dilation=dilation,
                                             ndim=1,
                                             grid_size=grid_size, base_activation=base_activation,
                                             grid_range=grid_range, dropout=dropout, **norm_kwargs)

class KAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float = 0.0, grid_size=5, spline_order=3, base_activation=nn.GELU,
                    grid_range: List = [-1, 1], l1_decay: float = 0.0, first_dropout: bool = True, **kwargs):
        super(KAN, self).__init__()  

        self.layers_hidden = layers_hidden
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation() if base_activation is not None else nn.Identity()
        self.grid_range = grid_range

        # list of layers
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))

        self.num_layers = len(layers_hidden) - 1 
        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = KANLayer(in_features, out_features, grid_size=grid_size, spline_order=spline_order,
                                base_activation=self.base_activation, grid_range=grid_range)

            is_last_kan_layer = (i == self.num_layers - 1)
            if l1_decay > 0 and not is_last_kan_layer:
                 layer = L1(layer, l1_decay)

            self.layers.append(layer)

            if dropout > 0 and not is_last_kan_layer:
                self.layers.append(nn.Dropout(p=dropout))


    def forward(self, x):
        original_shape = x.shape

        if x.ndim > 2:
            x = x.view(original_shape[0], -1)

        for i, layer in enumerate(self.layers):
            x = layer(x)

        return x
