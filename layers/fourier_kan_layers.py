import math
import torch
import torch.nn as nn
import numpy as np
from inspect import signature
from typing import Tuple, Type, Union
class FourierKANLayer(nn.Module):
    def __init__( self,
                 input_dim,
                 output_dim,
                 grid_size,
                 add_bias=True,
                 smooth_initialization=False):
        super(FourierKANLayer,self).__init__()
        self.grid_size= grid_size
        self.add_bias = add_bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # With smooth_initialization, fourier coefficients are attenuated by the square of their frequency.
        # This makes KAN's scalar functions smooth at initialization.
        # Without smooth_initialization, high grid_sizes will lead to high-frequency scalar functions,
        # with high derivatives and low correlation between similar inputs.
        grid_norm_factor = (torch.arange(grid_size) + 1)**2 if smooth_initialization else np.sqrt(grid_size)
        
        #The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        #then each coordinates of the output is of unit variance 
        #independently of the various sizes
        self.fouriercoeffs = nn.Parameter( torch.randn(2,output_dim,input_dim,grid_size) / 
                                                (np.sqrt(input_dim) * grid_norm_factor ) )
        if( self.add_bias ):
            self.bias  = nn.Parameter( torch.zeros(1,output_dim))

    #x.shape ( ... , indim ) 
    #out.shape ( ..., output_dim)
    def forward(self,x):
        xshp = x.shape
        outshape = xshp[0:-1]+(self.output_dim,)
        x = torch.reshape(x,(-1,self.input_dim))
        #Starting at 1 because constant terms are in the bias
        k = torch.reshape( torch.arange(1,self.grid_size+1,device=x.device),(1,1,1,self.grid_size))
        xrshp = torch.reshape(x,(x.shape[0],1,x.shape[1],1) ) 
        #This should be fused to avoid materializing memory
        c = torch.cos( k*xrshp )
        s = torch.sin( k*xrshp )
        #We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them 
        y =  torch.sum( c*self.fouriercoeffs[0:1],(-2,-1)) 
        y += torch.sum( s*self.fouriercoeffs[1:2],(-2,-1))
        if( self.add_bias):
            y += self.bias
        #End fuse
        '''
        #You can use einsum instead to reduce memory usage
        #It stills not as good as fully fused but it should help
        #einsum is usually slower though
        c = torch.reshape(c,(1,x.shape[0],x.shape[1],self.grid_size))
        s = torch.reshape(s,(1,x.shape[0],x.shape[1],self.grid_size))
        y2 = torch.einsum( "dbik,djik->bj", torch.concat([c,s],axis=0) ,self.fouriercoeffs )
        if( self.add_bias):
            y2 += self.bias
        diff = torch.sum((y2-y)**2)
        print("diff")
        print(diff) #should be ~0
        '''
        y = torch.reshape( y, outshape)
        return y
class FourierKANConvNDLayer(nn.Module):
    def __init__(self,
                 conv_class: Type[nn.Module],
                 norm_class: Type[nn.Module],
                 input_dim: int,
                 output_dim: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 grid_size: int, # Number of Fourier frequencies (k=1 to grid_size)
                 groups: int = 1,
                 padding: Union[int, Tuple[int, ...], str] = 0,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 ndim: int = 2,
                 base_activation: Type[nn.Module] = nn.GELU,
                 dropout: float = 0.0,
                 smooth_initialization: bool = False, 
                 **norm_kwargs):
        super(FourierKANConvNDLayer, self).__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')
        if grid_size < 1:
            raise ValueError('grid_size must be at least 1')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.grid_size = grid_size
        self.groups = groups
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.ndim = ndim
        self.base_activation = base_activation() if base_activation is not None else nn.Identity()
        self.norm_kwargs = norm_kwargs
        # self.smooth_initialization = smooth_initialization # Store if needed

        # Calculate dimensions per group
        self.input_dim_group = input_dim // groups
        self.output_dim_group = output_dim // groups
        self.fourier_input_dim_group = self.input_dim_group * (2 * grid_size)

        # --- Modules ---
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

        self.fourier_conv = nn.ModuleList([
            conv_class(
                self.fourier_input_dim_group,
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

        # Initialize weights (Standard Kaiming for base, potentially adjusted for Fourier)
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        for conv_layer in self.fourier_conv:
            # TODO: Consider initialization strategy from original ourierKANLayer if needed
            # Example: Apply scaling factor based on grid_size if smooth_initialization=True
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear') # Adjust gain potentially

    def _compute_fourier_basis(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        channels = x.shape[1]
        spatial_dims = x.shape[2:]

        # Create frequency tensor k = [1, 2, ..., grid_size]
        k = torch.arange(1, self.grid_size + 1, device=x.device, dtype=x.dtype)
        # Reshape k for broadcasting: (1, 1, grid_size, 1, ...)
        k_shape = [1, 1, self.grid_size] + [1] * len(spatial_dims)
        k = k.view(*k_shape)

        # Expand x for broadcasting: (batch, channels, 1, spatial_dims...)
        x_expanded = x.unsqueeze(2)

        # Calculate k * x
        kx = k * x_expanded 

        # Compute cos(kx) and sin(kx)
        cos_basis = torch.cos(kx)
        sin_basis = torch.sin(kx)

        fourier_basis = torch.cat((cos_basis, sin_basis), dim=2)

        fourier_features = fourier_basis.view(batch_size, channels * (2 * self.grid_size), *spatial_dims)
        return fourier_features

    def _forward_kan_group(self, x: torch.Tensor, group_index: int) -> torch.Tensor:
        """Applies the KAN logic for a single group."""
        # Base path
        base_output = self.base_conv[group_index](self.base_activation(x))

        # Fourier path
        fourier_features = self._compute_fourier_basis(x)
        fourier_output = self.fourier_conv[group_index](fourier_features)

        combined_output = base_output + fourier_output

        output = self.prelus[group_index](self.layer_norm[group_index](combined_output))

        if self.dropout is not None:
            output = self.dropout(output)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Fourier KAN convolutional layer."""
        split_x = torch.split(x, self.input_dim_group, dim=1)
        output = [self._forward_kan_group(_x, group_ind) for group_ind, _x in enumerate(split_x)]
        y = torch.cat(output, dim=1)
        return y


class FourierKANConv3DLayer(FourierKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, grid_size, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super().__init__(
            conv_class=nn.Conv3d, norm_class=norm_layer, input_dim=input_dim, output_dim=output_dim,
            kernel_size=kernel_size, grid_size=grid_size, groups=groups, padding=padding, stride=stride,
            dilation=dilation, ndim=3, base_activation=base_activation, dropout=dropout, **norm_kwargs
        )

class FourierKANConv2DLayer(FourierKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, grid_size, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super().__init__(
            conv_class=nn.Conv2d, norm_class=norm_layer, input_dim=input_dim, output_dim=output_dim,
            kernel_size=kernel_size, grid_size=grid_size, groups=groups, padding=padding, stride=stride,
            dilation=dilation, ndim=2, base_activation=base_activation, dropout=dropout, **norm_kwargs
        )

class FourierKANConv1DLayer(FourierKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, grid_size, groups=1, padding=0, stride=1, dilation=1,
                 base_activation=nn.GELU, dropout=0.0, norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super().__init__(
            conv_class=nn.Conv1d, norm_class=norm_layer, input_dim=input_dim, output_dim=output_dim,
            kernel_size=kernel_size, grid_size=grid_size, groups=groups, padding=padding, stride=stride,
            dilation=dilation, ndim=1, base_activation=base_activation, dropout=dropout, **norm_kwargs
        )

def demo():
    bs = 10
    L = 3 #Not necessary just to show that additional dimensions are batched like Linear
    inputdim = 50
    hidden = 200
    outdim = 100
    grid_size = 30 # Smaller grid_size for conv demo

    device = "cpu" #"cuda"
    kernel_size = 3
    padding = 1

    # Example using 2D Convolutional Layers
    H, W = 16, 16 # Example spatial dimensions
    fkan_conv1 = FourierKANConv2DLayer(inputdim, hidden, kernel_size, grid_size, padding=padding, norm_layer=nn.BatchNorm2d).to(device)
    fkan_conv2 = FourierKANConv2DLayer(hidden, outdim, kernel_size, grid_size, padding=padding, norm_layer=nn.BatchNorm2d).to(device)

    x0 = torch.randn(bs, inputdim, H, W).to(device)

    h = fkan_conv1(x0)
    y = fkan_conv2(h)
    print("--- Conv 2D Example ---")
    print("x0.shape:", x0.shape)
    print("h.shape:", h.shape)
    print("y.shape:", y.shape)
    print("torch.mean(h):", torch.mean(h).item())
    print("torch.mean(torch.var(h, dim=1)):", torch.mean(torch.var(h, dim=1)).item()) # Variance across channels
    print("torch.mean(y):", torch.mean(y).item())
    print("torch.mean(torch.var(y, dim=1)):", torch.mean(torch.var(y, dim=1)).item()) # Variance across channels

    # Use FourierKANConv1DLayer with kernel_size=1 to mimic linear layer behavior on flattened features
    print("\n--- Mimicking Linear Layers with Conv1D (kernel_size=1) ---")
    fkan_lin1 = FourierKANConv1DLayer(inputdim, hidden, kernel_size=1, grid_size=grid_size, norm_layer=nn.BatchNorm1d).to(device)
    fkan_lin2 = FourierKANConv1DLayer(hidden, outdim, kernel_size=1, grid_size=grid_size, norm_layer=nn.BatchNorm1d).to(device)

    # Treat spatial dimensions as sequence length for Conv1D
    x_seq = torch.randn(bs, inputdim, L).to(device) # Shape (batch, channels, length)

    h_seq = fkan_lin1(x_seq)
    y_seq = fkan_lin2(h_seq)
    print("x_seq.shape:", x_seq.shape)
    print("h_seq.shape:", h_seq.shape)
    print("y_seq.shape:", y_seq.shape)
    print("torch.mean(h_seq):", torch.mean(h_seq).item())
    print("torch.mean(torch.var(h_seq, dim=1)):", torch.mean(torch.var(h_seq, dim=1)).item()) # Variance across channels
    print("torch.mean(y_seq):", torch.mean(y_seq).item())
    print("torch.mean(torch.var(y_seq, dim=1)):", torch.mean(torch.var(y_seq, dim=1)).item()) # Variance across channels

