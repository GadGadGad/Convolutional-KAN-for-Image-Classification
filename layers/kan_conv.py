from typing import Callable, List, Optional, Union, Tuple

import torch.nn as nn

from layers import (
    BersnsteinKANConv2DLayer, BesselKANConv2DLayer, ChebyKANConv2DLayer, FastKANConv2DLayer, FibonacciKANConv2DLayer, \
    FourierKANConv2DLayer, GegenbauerKANConv2DLayer, GRAMKANConv2DLayer, HermiteKANConv2DLayer, JacobiKANConv2DLayer, KANConv2DLayer, \
    LaguerreKANConv2DLayer, LegendreKANConv2DLayer, LucasKANConv2DLayer, ReLUKANConv2DLayer, TaylorKANConv2DLayer, WavKANConv2DLayer
)
from utils.regularization import L1, L2

def _calculate_same_padding(kernel_size: Union[int, Tuple[int, int]],
                            dilation: Union[int, Tuple[int, int]]) -> Union[int, Tuple[int, int]]:
    """Calculates padding for 'same' output spatial size (assuming stride 1)."""
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    padding_h = (dilation[0] * (kernel_size[0] - 1)) // 2
    padding_w = (dilation[1] * (kernel_size[1] - 1)) // 2
    if padding_h == padding_w and isinstance(kernel_size, tuple) and kernel_size[0] == kernel_size[1]:
         return padding_h
    else:
        return (padding_h, padding_w)

def kan_conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, Tuple[int, int]],
    spline_order: int = 3,
    groups: int = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    grid_size: int = 5,
    base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
    grid_range: List = [-1, 1],
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
    **norm_kwargs
) -> KANConv2DLayer:
    """
    Generalized KAN convolution layer. Automatically calculates 'same' padding
    if `padding` is not provided.
    """
    if padding is None:
        padding = _calculate_same_padding(kernel_size, dilation)

    conv = KANConv2DLayer(
        input_dim=in_planes,
        output_dim=out_planes,
        kernel_size=kernel_size,
        spline_order=spline_order,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        grid_size=grid_size,
        base_activation=base_activation,
        grid_range=grid_range,
        dropout=dropout,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv

def conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, Tuple[int, int]],
    groups: int = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
    norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    **kwargs
) -> nn.Sequential:
    """
    Generalized standard convolution layer with optional normalization and activation.
    Automatically calculates 'same' padding if `padding` is not provided.
    """
    if padding is None:
        padding = _calculate_same_padding(kernel_size, dilation)

    layers = []
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))

    conv_layer = nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=norm_layer is None 
    )
    if l1_decay > 0:
        layers.append(L1(conv_layer, l1_decay))
    else:
        layers.append(conv_layer)

    if norm_layer is not None:
        layers.append(norm_layer(out_planes))

    if base_activation is not None:
        layers.append(base_activation())

    return nn.Sequential(*layers)


def legendrekan_conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, Tuple[int, int]],
    degree: int = 3,
    groups: int = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    dropout: float = 0.0,
    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
    l1_decay: float = 0.0,
    **norm_kwargs
) -> LegendreKANConv2DLayer:
    """
    Generalized LegendreKAN convolution layer. Automatically calculates 'same' padding
    if `padding` is not provided.
    """
    if padding is None:
        padding = _calculate_same_padding(kernel_size, dilation)

    conv = LegendreKANConv2DLayer(
        input_dim=in_planes,
        output_dim=out_planes,
        kernel_size=kernel_size,
        degree=degree,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv

def gramkan_conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, Tuple[int, int]],
    degree: int = 3,
    groups: int = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    dropout: float = 0.0,
    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
    l1_decay: float = 0.0,
    **norm_kwargs
) -> GRAMKANConv2DLayer:
    """
    Generalized KAGN convolution layer. Automatically calculates 'same' padding
    if `padding` is not provided.
    """
    if padding is None:
        padding = _calculate_same_padding(kernel_size, dilation)

    conv = GRAMKANConv2DLayer(
        input_dim=in_planes,
        output_dim=out_planes,
        kernel_size=kernel_size,
        degree=degree,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def chebykan_conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, Tuple[int, int]],
    degree: int = 3,
    groups: int = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
    **norm_kwargs
) -> ChebyKANConv2DLayer:
    """
    Generalized KACN convolution layer. Automatically calculates 'same' padding
    if `padding` is not provided.
    """
    if padding is None:
        padding = _calculate_same_padding(kernel_size, dilation)
    conv = ChebyKANConv2DLayer(
        input_dim=in_planes,
        output_dim=out_planes,
        kernel_size=kernel_size,
        degree=degree,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        dropout=dropout,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def fastkan_conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, Tuple[int, int]],
    groups: int = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    grid_size: int = 8, # Note: Default changed in original code
    base_activation: Callable[..., nn.Module] = nn.SiLU, # Note: Default changed in original code
    grid_range: List = [-2, 2], # Note: Default changed in original code
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
    **norm_kwargs
) -> FastKANConv2DLayer:
    """
    Generalized FastKAN convolution layer. Automatically calculates 'same' padding
    if `padding` is not provided.
    """
    if padding is None:
        padding = _calculate_same_padding(kernel_size, dilation)

    conv = FastKANConv2DLayer(
        input_dim=in_planes,
        output_dim=out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        grid_size=grid_size,
        base_activation=base_activation,
        grid_range=grid_range,
        dropout=dropout,
        l1_decay=l1_decay,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv

def wavkan_conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, Tuple[int, int]],
    groups: int = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    wavelet_type: str = 'mexican_hat',
    wav_version: str = 'fast',
    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
    **norm_kwargs
) -> WavKANConv2DLayer:
    """
    Generalized WavKAN convolution layer. Automatically calculates 'same' padding
    if `padding` is not provided.
    """
    if padding is None:
        padding = _calculate_same_padding(kernel_size, dilation)

    conv = WavKANConv2DLayer(
        input_dim=in_planes,
        output_dim=out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        wavelet_type=wavelet_type,
        wav_version=wav_version,
        dropout=dropout,
        l1_decay=l1_decay,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv

def bersnsteinkan_conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, Tuple[int, int]],
    groups: int = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    degree: int = 3,
    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
    **norm_kwargs
) -> BersnsteinKANConv2DLayer:
    if padding is None:
        padding = _calculate_same_padding(kernel_size, dilation)
    
    conv = BersnsteinKANConv2DLayer(
        input_dim=in_planes,
        output_dim=out_planes,
        kernel_size=kernel_size,
        degree=degree,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
        dropout=dropout,
        l1_decay=l1_decay,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv

def besselkan_conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, Tuple[int, int]],
    groups: int = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    degree: int = 3,
    base_activation: nn.Module=nn.GELU,
    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
    **norm_kwargs
) -> BesselKANConv2DLayer:
    if padding is None:
        padding = _calculate_same_padding(kernel_size, dilation)
    
    conv = BesselKANConv2DLayer(
        input_dim=in_planes,
        output_dim=out_planes,
        kernel_size=kernel_size,
        degree=degree,
        groups=groups,
        padding=padding,
        stride=stride,
        l1_decay=l1_decay,
        dropout=dropout,
        base_activation=base_activation,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def fibonaccikan_conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, Tuple[int, int]],
    groups: int = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    degree: int = 3,
    base_activation: nn.Module=nn.GELU,
    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
    **norm_kwargs
) -> FibonacciKANConv2DLayer:
    if padding is None:
        padding = _calculate_same_padding(kernel_size, dilation)
    
    conv = FibonacciKANConv2DLayer(
        input_dim=in_planes,
        output_dim=out_planes,
        kernel_size=kernel_size,
        degree=degree,
        groups=groups,
        padding=padding,
        stride=stride,
        l1_decay=l1_decay,
        dropout=dropout,
        base_activation=base_activation,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv

def fourierkan_conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, Tuple[int, int]],
    groups: int = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    grid_size: int = 3,
    base_activation: nn.Module=nn.GELU,
    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
    **norm_kwargs
) -> FourierKANConv2DLayer:
    if padding is None:
        padding = _calculate_same_padding(kernel_size, dilation)
    
    conv = FourierKANConv2DLayer(
        input_dim=in_planes,
        output_dim=out_planes,
        kernel_size=kernel_size,
        grid_size=grid_size,
        groups=groups,
        padding=padding,
        stride=stride,
        l1_decay=l1_decay,
        dropout=dropout,
        base_activation=base_activation,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv


def gegenbauerkan_conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, Tuple[int, int]],
    groups: int = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    degree: int = 3,
    alpha_param: float=0.0,
    base_activation: nn.Module=nn.GELU,
    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
    **norm_kwargs
) -> GegenbauerKANConv2DLayer:
    if padding is None:
        padding = _calculate_same_padding(kernel_size, dilation)
    
    conv = GegenbauerKANConv2DLayer(
        input_dim=in_planes,
        output_dim=out_planes,
        kernel_size=kernel_size,
        degree=degree,
        groups=groups,
        padding=padding,
        stride=stride,
        l1_decay=l1_decay,
        dropout=dropout,
        alpha_param=alpha_param,
        base_activation=base_activation,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv

def hermitekan_conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, Tuple[int, int]],
    groups: int = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    degree: int = 3,
    base_activation: nn.Module=nn.GELU,
    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
    **norm_kwargs
) -> HermiteKANConv2DLayer:
    if padding is None:
        padding = _calculate_same_padding(kernel_size, dilation)
    
    conv = HermiteKANConv2DLayer(
        input_dim=in_planes,
        output_dim=out_planes,
        kernel_size=kernel_size,
        degree=degree,
        groups=groups,
        padding=padding,
        stride=stride,
        l1_decay=l1_decay,
        dropout=dropout,
        base_activation=base_activation,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv

def jacobikan_conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, Tuple[int, int]],
    groups: int = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    degree: int = 3,
    a: float = 1.0,
    b: float = 1.0,
    base_activation: nn.Module=nn.GELU,
    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
    **norm_kwargs
) -> JacobiKANConv2DLayer:
    if padding is None:
        padding = _calculate_same_padding(kernel_size, dilation)
    
    conv = JacobiKANConv2DLayer(
        input_dim=in_planes,
        output_dim=out_planes,
        kernel_size=kernel_size,
        degree=degree,
        a=a,
        b=b,
        groups=groups,
        padding=padding,
        stride=stride,
        l1_decay=l1_decay,
        dropout=dropout,
        base_activation=base_activation,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv

def laguerrekan_conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, Tuple[int, int]],
    groups: int = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    degree: int = 3,
    alpha: float= 1.0,
    base_activation: nn.Module=nn.GELU,
    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
    **norm_kwargs
) -> LaguerreKANConv2DLayer:
    if padding is None:
        padding = _calculate_same_padding(kernel_size, dilation)
    
    conv = LaguerreKANConv2DLayer(
        input_dim=in_planes,
        output_dim=out_planes,
        kernel_size=kernel_size,
        degree=degree,
        alpha=alpha,
        groups=groups,
        padding=padding,
        stride=stride,
        l1_decay=l1_decay,
        dropout=dropout,
        base_activation=base_activation,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv

def lucaskan_conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, Tuple[int, int]],
    groups: int = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    degree: int = 3,
    base_activation: nn.Module=nn.GELU,
    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
    **norm_kwargs
) -> LucasKANConv2DLayer:
    if padding is None:
        padding = _calculate_same_padding(kernel_size, dilation)
    
    conv = LucasKANConv2DLayer(
        input_dim=in_planes,
        output_dim=out_planes,
        kernel_size=kernel_size,
        degree=degree,
        groups=groups,
        padding=padding,
        stride=stride,
        l1_decay=l1_decay,
        dropout=dropout,
        base_activation=base_activation,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv

def relukan_conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, Tuple[int, int]],
    groups: int = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    g: int = 5,
    k: int = 3,
    train_ab: bool = True,
    base_activation: nn.Module=nn.GELU,
    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
    **norm_kwargs
) -> ReLUKANConv2DLayer:
    if padding is None:
        padding = _calculate_same_padding(kernel_size, dilation)
    
    conv = ReLUKANConv2DLayer(
        input_dim=in_planes,
        output_dim=out_planes,
        kernel_size=kernel_size,
        g=g,
        k=k,
        train_ab=train_ab,
        groups=groups,
        padding=padding,
        stride=stride,
        l1_decay=l1_decay,
        dropout=dropout,
        base_activation=base_activation,
        norm_layer=norm_layer,
        **norm_kwargs
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv

def taylorkan_conv(
    in_planes: int,
    out_planes: int,
    kernel_size: Union[int, Tuple[int, int]],
    groups: int = 1,
    stride: Union[int, Tuple[int, int]] = 1,
    dilation: Union[int, Tuple[int, int]] = 1,
    padding: Optional[Union[int, Tuple[int, int]]] = None,
    l1_decay: float = 0.0,
    dropout: float = 0.0,
    degree: int = 3,
    base_activation: Optional[Callable[..., nn.Module]] = nn.GELU,
    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
    **norm_kwargs
) -> TaylorKANConv2DLayer:
    if padding is None:
        padding = _calculate_same_padding(kernel_size, dilation)
    conv = TaylorKANConv2DLayer(
        input_dim=in_planes,
        output_dim=out_planes,
        kernel_size=kernel_size,
        degree=degree,
        groups=groups,
        padding=padding,
        stride=stride,
        dropout=dropout,
        base_activation=base_activation,
        norm_layer=norm_layer,
        **norm_kwargs,
    )
    if l1_decay > 0:
        conv = L1(conv, l1_decay)
    return conv

CONV_KAN_FACTORY = {
    "KAN": kan_conv,
    "FastKAN": fastkan_conv,
    "LegendreKAN": legendrekan_conv,
    "GRAMKAN": gramkan_conv,
    "ChebyKAN": chebykan_conv,
    "WavKAN": wavkan_conv,
    "BersnsteinKAN": bersnsteinkan_conv,
    "BesselKAN": besselkan_conv,
    "FibonacciKAN": fibonaccikan_conv,
    "FourierKAN": fourierkan_conv,
    "GegenbauerKAN": gegenbauerkan_conv,
    "HermiteKAN": hermitekan_conv,
    "JacobiKAN": jacobikan_conv,
    "LaguerreKAN": laguerrekan_conv,
    "LucasKAN": lucaskan_conv,
    "ReLUKAN": relukan_conv,
    "TaylorKAN": taylorkan_conv,
    "conv": conv,
}
