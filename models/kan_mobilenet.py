import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Union, Tuple
from inspect import signature

from huggingface_hub import PyTorchModelHubMixin

from layers.kan_conv import CONV_KAN_FACTORY, _calculate_same_padding, conv
from .kans import MLP_KAN_FACTORY

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_planes: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
        inplace: bool = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., nn.Module] = nn.Conv2d,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        affine: bool = True, 
    ) -> None:

        if padding is None:
             padding = (kernel_size - 1) // 2 * dilation 

        if bias is None:
            bias = norm_layer is None

        layers: List[nn.Module] = [
            conv_layer(
                in_channels, out_planes, kernel_size, stride, padding,
                dilation=dilation, groups=groups, bias=bias
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_planes, affine=affine))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))

        super().__init__(*layers)
        self.out_channels = out_planes
class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_planes: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        conv_layer_factory: Callable[..., nn.Module], 
        replace_depthwise: bool = False,
        **factory_kwargs
    ):
        super().__init__()
        self.stride = stride
        self.replace_depthwise = replace_depthwise
        self.in_channels = in_channels
        self.out_planes = out_planes
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.conv_layer_factory = conv_layer_factory
        self.factory_kwargs = factory_kwargs

        if self.replace_depthwise:
            dw_factory_kwargs = factory_kwargs.copy()
            self.depthwise = conv_layer_factory(
                in_channels=in_channels,
                out_planes=in_channels, 
                kernel_size=3,
                stride=stride,
                groups=in_channels,
                activation_layer=activation_layer, 
                norm_layer=norm_layer,
                **dw_factory_kwargs
            )
        else:
            dw_norm = norm_layer if norm_layer is not None else nn.Identity
            dw_act = activation_layer if activation_layer is not None else nn.Identity
            dw_bias = norm_layer is None
            dw_affine = factory_kwargs.get('affine', True) 
            self.depthwise = ConvNormActivation(
                in_channels=in_channels,
                out_planes=in_channels,
                kernel_size=3,
                stride=stride,
                groups=in_channels,
                norm_layer=dw_norm,
                activation_layer=dw_act,
                conv_layer=nn.Conv2d, 
                bias=dw_bias,
                affine=dw_affine,
                inplace=True, 
            )

        self.pointwise = conv_layer_factory(
            in_channels=in_channels,
            out_planes=out_planes,
            kernel_size=1,
            stride=1,
            groups=1, 
            activation_layer=activation_layer, 
            norm_layer=norm_layer,
            **factory_kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MobileNetV1KAN(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        dropout: float = 0.2,
        input_channels: int = 3,
        conv_type: str = 'kanconv',
        kan_conv: Optional[str] = "KAN",
        kan_classifier: Optional[str] = "KAN",
        classifier_type: str = 'Linear',
        groups: int = 1,
        spline_order: int = 3,
        grid_size: int = 5,
        base_activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
        grid_range: List = [-1, 1],
        l1_decay: float = 0.0,
        degree: Optional[int] = 3,
        affine: bool = True, 
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        kan_norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        replace_depthwise: bool = False, 
        classifier_spline_order: Optional[int] = None,
        classifier_grid_size: Optional[int] = None,
        classifier_base_activation: Optional[Callable[..., nn.Module]] = None,
        classifier_grid_range: Optional[List] = None,
        classifier_l1_decay: Optional[float] = None,
        classifier_dropout: Optional[float] = None,
        classifier_degree: Optional[int] = None,
        conv_dropout: float = 0.0, 
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if kan_norm_layer is None:
             kan_norm_layer = norm_layer
             
        block = DepthwiseSeparableConv
        activation_layer = base_activation if base_activation else nn.ReLU 

        conv_layer_factory: Callable[..., nn.Module]
        factory_kwargs = {
            "spline_order": spline_order,
            "grid_size": grid_size,
            "base_activation": activation_layer,
            "grid_range": grid_range,
            "l1_decay": l1_decay,
            "dropout": conv_dropout, 
            "degree": degree,
            "affine": affine,
             **{k: v for k, v in kwargs.items() if k in signature(CONV_KAN_FACTORY.get(kan_conv, nn.Conv2d)).parameters}
        }

        if conv_type == 'kanconv':
            if kan_conv is None or kan_conv not in CONV_KAN_FACTORY:
                kan_conv = "KAN"
            kan_conv_func = CONV_KAN_FACTORY[kan_conv]

            kan_conv_args = {
                "spline_order": spline_order,
                "grid_size": grid_size,
                "base_activation": base_activation,
                "grid_range": grid_range,
                "dropout": conv_dropout,
                "l1_decay": l1_decay,
                "groups": groups,
                "norm_layer": kan_norm_layer,
                "affine": affine,
                "degree": degree,
            }
            relevant_kwargs = {k: v for k, v in kwargs.items() if k in signature(kan_conv_func).parameters}
            kan_conv_args.update(relevant_kwargs)

            conv_layer_factory = partial(kan_conv_func, **kan_conv_args)
            
        elif conv_type == 'conv':
            def std_conv_norm_act_wrapper(
                in_channels: int, out_planes: int, kernel_size: Union[int, Tuple[int, int]],
                stride: Union[int, Tuple[int, int]] = 1, groups: int = 1, dilation: int = 1,
                norm_layer: Optional[Callable[..., nn.Module]] = norm_layer,
                activation_layer: Optional[Callable[..., nn.Module]] = activation_layer,
                affine: bool = affine,
                padding: Optional[Union[int, Tuple[int, int], str]] = None,
                spline_order=None, grid_size=None, base_activation=None, grid_range=None,
                l1_decay=None, dropout=None, degree=None, **other_kwargs
            ) -> ConvNormActivation:
                if padding is None:
                     if isinstance(kernel_size, int): 
                        padding = _calculate_same_padding(kernel_size, dilation)
                     elif isinstance(kernel_size, tuple):
                        padding = _calculate_same_padding(kernel_size, (dilation, dilation))
                return ConvNormActivation(
                    in_channels=in_channels, out_planes=out_planes, kernel_size=kernel_size,
                    stride=stride, padding=padding, groups=groups, dilation=dilation,
                    norm_layer=norm_layer, activation_layer=activation_layer,
                    affine=affine, conv_layer=nn.Conv2d,
                    inplace=True if activation_layer is nn.ReLU else False
                )
            conv_layer_factory = std_conv_norm_act_wrapper
            
        input_channel = _make_divisible(32 * width_mult, 8)
        
        interverted_residual_setting = [
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            (512, 1), (512, 1), (512, 1), (512, 1), (512, 1), 
            (1024, 2),
            (1024, 1),
        ]

        features: List[nn.Module] = [
            conv_layer_factory(
                in_channels=input_channels,
                out_planes=input_channel,
                kernel_size=3,
                stride=2, 
                groups=1, 
                activation_layer=activation_layer, 
                norm_layer=norm_layer, 
                 **factory_kwargs
            )
        ]

        for c, s in interverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 8)
            features.append(block(input_channel, output_channel, s,
                                  norm_layer=norm_layer,
                                  activation_layer=activation_layer,
                                  conv_layer_factory=conv_layer_factory,
                                  replace_depthwise=replace_depthwise,
                                  **factory_kwargs))
            input_channel = output_channel

        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        last_channel = input_channel

        classifier_factory: Callable[..., nn.Module]
        classifier_dropout = classifier_dropout if classifier_dropout is not None else dropout
        classifier_spline_order = classifier_spline_order if classifier_spline_order is not None else spline_order
        classifier_grid_size = classifier_grid_size if classifier_grid_size is not None else grid_size
        classifier_base_activation = classifier_base_activation if classifier_base_activation is not None else activation_layer 
        classifier_grid_range = classifier_grid_range if classifier_grid_range is not None else grid_range
        classifier_l1_decay = classifier_l1_decay if classifier_l1_decay is not None else l1_decay
        classifier_degree = classifier_degree if classifier_degree is not None else degree 

        head_kan_suffix = ""
        if classifier_type in ['KAN']:
            _kan_classifier_type = kan_classifier if kan_classifier else 'KAN'
            classifier_kan_func = MLP_KAN_FACTORY[_kan_classifier_type]
            head_kan_suffix = f"_{_kan_classifier_type.upper()}"

            classifier_kan_args = {
                "dropout": classifier_dropout,
                "spline_order": classifier_spline_order,
                "grid_size": classifier_grid_size,
                "base_activation": classifier_base_activation,
                "grid_range": classifier_grid_range,
                "l1_decay": classifier_l1_decay,
                "degree": classifier_degree,
                "first_dropout": True,
            }
            classifier_kan_sig = signature(classifier_kan_func)
            valid_classifier_args = {k: v for k, v in classifier_kan_args.items() if k in classifier_kan_sig.parameters}

            classifier_factory = partial(classifier_kan_func, **valid_classifier_args)
            self.classifier = nn.Sequential(
                nn.Dropout(p=classifier_dropout),
                classifier_factory(layers_hidden=[last_channel, num_classes])
            )
        elif classifier_type == 'Linear':
            self.classifier = nn.Sequential(
                nn.Dropout(p=classifier_dropout),
                nn.Linear(last_channel, num_classes),
            )
            head_kan_suffix = "_Linear"
        else:
             self.classifier = nn.Identity()
             head_kan_suffix = f"_{classifier_type}"


        self._initialize_weights()

        kan_conv_suffix = f"_{kan_conv.upper()}" if conv_type == 'kanconv' else "_CONV"
        replace_suffix = "_RDW" if replace_depthwise and conv_type=='kanconv' else ""
        self.name = f"MobileNetV1KAN{head_kan_suffix}{kan_conv_suffix}{replace_suffix}_w{width_mult}"


    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def mobilenet_v1_kan(
        num_classes: int = 1000,
        width_mult: float = 1.0,
        input_channels: int = 3,
        dropout: float = 0.2,
        conv_type: str = 'kanconv',
        kan_conv: Optional[str] = "KAN",
        kan_classifier: Optional[str] = "KAN",
        classifier_type: str = 'Linear',
        groups: int = 1,
        spline_order: int = 3,
        grid_size: int = 5,
        base_activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
        grid_range: List = [-1, 1],
        l1_decay: float = 0.0,
        degree: Optional[int] = 3,
        affine: bool = True,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        kan_norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        replace_depthwise: bool = False,
        classifier_spline_order: Optional[int] = None,
        classifier_grid_size: Optional[int] = None,
        classifier_base_activation: Optional[Callable[..., nn.Module]] = None,
        classifier_grid_range: Optional[List] = None,
        classifier_l1_decay: Optional[float] = None,
        classifier_dropout: Optional[float] = None,
        classifier_degree: Optional[int] = None,
        conv_dropout: float = 0.0,
        **kwargs: Any
    ) -> MobileNetV1KAN:

    model = MobileNetV1KAN(
        num_classes=num_classes,
        width_mult=width_mult,
        input_channels=input_channels,
        dropout=dropout,
        conv_type=conv_type,
        kan_conv=kan_conv,
        kan_classifier=kan_classifier,
        classifier_type=classifier_type,
        groups=groups, 
        spline_order=spline_order,
        grid_size=grid_size,
        base_activation=base_activation,
        grid_range=grid_range,
        l1_decay=l1_decay,
        degree=degree,
        affine=affine,
        norm_layer=norm_layer,
        kan_norm_layer=kan_norm_layer,
        replace_depthwise=replace_depthwise,
        classifier_spline_order=classifier_spline_order,
        classifier_grid_size=classifier_grid_size,
        classifier_base_activation=classifier_base_activation,
        classifier_grid_range=classifier_grid_range,
        classifier_l1_decay=classifier_l1_decay,
        classifier_dropout=classifier_dropout,
        classifier_degree=classifier_degree,
        conv_dropout=conv_dropout,
        **kwargs,
    )
    return model