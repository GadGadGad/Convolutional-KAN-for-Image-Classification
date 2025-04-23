import torch
import torch.nn as nn
from torch import Tensor
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Union, Tuple
from inspect import signature

from torchvision.ops.misc import SqueezeExcitation as SEModule 

from huggingface_hub import PyTorchModelHubMixin

from layers.kan_conv import CONV_KAN_FACTORY, _calculate_same_padding
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
        in_planes: int,
        out_planes: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = None, 
        bias: Optional[bool] = None,
        conv_layer: Callable[..., nn.Module] = nn.Conv2d,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        affine: bool = True,
    ) -> None:

        if padding is None:
             padding = (kernel_size - 1) // 2 * dilation

        if bias is None:
            bias = norm_layer is None

        if inplace is None:
            inplace = activation_layer is nn.ReLU

        layers: List[nn.Module] = [
            conv_layer(
                in_planes, out_planes, kernel_size, stride, padding,
                dilation=dilation, groups=groups, bias=bias
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_planes, affine=affine))

        if activation_layer is not None:
            params = {}
            if inplace and 'inplace' in signature(activation_layer).parameters:
                 params["inplace"] = True
            layers.append(activation_layer(**params))

        super().__init__(*layers)
        self.out_channels = out_planes


class InvertedResidualConfig:
    def __init__(
        self,
        input_channels: int, kernel: int, expanded_channels: int, out_channels: int,
        use_se: bool, activation: str, stride: int, dilation: int, width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)

class InvertedResidual(nn.Module):
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module],
        conv_layer_factory: Callable[..., nn.Module],
        replace_depthwise: bool = False,
        **factory_kwargs 
    ):
        super().__init__()
        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        self.replace_depthwise = replace_depthwise

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU
        self.activation_layer = activation_layer

        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                conv_layer_factory(
                    in_planes=cnf.input_channels,
                    out_planes=cnf.expanded_channels,
                    kernel_size=1,
                    stride=1,
                    groups=1,
                    activation_layer=activation_layer, 
                    norm_layer=norm_layer,
                     **factory_kwargs
                )
            )

        stride = 1 if cnf.dilation > 1 else cnf.stride
        kernel_size = cnf.kernel
        current_dilation = cnf.dilation
        groups = cnf.expanded_channels

        if self.replace_depthwise:
            dw_factory_kwargs = factory_kwargs.copy()
            layers.append(
                conv_layer_factory(
                    in_planes=groups,
                    out_planes=groups,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=groups,
                    dilation=current_dilation,
                    activation_layer=activation_layer, 
                    norm_layer=norm_layer,
                    **dw_factory_kwargs
                )
            )
        else:
            dw_norm = norm_layer if norm_layer is not None else nn.Identity
            dw_act = activation_layer
            dw_bias = norm_layer is None
            dw_affine = factory_kwargs.get('affine', True)
            dw_padding = _calculate_same_padding(kernel_size, current_dilation)

            layers.append(
                ConvNormActivation(
                    in_planes=groups,
                    out_planes=groups,
                    kernel_size=kernel_size,
                    stride=stride,
                    groups=groups,
                    norm_layer=dw_norm,
                    activation_layer=dw_act,
                    dilation=current_dilation,
                    padding=dw_padding,
                    conv_layer=nn.Conv2d,
                    bias=dw_bias,
                    affine=dw_affine,
                    inplace=None,
                )
            )

        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8) 
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels, activation=nn.ReLU))

        layers.append(
            conv_layer_factory(
                in_planes=cnf.expanded_channels,
                out_planes=cnf.out_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                activation_layer=None, 
                norm_layer=norm_layer,
                **factory_kwargs
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3KAN(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        arch: str, 
        num_classes: int = 1000,
        dropout: float = 0.2, 
        input_channels: int = 3,
        reduced_tail: bool = False, 
        dilated: bool = False,
        width_mult: float = 1.0, 

        conv_type: str = 'kanconv',
        kan_conv: Optional[str] = "KAN",
        kan_classifier: Optional[str] = "KAN",
        classifier_type: str = 'Linear',
        groups: int = 1,
        spline_order: int = 3,
        grid_size: int = 5,
        base_activation: Optional[Callable[..., nn.Module]] = None, 
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
        self.arch = arch

        if arch == "large":
            inverted_residual_setting, last_channel = _mobilenet_v3_conf(
                arch="mobilenet_v3_large", width_mult=width_mult, reduced_tail=reduced_tail, dilated=dilated
            )
            self.default_activation = nn.Hardswish
        elif arch == "small":
            inverted_residual_setting, last_channel = _mobilenet_v3_conf(
                arch="mobilenet_v3_small", width_mult=width_mult, reduced_tail=reduced_tail, dilated=dilated
            )
            self.default_activation = nn.Hardswish
        self.width_mult = width_mult

        block = InvertedResidual
        effective_norm_layer = partial(norm_layer, eps=0.001, momentum=0.01) if norm_layer is nn.BatchNorm2d else norm_layer
        effective_kan_norm_layer = kan_norm_layer if kan_norm_layer is not None else effective_norm_layer
        se_layer = partial(SEModule, scale_activation=nn.Hardsigmoid)

        activation_to_use = base_activation if base_activation is not None else self.default_activation

        conv_layer_factory: Callable[..., nn.Module]
        factory_kwargs = {
            "spline_order": spline_order, "grid_size": grid_size,
            "base_activation": activation_to_use,
            "grid_range": grid_range, "l1_decay": l1_decay,
            "dropout": conv_dropout,
            "degree": degree, "affine": affine,
             **{k: v for k, v in kwargs.items() if k in signature(CONV_KAN_FACTORY.get(kan_conv, nn.Conv2d)).parameters}
        }

        if conv_type == 'kanconv':
            if kan_conv is None or kan_conv not in CONV_KAN_FACTORY:
                kan_conv = "KAN"
            kan_conv_func = CONV_KAN_FACTORY[kan_conv]
            conv_layer_factory = partial(
                kan_conv_func,
                norm_layer=effective_kan_norm_layer,
                **factory_kwargs
            )
        elif conv_type == 'conv':
            def std_conv_norm_act_wrapper(
                in_planes: int, out_planes: int, kernel_size: Union[int, Tuple[int, int]],
                stride: Union[int, Tuple[int, int]] = 1, groups: int = 1, dilation: int = 1,
                norm_layer: Optional[Callable[..., nn.Module]] = effective_norm_layer,
                activation_layer: Optional[Callable[..., nn.Module]] = activation_to_use, # Outer activation
                affine: bool = affine,
                padding: Optional[Union[int, Tuple[int, int], str]] = None,
                spline_order=None, grid_size=None, base_activation=None, grid_range=None,
                l1_decay=None, dropout=None, degree=None, **other_kwargs
            ) -> ConvNormActivation:
                return ConvNormActivation(
                    in_planes=in_planes, out_planes=out_planes, kernel_size=kernel_size,
                    stride=stride, padding=padding, groups=groups, dilation=dilation,
                    norm_layer=norm_layer, activation_layer=activation_layer,
                    affine=affine, conv_layer=nn.Conv2d,
                    inplace=None 
                )
            conv_layer_factory = std_conv_norm_act_wrapper

        features: List[nn.Module] = []

        firstconv_output_channels = inverted_residual_setting[0].input_channels
        features.append(
            conv_layer_factory(
                in_planes=input_channels,
                out_planes=firstconv_output_channels,
                kernel_size=3,
                stride=2,
                groups=1,
                activation_layer=activation_to_use, 
                norm_layer=effective_norm_layer,
                 **factory_kwargs
            )
        )
        for cnf in inverted_residual_setting:
            features.append(block(cnf, effective_norm_layer, se_layer, conv_layer_factory, replace_depthwise, **factory_kwargs))

        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = cnf.expanded_channels 
        
        features.append(
            conv_layer_factory(
                in_planes=lastconv_input_channels,
                out_planes=lastconv_output_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                activation_layer=activation_to_use,
                norm_layer=effective_norm_layer,
                 **factory_kwargs
            )
        )
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        classifier_factory: Callable[..., nn.Module]
        classifier_dropout = classifier_dropout if classifier_dropout is not None else dropout
        classifier_spline_order = classifier_spline_order if classifier_spline_order is not None else spline_order
        classifier_grid_size = classifier_grid_size if classifier_grid_size is not None else grid_size
        classifier_base_activation = classifier_base_activation if classifier_base_activation is not None else nn.Hardswish # Default to V3 classifier activation
        classifier_grid_range = classifier_grid_range if classifier_grid_range is not None else grid_range
        classifier_l1_decay = classifier_l1_decay if classifier_l1_decay is not None else l1_decay
        classifier_degree = classifier_degree if classifier_degree is not None else degree

        head_kan_suffix = ""
        feat_dimension = lastconv_output_channels 
        classifier_output_dim = last_channel 

        if classifier_type == "KAN":
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
                "first_dropout": False,
            }
            classifier_kan_sig = signature(classifier_kan_func)
            valid_classifier_args = {k: v for k, v in classifier_kan_args.items() if k in classifier_kan_sig.parameters}

            classifier_factory = partial(classifier_kan_func, **valid_classifier_args)

            self.classifier = nn.Sequential(
                classifier_factory(feat_dimension, classifier_output_dim),
                # nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                classifier_factory(layers_hidden=[classifier_output_dim, num_classes]), 
            )

        elif classifier_type == 'Linear':
            self.classifier = nn.Sequential(
                nn.Linear(feat_dimension, classifier_output_dim),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(classifier_output_dim, num_classes),
            )
            head_kan_suffix = "_Linear"
        else:
            self.classifier = nn.Identity()
            head_kan_suffix = f"_{classifier_type}"


        self._initialize_weights()

        kan_conv_suffix = f"_{kan_conv.upper()}" if conv_type == 'kanconv' else "_CONV"
        replace_suffix = "_RDW" if replace_depthwise and conv_type=='kanconv' else ""
        arch_suffix = f"_{arch.upper()}"
        self.name = f"MobileNetV3KAN{head_kan_suffix}{kan_conv_suffix}{replace_suffix}{arch_suffix}_w{width_mult}"


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

def _mobilenet_v3_conf(
    arch: str, width_mult: float = 1.0, **kwargs: Any
) -> Tuple[List[InvertedResidualConfig], Optional[int]]:
    reduce_divider = 2 if kwargs.pop("reduced_tail", False) else 1
    dilation = 2 if kwargs.pop("dilated", False) else 1
    arch_settings = {
        "mobilenet_v3_large": (
            [16, 3, 16, 16, False, "RE", 1, 1],
            [16, 3, 64, 24, False, "RE", 2, 1],
            [24, 3, 72, 24, False, "RE", 1, 1],
            [24, 5, 72, 40, True, "RE", 2, 1],
            [40, 5, 120, 40, True, "RE", 1, 1],
            [40, 5, 120, 40, True, "RE", 1, 1],
            [40, 3, 240, 80, False, "HS", 2, 1],
            [80, 3, 200, 80, False, "HS", 1, 1],
            [80, 3, 184, 80, False, "HS", 1, 1],
            [80, 3, 184, 80, False, "HS", 1, 1],
            [80, 3, 480, 112, True, "HS", 1, 1],
            [112, 3, 672, 112, True, "HS", 1, 1],
            [112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation],
            [160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation],
            [160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation],
        ),
        "mobilenet_v3_small": (
            [16, 3, 16, 16, True, "RE", 2, 1],
            [16, 3, 72, 24, False, "RE", 2, 1],
            [24, 3, 88, 24, False, "RE", 1, 1],
            [24, 5, 96, 40, True, "HS", 2, 1],
            [40, 5, 240, 40, True, "HS", 1, 1],
            [40, 5, 240, 40, True, "HS", 1, 1],
            [40, 5, 120, 48, True, "HS", 1, 1],
            [48, 5, 144, 48, True, "HS", 1, 1],
            [48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation],
            [96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation],
            [96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation],
        ),
    }


    setting = arch_settings[arch]
    inverted_residual_setting: List[InvertedResidualConfig] = []
    last_channel = None

    for i, (ic, k, ec, oc, se, act, s, d) in enumerate(setting):
        cnf = InvertedResidualConfig(ic, k, ec, oc, se, act, s, d * dilation, width_mult)
        inverted_residual_setting.append(cnf)

    if arch == "mobilenet_v3_large":
        last_channel = _make_divisible(960 // reduce_divider * width_mult, 8)
    elif arch == "mobilenet_v3_small":
        last_channel = _make_divisible(576 // reduce_divider * width_mult, 8)

    return inverted_residual_setting, last_channel


def mobilenet_v3_kan(
        arch: str, 
        num_classes: int = 1000,
        input_channels: int = 3,
        dropout: float = 0.2,
        width_mult: float = 1.0,
        reduced_tail: bool = False,
        dilated: bool = False,
        conv_type: str = 'kanconv',
        kan_conv: Optional[str] = "KAN",
        kan_classifier: Optional[str] = "KAN",
        classifier_type: str = 'Linear',
        groups: int = 1,
        spline_order: int = 3,
        grid_size: int = 5,
        base_activation: Optional[Callable[..., nn.Module]] = nn.Hardswish, 
        grid_range: List = [-1, 1],
        l1_decay: float = 0.0,
        degree: Optional[int] = 3,
        affine: bool = True,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        kan_norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        replace_depthwise: bool = False,
        classifier_spline_order: Optional[int] = None,
        classifier_grid_size: Optional[int] = None,
        classifier_base_activation: Optional[Callable[..., nn.Module]] = nn.Hardswish,
        classifier_grid_range: Optional[List] = None,
        classifier_l1_decay: Optional[float] = None,
        classifier_dropout: Optional[float] = None,
        classifier_degree: Optional[int] = None,
        conv_dropout: float = 0.0,
        **kwargs: Any
    ) -> MobileNetV3KAN:
    
    model = MobileNetV3KAN(
        arch=arch,
        num_classes=num_classes,
        input_channels=input_channels,
        dropout=dropout,
        width_mult=width_mult,
        reduced_tail=reduced_tail,
        dilated=dilated,
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