import torch.nn as nn
from torch import Tensor
from functools import partial
from typing import Any, Callable, List, Optional, Sequence, Union, Tuple

from huggingface_hub import PyTorchModelHubMixin
from inspect import signature
from layers.kan_conv import CONV_KAN_FACTORY, _calculate_same_padding
from .kans import MLP_KAN_FACTORY

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py
    """
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
            padding = _calculate_same_padding(kernel_size, dilation)

        if bias is None:
            bias = norm_layer is None or not affine

        if inplace is None and activation_layer is not None:
             inplace = activation_layer in [nn.ReLU, nn.ReLU6]

        layers = [
            conv_layer(
                in_planes,
                out_planes,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_planes, affine=affine))

        if activation_layer is not None:
            params = {}
            act_sig = signature(activation_layer)
            if 'inplace' in act_sig.parameters:
                 params["inplace"] = inplace if inplace is not None else False
            layers.append(activation_layer(**params))

        super().__init__(*layers)
        self.out_channels = out_planes


class InvertedResidual(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]],
        activation_layer: Callable[..., nn.Module], 
        conv_layer_factory: Callable[..., nn.Module],
        replace_depthwise: bool = False, 
        **factory_kwargs 
    ) -> None:
        super().__init__()
        self.stride = stride

        hidden_dim = int(round(input_dim * expand_ratio))
        self.use_res_connect = self.stride == 1 and input_dim == output_dim

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            layers.append(
                conv_layer_factory(
                    in_planes=input_dim,
                    out_planes=hidden_dim,
                    kernel_size=1,
                    stride=1, 
                    activation_layer=activation_layer, 
                    norm_layer=norm_layer,
                    **factory_kwargs 
                )
            )

        if replace_depthwise:
            layers.append(
                conv_layer_factory(
                    in_planes=hidden_dim,
                    out_planes=hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    groups=hidden_dim,
                    activation_layer=activation_layer,
                    norm_layer=norm_layer,
                    **factory_kwargs
                )
            )
        else:
            dw_norm_layer = norm_layer if norm_layer is not None else nn.Identity
            dw_act_layer = activation_layer if activation_layer is not None else nn.Identity
            dw_bias = norm_layer is None

            layers.append(
                ConvNormActivation( 
                    in_planes=hidden_dim,
                    out_planes=hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=dw_norm_layer,
                    activation_layer=dw_act_layer,
                    conv_layer=nn.Conv2d,
                    bias=dw_bias,
                    affine=factory_kwargs.get('affine', True) 
                )
            )


        layers.append(
            conv_layer_factory(
                in_planes=hidden_dim,
                out_planes=output_dim,
                kernel_size=1,
                stride=1,
                activation_layer=None,
                norm_layer=norm_layer,
                **factory_kwargs
            )
        )

        self.conv = nn.Sequential(*layers)
        self.out_channels = output_dim
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2KAN(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        dropout: float = 0.2,
        input_channels: int = 3,
        arch: str = "default",
        conv_type: str = 'kanconv',
        kan_conv: Optional[str] = "KAN",
        kan_classifier: Optional[str] = "KAN",
        classifier_type: str = 'Linear',
        groups: int = 1, 
        degree: int = 3,
        spline_order: int = 3,
        grid_size: int = 5,
        base_activation: Optional[Callable[..., nn.Module]] = nn.SiLU, 
        grid_range: List = [-1, 1],
        l1_decay: float = 0.0,
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
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
    
        initial_stride: int

        if arch == "default":
            initial_stride = 2
        elif arch == "small":
            initial_stride = 1
        elif arch == "kan_small":
            initial_stride = 1
            inverted_residual_setting = [
                [1, 16, 1, 1],
                [6, 24, 1, 2],
                [6, 32, 1, 2],
                [6, 48, 1, 2],
                [6, 64, 1, 1],
                [6, 96, 1, 2],
                [6, 160, 1, 1],
            ]
        print(inverted_residual_setting)
        block = InvertedResidual
        activation_layer = nn.ReLU6 
        
        conv_layer_factory: Callable[..., nn.Module]
        if kan_norm_layer is None:
            kan_norm_layer = norm_layer 

        factory_kwargs = {
            "spline_order": spline_order,
            "grid_size": grid_size,
            "base_activation": base_activation,
            "grid_range": grid_range,
            "l1_decay": l1_decay,
            "dropout": kwargs.get('conv_dropout', 0.0),
            "degree": degree,
            "affine": affine,
            **kwargs
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
                "dropout": kwargs.get('conv_dropout', 0.0),
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
                in_planes: int,
                out_planes: int,
                kernel_size: Union[int, Tuple[int, int]],
                stride: Union[int, Tuple[int, int]] = 1,
                padding: Optional[Union[int, Tuple[int, int]]] = None,
                groups: int = 1,
                dilation: Union[int, Tuple[int, int]] = 1,
                norm_layer: Optional[Callable[..., nn.Module]] = norm_layer,
                activation_layer: Optional[Callable[..., nn.Module]] = activation_layer,
                affine: bool = affine,
                 spline_order=None, grid_size=None, base_activation=None, grid_range=None, l1_decay=None, dropout=None, degree=None, **other_kwargs
            ) -> nn.Sequential:
                if padding is None:
                     if isinstance(kernel_size, int): 
                        padding = _calculate_same_padding(kernel_size, dilation)
                     elif isinstance(kernel_size, tuple):
                        padding = _calculate_same_padding(kernel_size, (dilation, dilation))
                        
                bias = norm_layer is None or not affine
                layers = [
                    nn.Conv2d(in_channels=in_planes,
                         out_channels=out_planes,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias)
                ]
                if norm_layer is not None:
                    layers.append(norm_layer(out_planes, affine=affine))
                if activation_layer is not None:
                    layers.append(activation_layer(inplace=True))
                return nn.Sequential(*layers)
            
            conv_layer_factory = std_conv_norm_act_wrapper


        input_channel = 32 
        last_channel = 1280 

        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError(f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}")

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [
             conv_layer_factory(
                 in_planes=input_channels,
                 out_planes=input_channel,
                 kernel_size=3,
                 stride=initial_stride,
                 activation_layer=activation_layer,
                 norm_layer=norm_layer,
                 **factory_kwargs
             )
        ]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t,
                                      norm_layer=norm_layer,
                                      activation_layer=activation_layer,
                                      conv_layer_factory=conv_layer_factory,
                                      replace_depthwise=replace_depthwise,
                                      **factory_kwargs)) 
                input_channel = output_channel
        
        features.append(
            conv_layer_factory(
                in_planes=input_channel,
                out_planes=self.last_channel,
                kernel_size=1,
                activation_layer=activation_layer,
                norm_layer=norm_layer,
                **factory_kwargs
            )
        )
        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dimension = self.last_channel

        classifier_factory: Callable[..., nn.Module]
        classifier_spline_order = classifier_spline_order if classifier_spline_order is not None else spline_order
        classifier_grid_size = classifier_grid_size if classifier_grid_size is not None else grid_size
        classifier_base_activation = classifier_base_activation if classifier_base_activation is not None else base_activation
        classifier_grid_range = classifier_grid_range if classifier_grid_range is not None else grid_range
        classifier_l1_decay = classifier_l1_decay if classifier_l1_decay is not None else l1_decay
        classifier_dropout = classifier_dropout if classifier_dropout is not None else dropout 
        classifier_degree = classifier_degree if classifier_degree is not None else degree     

        if classifier_type == 'KAN':
            _kan_classifier_type = kan_classifier if kan_classifier in MLP_KAN_FACTORY else 'KAN'
            classifier_kan_func = MLP_KAN_FACTORY[_kan_classifier_type]
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
            
            classifier_factory = partial(
                 classifier_kan_func, **valid_classifier_args
            )
        else:
            def linear_factory(layers_hidden: List[int]) -> nn.Linear:
                return nn.Linear(layers_hidden[0], layers_hidden[1])
            classifier_factory = linear_factory

        self.classifier = nn.Sequential()
        self.classifier.add_module("flatten", nn.Flatten(1)) 
        self.classifier.add_module("head_dropout", nn.Dropout(p=classifier_dropout))
        self.classifier.add_module("fc", classifier_factory(layers_hidden=[feat_dimension, num_classes]))


        self._initialize_weights()

        kan_conv_suffix = f"_{kan_conv.upper()}" if conv_type == 'kanconv' else "_CONV"
        head_suffix = classifier_type
        if classifier_type in MLP_KAN_FACTORY:
            kanclassifier_name = kan_classifier if kan_classifier else 'KAN'
            head_suffix += f"_{kanclassifier_name.upper()}"
        replace_suffix = "_RDW" if replace_depthwise and conv_type=='kanconv' else ""
        arch_suffix = f"_{arch}"
        self.name = f"MobileNetV2KAN_{head_suffix}{kan_conv_suffix}{replace_suffix}{arch_suffix}"


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
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        # x = x.mean([2, 3])
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def mobilenet_v2_kan(
        num_classes: int = 1000,
        width_mult: float = 1.0,
        input_channels: int = 3,
        dropout: float = 0.2,
        conv_type: str = 'kanconv',
        arch: str = "default",
        kan_conv: Optional[str] = "KAN",
        kan_classifier: Optional[str] = "KAN",
        classifier_type: str = 'Linear',
        groups: int = 1,
        spline_order: int = 3,
        grid_size: int = 5,
        base_activation: Optional[Callable[..., nn.Module]] = nn.SiLU,
        grid_range: List = [-1, 1],
        l1_decay: float = 0.0,
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
        degree: Optional[int] = 3,
        **kwargs: Any
    ) -> MobileNetV2KAN:
    model = MobileNetV2KAN(
        num_classes=num_classes,
        width_mult=width_mult,
        input_channels=input_channels,
        dropout=dropout,
        conv_type=conv_type,
        arch=arch,
        kan_conv=kan_conv,
        kan_classifier=kan_classifier,
        classifier_type=classifier_type,
        groups=groups,
        spline_order=spline_order,
        grid_size=grid_size,
        base_activation=base_activation,
        grid_range=grid_range,
        l1_decay=l1_decay,
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
        degree=degree,
        **kwargs,
    )
    return model
