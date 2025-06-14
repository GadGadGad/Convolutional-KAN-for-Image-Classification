import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops.misc import SqueezeExcitation
from inspect import signature
from huggingface_hub import PyTorchModelHubMixin

from layers.kan_conv import CONV_KAN_FACTORY, _calculate_same_padding, conv
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

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman/pytorch-image-models library.
    License: Apache 2.0 -> https://github.com/rwightman/pytorch-image-models/blob/master/LICENSE
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob,3):0.3f}"
class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
                in_channels, out_channels, kernel_size, stride, padding,
                dilation=dilation, groups=groups, bias=bias
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels, affine=affine))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))

        super().__init__(*layers)
        self.out_channels = out_channels
class MBConvConfig:
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float,
        depth_mult: float,
        se_ratio: float = 0.25, 
    ) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)
        self.se_ratio = se_ratio

    def adjust_channels(self, channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)

    def adjust_depth(self, num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))

class MBConv(nn.Module):
    def __init__(
        self,
        config: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
        conv_layer_factory: Callable[..., nn.Module] = None,
        replace_depthwise: bool = False,
        activation_layer: Callable[..., nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()


        self.use_res_connect = config.stride == 1 and config.input_channels == config.out_channels
        self.replace_depthwise = replace_depthwise

        layers: List[nn.Module] = []
        activation = activation_layer 

        expanded_channels = config.adjust_channels(config.input_channels, config.expand_ratio)
        if expanded_channels != config.input_channels:

            layers.append(conv_layer_factory(
                in_channels=config.input_channels,
                out_channels=expanded_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                base_activation=activation
            ))


        kernel_size = config.kernel
        stride = config.stride
        if self.replace_depthwise:
             layers.append(conv_layer_factory(
                 in_channels=expanded_channels,
                 out_channels=expanded_channels,
                 kernel_size=kernel_size,
                 stride=stride,
                 groups=expanded_channels, 
                 norm_layer=norm_layer,
                 base_activation=activation 
             ))
        else:
            dw_conv = ConvNormActivation(
                 in_channels=expanded_channels,
                 out_channels=expanded_channels,
                 kernel_size=kernel_size,
                 stride=stride,
                 groups=expanded_channels,
                 conv_layer=conv,
                 norm_layer=norm_layer,
                 activation_layer=activation,
                 padding=None 
            )
            layers.append(dw_conv)


        if config.se_ratio > 0.0:
             squeeze_channels = max(1, int(config.input_channels * config.se_ratio))
             layers.append(se_layer(expanded_channels, squeeze_channels, activation=activation))

        layers.append(conv_layer_factory(
            in_channels=expanded_channels,
            out_channels=config.out_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            base_activation=None 
        ))


        self.block = nn.Sequential(*layers)
        self.stochastic_depth = DropPath(stochastic_depth_prob, scale_by_keep=True)
        self.out_channels = config.out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNetKAN(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        in_channels: int,
        inverted_residual_setting: List[MBConvConfig],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
        conv_type: str = 'kanconv',
        kan_conv: Optional[str] = "KAN",
        kan_classifier: Optional[str] = "KAN",
        classifier_type: str = 'Linear',
        groups: int = 1,
        spline_order: int = 3,
        grid_size: int = 5,
        base_activation: Optional[Callable[..., nn.Module]] = nn.SiLU,
        grid_range: List = [-1, 1],
        l1_decay: float = 0.0,
        degree: int = 3,
        width_scale: float = 1.0,
        affine: bool = False, 
        kan_norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
        replace_depthwise: bool = False,
        classifier_spline_order: Optional[int] = None,
        classifier_grid_size: Optional[int] = None,
        classifier_base_activation: Optional[Callable[..., nn.Module]] = None,
        classifier_grid_range: Optional[List] = None,
        classifier_l1_decay: Optional[float] = None,
        classifier_dropout: Optional[float] = None, 
        classifier_degree: Optional[int] = 3,
        conv_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if kan_norm_layer is None:
             kan_norm_layer = norm_layer
        activation_layer = base_activation if base_activation else nn.ReLU 

        if conv_type == 'kanconv':
            if kan_conv is None or kan_conv not in CONV_KAN_FACTORY:
                 kan_conv = "KAN"
            kan_conv_func = CONV_KAN_FACTORY[kan_conv]
            conv_layer_factory: Callable[..., nn.Module]
            factory_kwargs = {
                "spline_order": spline_order,
                "grid_size": grid_size,
                "base_activation": base_activation,
                "grid_range": grid_range,
                "l1_decay": l1_decay,
                "dropout": conv_dropout, 
                "degree": degree,
                "affine": affine,
                **{k: v for k, v in kwargs.items() if k in signature(CONV_KAN_FACTORY.get(kan_conv, conv)).parameters}
            }
            
            conv_layer_factory = partial(
                 kan_conv_func,
                 spline_order=spline_order,
                 grid_size=grid_size,
                 base_activation=activation_layer,
                 grid_range=grid_range,
                 dropout=kwargs.get('conv_dropout', 0.0),
                 l1_decay=l1_decay,
                 groups=groups,
                 norm_layer=kan_norm_layer,
                 affine=affine,
                 **factory_kwargs 
            )
        elif conv_type == 'conv':
             conv_layer_factory = partial(
                 ConvNormActivation,
                 norm_layer=norm_layer,
                 activation_layer=activation_layer, 
                 bias=not affine if norm_layer is not None else True, 
             )
             def std_conv_wrapper(in_channels: int,
                                  out_channels: int,
                                  kernel_size: Union[int, Tuple[int, int]],
                                  stride=1,
                                  groups=1,
                                  norm_layer: Optional[Callable[..., nn.Module]] = norm_layer,
                                  activation_layer: Optional[Callable[..., nn.Module]] = activation_layer,
                                  **std_kwargs):
                if padding is None:
                    if isinstance(kernel_size, int):
                        padding = _calculate_same_padding(kernel_size, std_kwargs.get('dilation', 1)) if kernel_size > 1 else 0
                    elif padding(kernel_size, tuple):
                        padding = _calculate_same_padding(kernel_size, std_kwargs.get('dilation', (1,1)))

                return ConvNormActivation(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    conv_layer=conv,
                    bias=not affine if norm_layer is not None else True,
                    inplace=True if base_activation is nn.ReLU else False
                )
             conv_layer_factory = partial(std_conv_wrapper, norm_layer=norm_layer, base_activation=base_activation)

        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.stem = conv_layer_factory(
            in_channels=in_channels,
            out_channels=firstconv_output_channels,
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer,
            base_activation=base_activation
        )

        total_stage_blocks = sum(c.num_layers for c in inverted_residual_setting)
        stage_block_id = 0
        self.blocks = nn.Sequential()
        for i, config in enumerate(inverted_residual_setting):
            stage: List[nn.Module] = []
            num_layers = config.num_layers
            for j in range(num_layers):
                 sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                 stage.append(
                     MBConv(
                         config,
                         sd_prob,
                         norm_layer,
                         conv_layer_factory=conv_layer_factory,
                         replace_depthwise=replace_depthwise,
                         activation_layer=base_activation,
                     )
                 )
                 stage_block_id += 1
                 config.input_channels = config.out_channels
                 config.stride = 1

            self.blocks.add_module(f"stage_{i}", nn.Sequential(*stage))


        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        self.head_conv = conv_layer_factory(
            in_channels=lastconv_input_channels,
            out_channels=lastconv_output_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            base_activation=base_activation
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        classifier_factory: Callable[..., nn.Module]
        classifier_spline_order = classifier_spline_order if classifier_spline_order is not None else spline_order
        classifier_grid_size = classifier_grid_size if classifier_grid_size is not None else grid_size
        classifier_base_activation = classifier_base_activation if classifier_base_activation is not None else base_activation
        classifier_grid_range = classifier_grid_range if classifier_grid_range is not None else grid_range
        classifier_l1_decay = classifier_l1_decay if classifier_l1_decay is not None else l1_decay
        classifier_dropout = classifier_dropout if classifier_dropout is not None else dropout 
        classifier_degree = classifier_degree if classifier_degree is not None else degree 
        classifier_factory = MLP_KAN_FACTORY[kan_classifier]
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
        classifier_kan_sig = signature(classifier_factory)
        valid_classifier_args = {k: v for k, v in classifier_kan_args.items() if k in classifier_kan_sig.parameters}

        feat_dimension = lastconv_output_channels
        self.classifier = nn.Sequential()
        self.classifier.add_module("flatten", nn.Flatten(1))

        if classifier_dropout > 0.0: 
            self.classifier.add_module("head_dropout", nn.Dropout(p=classifier_dropout, inplace=True))
        
        if classifier_type == 'Linear':
            self.classifier.add_module("fc", nn.Linear(feat_dimension, num_classes))
        elif classifier_type == 'KAN': 
            if kan_classifier is None or kan_classifier not in MLP_KAN_FACTORY:
                classifier_factory = MLP_KAN_FACTORY[kan_classifier]
            self.classifier.add_module("kan_fc", classifier_factory(**valid_classifier_args))

        elif classifier_type == 'HiddenKAN': 
            hidden_dim = kwargs.get('head_hidden_dim', 1024) 
            if kan_classifier is None or kan_classifier not in MLP_KAN_FACTORY:
                kan_classifier = "KAN"

            self.classifier.add_module("kan_fc1", classifier_factory(layers_hidden=[feat_dimension, hidden_dim], **valid_classifier_args))
            self.classifier.add_module("fc2", nn.Linear(hidden_dim, num_classes))

        else:
            self.classifier = nn.Identity() 


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                 if conv_type == 'conv':
                     nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu") 
                     if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



def _efficientnet_conf(
    width_mult: float, depth_mult: float, **kwargs: Any
) -> List[MBConvConfig]:
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32, 16, 1, se_ratio=0.25),
        bneck_conf(6, 3, 2, 16, 24, 2, se_ratio=0.25),
        bneck_conf(6, 5, 2, 24, 40, 2, se_ratio=0.25),
        bneck_conf(6, 3, 2, 40, 80, 3, se_ratio=0.25),
        bneck_conf(6, 5, 1, 80, 112, 3, se_ratio=0.25),
        bneck_conf(6, 5, 2, 112, 192, 4, se_ratio=0.25),
        bneck_conf(6, 3, 1, 192, 320, 1, se_ratio=0.25),
    ]
    return inverted_residual_setting



def efficientnet_kan(
        arch: str = "b0", 
        in_channels: int = 3,
        num_classes: int = 1000,
        stem_stride:int = 2,
        conv_type: str = 'kanconv',
        kan_conv: Optional[str] = "KAN",
        kan_classifier: Optional[str] = "KAN",
        classifier_type: str = 'Linear',
        groups: int = 1,
        spline_order: int = 3,
        grid_size: int = 5,
        base_activation: Optional[Callable[..., nn.Module]] = nn.SiLU,
        grid_range: List = [-1, 1],
        l1_decay: float = 0.0,
        dropout: float = 0.2, 
        stochastic_depth_prob: float = 0.2, 
        affine: bool = True, 
        kan_norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        replace_depthwise: bool = False,
        classifier_spline_order: Optional[int] = None,
        classifier_grid_size: Optional[int] = None,
        classifier_base_activation: Optional[Callable[..., nn.Module]] = None,
        classifier_grid_range: Optional[List] = None,
        classifier_l1_decay: Optional[float] = None,
        **kwargs: Any
    ) -> EfficientNetKAN:

    if arch == "b0":
        width_mult = 1.0
        depth_mult = 1.0
        dropout = 0.2
    elif arch == "b1":
        width_mult = 1.0
        depth_mult = 1.1
        dropout = 0.2
    elif arch == "b2":
        width_mult = 1.1
        depth_mult = 1.2
        dropout = 0.3

    inverted_residual_setting = _efficientnet_conf(width_mult, depth_mult)

    last_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280

    classifier_dropout = dropout

    return EfficientNetKAN(
        in_channels = in_channels,
        inverted_residual_setting=inverted_residual_setting,
        dropout=dropout, 
        stochastic_depth_prob=stochastic_depth_prob,
        num_classes=num_classes,
        norm_layer=norm_layer,
        last_channel=last_channel,
        stem_stride=stem_stride,
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
        width_scale=width_mult, 
        affine=affine,
        kan_norm_layer=kan_norm_layer,
        replace_depthwise=replace_depthwise,
        classifier_spline_order=classifier_spline_order,
        classifier_grid_size=classifier_grid_size,
        classifier_base_activation=classifier_base_activation,
        classifier_grid_range=classifier_grid_range,
        classifier_l1_decay=classifier_l1_decay,
        classifier_dropout=classifier_dropout,
        **kwargs,
    )
def efficientnet_small_conf(
    width_mult: float = 0.5, depth_mult: float = 0.5, se_ratio: float = 0.25 
) -> List[MBConvConfig]:
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult, se_ratio=se_ratio)

    inverted_residual_setting = [
        bneck_conf(1, 3, 1, 32,  16,  1),  
        bneck_conf(6, 3, 2, 16,  24,  1),
        bneck_conf(6, 5, 2, 24,  40,  2),  
        bneck_conf(6, 3, 1, 40,  80,  2), 
        bneck_conf(6, 5, 2, 80, 112,  2),  
        bneck_conf(6, 5, 1, 112, 192, 3),  
        bneck_conf(6, 3, 1, 192, 320, 1), 
    ]

    return inverted_residual_setting


def efficientnet_kan_small(
        arch: int = 'b0_small',
        in_channels: int = 3,
        num_classes: int = 10, 
        width_mult: float = 0.5, 
        depth_mult: float = 0.5, 
        se_ratio: float = 0.25,  
        stem_stride: int = 1,
        conv_type: str = 'kanconv',
        kan_conv: Optional[str] = "KAN",
        kan_classifier: Optional[str] = "KAN",
        classifier_type: str = 'Linear',
        groups: int = 1,
        spline_order: int = 3,
        grid_size: int = 5,
        base_activation: Optional[Callable[..., nn.Module]] = nn.SiLU,
        grid_range: List = [-1, 1],
        l1_decay: float = 0.0,
        dropout: float = 0.1, 
        stochastic_depth_prob: float = 0.1, 
        affine: bool = True, 
        kan_norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        replace_depthwise: bool = False,
        classifier_spline_order: Optional[int] = None,
        classifier_grid_size: Optional[int] = None,
        classifier_base_activation: Optional[Callable[..., nn.Module]] = None,
        classifier_grid_range: Optional[List] = None,
        classifier_l1_decay: Optional[float] = None,
        classifier_dropout: Optional[float] = None, 
        last_channel_mult: int = 4, 
        **kwargs: Any
    ) -> EfficientNetKAN:
    if arch == "b0_small":
        width_mult = 0.35
        depth_mult = 0.35
        dropout = 0.05
        stochastic_depth_prob = 0.05
        se_ratio = 0.1
    elif arch == "b1_small":
        width_mult = 0.5
        depth_mult = 0.6
        dropout = 0.1
        stochastic_depth_prob = 0.1
        se_ratio = 0.15
    elif arch == "b2_small":
        width_mult = 0.6
        depth_mult = 0.65
        dropout = 0.15
        stochastic_depth_prob = 0.15
        se_ratio = 0.2

    inverted_residual_setting = efficientnet_small_conf(
        width_mult=width_mult, depth_mult=depth_mult, se_ratio=se_ratio
    )


    last_block_out_channels = inverted_residual_setting[-1].out_channels

    last_channel = _make_divisible(last_block_out_channels * last_channel_mult, 8)

    classifier_dropout = classifier_dropout if classifier_dropout is not None else dropout

    model = EfficientNetKAN(
        inverted_residual_setting=inverted_residual_setting,
        dropout=dropout,
        stochastic_depth_prob=stochastic_depth_prob,
        num_classes=num_classes,
        norm_layer=norm_layer,
        in_channels = in_channels,
        last_channel=last_channel,
        stem_stride=stem_stride, 
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
        width_scale=width_mult, 
        affine=affine,
        kan_norm_layer=kan_norm_layer,
        replace_depthwise=replace_depthwise,
        classifier_spline_order=classifier_spline_order,
        classifier_grid_size=classifier_grid_size,
        classifier_base_activation=classifier_base_activation,
        classifier_grid_range=classifier_grid_range,
        classifier_l1_decay=classifier_l1_decay,
        classifier_dropout=classifier_dropout,
        **kwargs,
    )

    model.name += f'_w{width_mult}_d{depth_mult}_cifar'

    return model