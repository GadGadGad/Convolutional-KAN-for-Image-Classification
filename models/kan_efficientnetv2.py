import math
import copy
from functools import partial
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from inspect import signature
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops.misc import SqueezeExcitation 

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

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = None,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., torch.nn.Module] = torch.nn.Conv2d,
    ) -> None:

        if padding is None:
             if isinstance(kernel_size, int) and isinstance(dilation, int):
                 padding = (kernel_size - 1) // 2 * dilation
             else:
                 _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else 2
                 kernel_size = nn.modules.utils._pair(kernel_size)
                 dilation = nn.modules.utils._pair(dilation)
                 padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
        if bias is None:
            bias = norm_layer is None

        if inplace is None:
             inplace = activation_layer is not None and activation_layer is nn.ReLU

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
             params = {} if inplace is None else {"inplace": inplace}
             layers.append(activation_layer(**params))
        super().__init__(*layers)
        self.out_channels = out_channels


class MBConvConfigV2:
    def __init__(
        self,
        block_type: str,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        se_ratio: Optional[float] = None,
    ) -> None:
        self.block_type = block_type
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


class FusedMBConv(nn.Module):
    def __init__(
        self,
        config: MBConvConfigV2,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        conv_layer_factory: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()


        self.use_res_connect = config.stride == 1 and config.input_channels == config.out_channels

        layers: List[nn.Module] = []
        activation = activation_layer

        expanded_channels = config.adjust_channels(config.input_channels, config.expand_ratio)

        if expanded_channels != config.input_channels:
            layers.append(
                conv_layer_factory(
                    in_planes=config.input_channels,
                    out_planes=expanded_channels,
                    kernel_size=config.kernel,
                    stride=config.stride,
                    norm_layer=norm_layer,
                    base_activation=activation,
                )
            )
            layers.append(
                conv_layer_factory(
                    in_planes=expanded_channels,
                    out_planes=config.out_channels,
                    kernel_size=1,
                    stride=1,
                    norm_layer=norm_layer,
                    base_activation=None,
                )
            )
        else:
             layers.append(
                 conv_layer_factory(
                     in_planes=config.input_channels,
                     out_planes=config.out_channels,
                     kernel_size=config.kernel,
                     stride=config.stride,
                     norm_layer=norm_layer,
                     base_activation=activation,
                 )
             )


        self.block = nn.Sequential(*layers)
        self.stochastic_depth = DropPath(stochastic_depth_prob, scale_by_keep=True)
        self.out_channels = config.out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result

class MBConv(nn.Module):
    def __init__(
        self,
        config: MBConvConfigV2,
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
                in_planes=config.input_channels,
                out_planes=expanded_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                base_activation=activation
            ))

        kernel_size = config.kernel
        stride = config.stride
        if self.replace_depthwise:
             layers.append(conv_layer_factory(
                 in_planes=expanded_channels,
                 out_planes=expanded_channels,
                 kernel_size=kernel_size,
                 stride=stride,
                 groups=expanded_channels,
                 norm_layer=norm_layer,
                 base_activation=activation
             ))
        else:
            dw_conv = ConvNormActivation(
                 expanded_channels,
                 expanded_channels,
                 kernel_size=kernel_size,
                 stride=stride,
                 groups=expanded_channels,
                 conv_layer=nn.Conv2d,
                 norm_layer=norm_layer,
                 activation_layer=activation,
                 padding=None
            )
            layers.append(dw_conv)

        if config.se_ratio is not None and config.se_ratio > 0.0:
             squeeze_channels = max(1, int(config.input_channels * config.se_ratio))
             layers.append(se_layer(expanded_channels, squeeze_channels, activation=activation))

        layers.append(conv_layer_factory(
            in_planes=expanded_channels,
            out_planes=config.out_channels,
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


class EfficientNetV2KAN(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        inverted_residual_setting: List[MBConvConfigV2],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        in_channels: int = 3,
        last_channel: Optional[int] = None,
        stem_stride: int = 2,
        conv_type: str = 'kanconv',
        conv_dropout: float = 0.0,
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
        width_scale: float = 1.0,
        affine: bool = False,
        kan_norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
        replace_depthwise: bool = False,
        classifier_spline_order: Optional[int] = None,
        classifier_grid_size: Optional[int] = None,
        classifier_base_activation: Optional[Callable[..., nn.Module]] = nn.SiLU,
        classifier_grid_range: Optional[List] = None,
        classifier_l1_decay: Optional[float] = None,
        classifier_dropout: Optional[float] = None,
        classifier_degree: Optional[int] = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if kan_norm_layer is None:
             kan_norm_layer = norm_layer
        activation_layer = base_activation if base_activation else nn.ReLU 

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if kan_norm_layer is None:
             kan_norm_layer = norm_layer

        if conv_type == 'kanconv':
            if kan_conv is None or kan_conv not in CONV_KAN_FACTORY:
                 kan_conv = "KAN"
            kan_conv_func = CONV_KAN_FACTORY[kan_conv]

            conv_layer_factory = partial(
                 kan_conv_func,
                 spline_order=spline_order,
                 grid_size=grid_size,
                 base_activation=activation_layer,
                 grid_range=grid_range,
                 dropout=conv_dropout,
                 l1_decay=l1_decay,
                 groups=groups,
                 norm_layer=kan_norm_layer,
                 affine=affine,
                 **kwargs
            )

        elif conv_type == 'conv':
             def std_conv_wrapper(in_planes, out_planes, kernel_size, stride=1, groups=1, norm_layer=None, base_activation=None, **std_kwargs):
                 _padding = None
                 if isinstance(kernel_size, int):
                      _padding = _calculate_same_padding(kernel_size, std_kwargs.get('dilation', 1)) if kernel_size > 1 else 0
                 elif isinstance(kernel_size, tuple):
                       _padding = _calculate_same_padding(kernel_size, std_kwargs.get('dilation', (1,1)))

                 return ConvNormActivation(
                     in_channels=in_planes,
                     out_channels=out_planes,
                     kernel_size=kernel_size,
                     stride=stride,
                     padding=_padding,
                     groups=groups,
                     conv_layer=nn.Conv2d,
                     norm_layer=norm_layer,
                     activation_layer=activation_layer,
                     bias=not affine if norm_layer is not None else True
                 )
             conv_layer_factory = partial(std_conv_wrapper, norm_layer=norm_layer, base_activation=base_activation)

        firstconv_output_channels = inverted_residual_setting[0].input_channels
        self.stem = conv_layer_factory(
            in_planes=in_channels,
            out_planes=firstconv_output_channels,
            kernel_size=3,
            stride=stem_stride,
            norm_layer=norm_layer,
            base_activation=activation_layer
        )

        total_stage_blocks = sum(c.num_layers for c in inverted_residual_setting)
        stage_block_id = 0
        self.blocks = nn.Sequential()
        for i, config in enumerate(inverted_residual_setting):
            stage: List[nn.Module] = []
            num_layers = config.num_layers
            for j in range(num_layers):
                 sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                 block_input_channels = config.input_channels if j == 0 else config.out_channels
                 block_stride = config.stride if j == 0 else 1

                 current_block_config = copy.deepcopy(config)
                 current_block_config.input_channels = block_input_channels
                 current_block_config.stride = block_stride

                 if config.block_type == 'fused':
                     stage.append(
                         FusedMBConv(
                             current_block_config,
                             sd_prob,
                             norm_layer,
                             conv_layer_factory=conv_layer_factory,
                             activation_layer=activation_layer,
                         )
                     )
                 elif config.block_type == 'mbconv':
                     stage.append(
                         MBConv(
                             current_block_config,
                             sd_prob,
                             norm_layer,
                             se_layer=SqueezeExcitation,
                             conv_layer_factory=conv_layer_factory,
                             replace_depthwise=replace_depthwise,
                             activation_layer=activation_layer,
                         )
                     )

                 stage_block_id += 1

            self.blocks.add_module(f"stage_{i}", nn.Sequential(*stage))

        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else _make_divisible(1280 * width_scale, 8)
        self.head_conv = conv_layer_factory(
            in_planes=lastconv_input_channels,
            out_planes=lastconv_output_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            base_activation=activation_layer
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        feat_dimension = lastconv_output_channels
        _kan_classifier_type = kan_classifier if kan_classifier else "KAN"

        classifier_kan_classifier = MLP_KAN_FACTORY[_kan_classifier_type]

        classifier_spline_order = classifier_spline_order if classifier_spline_order is not None else spline_order
        classifier_grid_size = classifier_grid_size if classifier_grid_size is not None else grid_size
        classifier_base_activation = classifier_base_activation if classifier_base_activation is not None else nn.SiLU
        classifier_grid_range = classifier_grid_range if classifier_grid_range is not None else grid_range
        classifier_l1_decay = classifier_l1_decay if classifier_l1_decay is not None else l1_decay
        classifier_degree = classifier_degree if classifier_degree is not None else degree

        classifier_kan_args = {
            "spline_order": classifier_spline_order,
            "grid_size": classifier_grid_size,
            "base_activation": classifier_base_activation,
            "grid_range": classifier_grid_range,
            "l1_decay": classifier_l1_decay,
            "degree": classifier_degree,
            "dropout": 0.0,
            "first_dropout": False,
            "bias": False 
        }
        classifier_kan_sig = signature(classifier_kan_classifier)
        valid_classifier_args = {k: v for k, v in classifier_kan_args.items() if k in classifier_kan_sig.parameters}
        valid_extra_classifier_args = {k: v for k, v in kwargs.items() if k.startswith('classifier_') and k[len('classifier_'):] in classifier_kan_sig.parameters}
        valid_classifier_args.update({k[len('classifier_'):]: v for k, v in valid_extra_classifier_args.items()})


        classifier_factory = partial(classifier_kan_classifier, **valid_classifier_args)
            
        self.classifier = nn.Sequential()
        self.classifier.add_module("flatten", nn.Flatten(1))

        classifier_dropout = classifier_dropout if classifier_dropout is not None else dropout
        if classifier_dropout > 0.0:
            self.classifier.add_module("head_dropout", nn.Dropout(p=classifier_dropout, inplace=True))

        if classifier_type == 'Linear':
            self.classifier.add_module("fc", nn.Linear(feat_dimension, num_classes))
        elif classifier_type == 'KAN': 
            self.classifier.add_module("kan_fc", classifier_factory(layers_hidden=[feat_dimension, num_classes],**valid_classifier_args))

        elif classifier_type == 'HiddenKAN': 
            hidden_dim = kwargs.get('head_hidden_dim', 1024) 
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
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
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
        x = self.classifier(x)
        return x


def _efficientnetv2_conf(
    arch: str = "s", width_mult: float = 1.0, depth_mult: float = 1.0
) -> List[MBConvConfigV2]:
    bneck_conf = partial(MBConvConfigV2, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting: List[MBConvConfigV2]

    if arch == "s":
        inverted_residual_setting = [
            bneck_conf("fused", 1, 3, 1, 24,  24,  2),
            bneck_conf("fused", 4, 3, 2, 24,  48,  4),
            bneck_conf("fused", 4, 3, 2, 48,  64,  4),
            bneck_conf("mbconv", 4, 3, 2, 64,  128, 6, se_ratio=0.25),
            bneck_conf("mbconv", 6, 3, 1, 128, 160, 9, se_ratio=0.25),
            bneck_conf("mbconv", 6, 3, 2, 160, 256, 15, se_ratio=0.25),
        ]
    elif arch == "m":
        inverted_residual_setting = [
             bneck_conf("fused", 1, 3, 1, 24,  24,  3),
             bneck_conf("fused", 4, 3, 2, 24,  48,  5),
             bneck_conf("fused", 4, 3, 2, 48,  80,  5),
             bneck_conf("mbconv", 4, 3, 2, 80,  160, 7, se_ratio=0.25),
             bneck_conf("mbconv", 6, 3, 1, 160, 176, 14, se_ratio=0.25),
             bneck_conf("mbconv", 6, 3, 2, 176, 304, 18, se_ratio=0.25),
             bneck_conf("mbconv", 6, 3, 1, 304, 512, 5, se_ratio=0.25),
        ]
    elif arch == "l":
        inverted_residual_setting = [
            bneck_conf("fused", 1, 3, 1, 32,  32,  4),
            bneck_conf("fused", 4, 3, 2, 32,  64,  7),
            bneck_conf("fused", 4, 3, 2, 64,  96,  7),
            bneck_conf("mbconv", 4, 3, 2, 96,  192, 10, se_ratio=0.25),
            bneck_conf("mbconv", 6, 3, 1, 192, 224, 19, se_ratio=0.25),
            bneck_conf("mbconv", 6, 3, 2, 224, 384, 25, se_ratio=0.25),
            bneck_conf("mbconv", 6, 3, 1, 384, 640, 7, se_ratio=0.25),
        ]

    return inverted_residual_setting


def efficientnetv2_kan(
        arch: str = "s",
        num_classes: int = 1000,
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
        in_channels: int=3,
        replace_depthwise: bool = False,
        conv_dropout: float = 0.0,
        classifier_spline_order: Optional[int] = None,
        classifier_grid_size: Optional[int] = None,
        classifier_base_activation: Optional[Callable[..., nn.Module]] = None,
        classifier_grid_range: Optional[List] = None,
        classifier_l1_decay: Optional[float] = None,
        classifier_dropout: Optional[float] = None,
        **kwargs: Any
    ) -> EfficientNetV2KAN:

    width_mult = 1.0
    depth_mult = 1.0

    inverted_residual_setting = _efficientnetv2_conf(arch, width_mult, depth_mult)

    if arch == "s":
        last_channel = 1280
    elif arch == "m":
        last_channel = 1280
    elif arch == "l":
        last_channel = 1280
    else:
        last_channel = _make_divisible(1280 * width_mult, 8)

    if classifier_dropout is None:
        classifier_dropout = dropout

    model = EfficientNetV2KAN(
        inverted_residual_setting=inverted_residual_setting,
        dropout=dropout,
        stochastic_depth_prob=stochastic_depth_prob,
        num_classes=num_classes,
        norm_layer=norm_layer,
        in_channels=in_channels,
        last_channel=last_channel,
        stem_stride=2,
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
        conv_dropout=conv_dropout,
        **kwargs,
    )

    kan_conv_suffix = f"_{kan_conv.upper()}" if conv_type == 'kanconv' else "_CONV"
    head_kan_suffix = ""
    if classifier_type in MLP_KAN_FACTORY or classifier_type == 'HiddenKAN':
        head_kan_suffix = f"_{kan_classifier.upper()}" if kan_classifier else "_KAN"

    model.name = f"EfficientNetV2{arch.upper()}-KAN_{classifier_type}{head_kan_suffix}{kan_conv_suffix}"

    return model
def _efficientnetv2_conf_small(
    arch: str = "tiny", width_mult: float = 1.0, depth_mult: float = 1.0
) -> tuple[List[MBConvConfigV2], Optional[int]]:
    bneck_conf = partial(MBConvConfigV2, width_mult=width_mult, depth_mult=depth_mult)
    inverted_residual_setting: List[MBConvConfigV2]
    last_channel: Optional[int] = None

    if arch == "tiny":
        inverted_residual_setting = [
            bneck_conf("fused", 1, 3, 1, 16,  16,  1),
            bneck_conf("fused", 4, 3, 2, 16,  24,  2),
            bneck_conf("fused", 4, 3, 2, 24,  40,  2),
            bneck_conf("mbconv", 4, 3, 2, 40,  80,  2, se_ratio=0.25),
            bneck_conf("mbconv", 6, 3, 1, 80, 112,  2, se_ratio=0.25),
        ]
        last_channel = _make_divisible(256 * width_mult, 8)
    elif arch == "kan_tiny":
        inverted_residual_setting = [
            bneck_conf("fused", 1, 3, 1, 16,  16,  1),
            bneck_conf("fused", 4, 3, 2, 16,  24,  1),
            bneck_conf("fused", 4, 3, 2, 24,  40,  1),
            bneck_conf("mbconv", 4, 3, 2, 40,  80,  1, se_ratio=0.25),
            bneck_conf("mbconv", 6, 3, 1, 80, 112,  1, se_ratio=0.25),
        ]
        last_channel = _make_divisible(256 * width_mult, 8)
    
    return inverted_residual_setting, last_channel

def efficientnetv2_kan_small(
        arch: str = "kan_tiny",
        num_classes: int = 10,
        in_channels: int = 3,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
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
        conv_dropout: float = 0.0,
        classifier_spline_order: Optional[int] = None,
        classifier_grid_size: Optional[int] = None,
        classifier_base_activation: Optional[Callable[..., nn.Module]] = None,
        classifier_grid_range: Optional[List] = None,
        classifier_l1_decay: Optional[float] = None,
        classifier_dropout: Optional[float] = None,
        **kwargs: Any
    ) -> EfficientNetV2KAN:

    inverted_residual_setting, last_channel_suggestion = _efficientnetv2_conf_small(
        arch, width_mult=width_mult, depth_mult=depth_mult
    )

    last_channel = kwargs.pop("last_channel", last_channel_suggestion)
    if last_channel is None:
        last_channel = _make_divisible(512 * width_mult, 8)

    stem_stride = 1 if arch == "kan_tiny" else 2
    first_conv_out_channels = inverted_residual_setting[0].input_channels

    if classifier_dropout is None:
        classifier_dropout = dropout
    model = EfficientNetV2KAN(
        inverted_residual_setting=inverted_residual_setting,
        dropout=dropout,
        stochastic_depth_prob=stochastic_depth_prob,
        num_classes=num_classes,
        norm_layer=norm_layer,
        in_channels=in_channels,
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
        conv_dropout=conv_dropout,
        **kwargs,
    )

    if arch == "kan_tiny" and stem_stride == 1:
        if hasattr(model, 'stem') and isinstance(model.stem, nn.Sequential) and len(model.stem) > 0:
            old_conv = model.stem[0]

    kan_conv_suffix = f"_{kan_conv.upper()}" if conv_type == 'kanconv' else "_CONV"
    head_kan_suffix = ""
    if classifier_type == 'HiddenKAN':
        head_kan_suffix = f"_{kan_classifier.upper()}" if kan_classifier else "_KAN"

    model.name = f"EfficientNetV2Small-{arch.upper()}-KAN_{classifier_type}{head_kan_suffix}{kan_conv_suffix}"

    return model