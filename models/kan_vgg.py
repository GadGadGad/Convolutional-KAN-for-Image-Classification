# Based on this https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html#vgg16
from functools import partial
from math import prod
from typing import cast, Dict, List, Optional, Union, Tuple, Callable, Any

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from inspect import signature

from layers.kan_conv import (
    conv, CONV_KAN_FACTORY,
)
from .kans import (
    MLP_KAN_FACTORY
)


cfgs: Dict[str, List[Union[str, int]]] = {
    "VGG16_small": [16, 16, "M", 32, 32, "M", 64, 64, 64, "M", 128, 128, 128, "M", 128, 128, 128],
    "VGG16_kansmall": [8, 8, "M", 16, 16, "M", 32, 32, 32, "M", 64, 64, 64, "M", 64, 64, 64],
    "VGG19_small": [16, 16, "M", 32, 32, "M", 64, 64, 64, 64, "M", 128, 128, 128, 128, "M", 128, 128, 128, 128],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(
            self, features: nn.ModuleList, classifier: nn.Module, expected_feature_shape: Tuple = (1, 1)
    ) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(expected_feature_shape)
        self.classifier = classifier
        self.expected_feature_shape = expected_feature_shape

    @staticmethod
    def make_layers(cfg: List[Union[str, int]],
                    conv_type: str,
                    classifier_type: str,
                    kan_conv: Optional[str] = None,
                    kan_classifier: Optional[Callable[..., nn.Module]] = None,
                    spline_order: int = 3,
                    grid_size: int = 5,
                    base_activation: Optional[Callable[..., nn.Module]] = nn.SiLU,
                    grid_range: List = [-1, 1],
                    dropout: float = 0.0,
                    l1_decay: float = 0.0,
                    groups: int = 1,
                    std_conv_kernel_size: int = 3,
                    std_conv_padding: int = 1,
                    std_conv_bias: bool = True,
                    expected_feature_shape: Tuple = (1,1),
                    num_input_features: int = 3,
                    num_classes: int = 10,
                    width_scale: int = 1,
                    classifier_dropout: float = 0.5,
                    affine: bool = False,
                    norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
                    kan_norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
                    degree: int=3,
                    conv_dropout: float = 0.0,
                    **kwargs: Any,
                    ):
        layers: List[nn.Module] = []
        in_channels = num_input_features

        _conv_fun: Callable[..., nn.Module]
        _conv_fun_first: Callable[..., nn.Module]

        if conv_type == 'kanconv':
            if kan_conv is None or kan_conv not in CONV_KAN_FACTORY:
                kan_conv = "KAN"
            kan_conv_func = CONV_KAN_FACTORY[kan_conv]
            factory_kwargs = {
                "spline_order": spline_order,
                "grid_size": grid_size,
                "base_activation": base_activation,
                "grid_range": grid_range,
                "l1_decay": l1_decay,
                "dropout": conv_dropout,
                "degree": degree,
                "affine": affine,
                "norm_layer": kan_norm_layer,
                "padding": std_conv_padding,
                "groups": groups,
                "kernel_size": std_conv_kernel_size
            }
            kan_conv_sig = signature(kan_conv_func)
            valid_factory_kwargs = {k: v for k, v in factory_kwargs.items() if k in kan_conv_sig.parameters}
            valid_extra_kwargs = {k: v for k, v in kwargs.items() if k in kan_conv_sig.parameters}
            valid_factory_kwargs.update(valid_extra_kwargs)


            _conv_fun = partial(kan_conv_func, **valid_factory_kwargs)

            _conv_fun_first_kwargs = valid_factory_kwargs.copy()
            _conv_fun_first_kwargs["dropout"] = 0.0
            _conv_fun_first = partial(kan_conv_func, **_conv_fun_first_kwargs)

        elif conv_type == 'conv':
            def create_std_conv_block(in_c, out_c, kernel_size, padding, bias, norm_layer_cls, use_affine):
                seq = [conv(in_c, out_c, kernel_size=kernel_size, padding=padding, bias=bias)]
                if norm_layer_cls is not None:
                    norm_module = norm_layer_cls(out_c, affine=use_affine) if 'affine' in signature(norm_layer_cls).parameters else norm_layer_cls(out_c)
                    seq.append(norm_module)
                seq.append(nn.ReLU(inplace=True))
                return nn.Sequential(*seq)

            bias = std_conv_bias if norm_layer is None else False

            _conv_fun = partial(create_std_conv_block, kernel_size=std_conv_kernel_size, padding=std_conv_padding,
                                bias=bias, norm_layer_cls=norm_layer, use_affine=affine)
            _conv_fun_first = _conv_fun


        for l_index, v in enumerate(cfg):
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                out_channels = v * width_scale
                if l_index == 0:
                    conv_block = _conv_fun_first(in_channels, out_channels)
                else:
                    conv_block = _conv_fun(in_channels, out_channels)
                layers.append(conv_block)
                in_channels = out_channels

        feat_dimension = in_channels * prod(expected_feature_shape)

        if classifier_type == 'KAN':
            classification = nn.Sequential(
                nn.Dropout(p=classifier_dropout),
                kan_classifier([feat_dimension, num_classes])
            )
        elif classifier_type == 'Linear':
            classification = nn.Sequential(
                nn.Dropout(p=classifier_dropout),
                nn.Linear(feat_dimension, num_classes),
            )
        elif classifier_type == 'HiddenKAN':
            classification = nn.Sequential(
                kan_classifier([feat_dimension, 1024]),
                nn.Dropout(p=classifier_dropout),
                nn.Linear(1024, num_classes)
            )
        elif classifier_type == 'VGGKAN':
             classification = nn.Sequential(
                nn.Linear(feat_dimension, 1024),
                # kan_classifier([feat_dimension, 1024]),
                nn.ReLU(True),
                nn.Dropout(p=classifier_dropout),
                nn.Linear(1024, 1024),
                # kan_classifier([1024, 1024]),
                nn.ReLU(True),
                nn.Dropout(p=classifier_dropout),
                # nn.Linear(1024, num_classes),
                kan_classifier([1024, num_classes]),
            )
        elif classifier_type == 'VGG':
            classification = nn.Sequential(
                nn.Linear(feat_dimension, 1024),
                nn.ReLU(True),
                nn.Dropout(p=classifier_dropout),
                nn.Linear(1024, 1024),
                nn.ReLU(True),
                nn.Dropout(p=classifier_dropout),
                nn.Linear(1024, num_classes),
            )
        else:
            classification = nn.Identity()

        return nn.ModuleList(layers), classification

    def forward_features(self, x):
        for layer in self.features:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VGGKAN(VGG, PyTorchModelHubMixin):
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 conv_type: str = 'kanconv',
                 kan_conv: Optional[str] = "KAN",
                 kan_classifier: Optional[str] = "KAN",
                 groups: int = 1,
                 spline_order: int = 3,
                 grid_size: int = 5,
                 base_activation: Optional[Callable[..., nn.Module]] = nn.SiLU,
                 grid_range: List = [-1, 1],
                 dropout: float = 0.0,
                 l1_decay: float = 0.0,
                 dropout_linear: float = 0.5,
                 arch: str = 'VGG16',
                 classifier_type: str = 'Linear',
                 expected_feature_shape: Tuple = (1, 1),
                 width_scale: int = 1,
                 affine: bool = False,
                 norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
                 kan_norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
                 std_conv_kernel_size: int = 3,
                 std_conv_padding: int = 1,
                 std_conv_bias: bool = True,
                 degree: int = 3,
                 conv_dropout: float = 0.0,
                 classifier_spline_order: Optional[int] = None,
                 classifier_grid_size: Optional[int] = None,
                 classifier_base_activation: Optional[Callable[..., nn.Module]] = None,
                 classifier_grid_range: Optional[List] = None,
                 classifier_l1_decay: Optional[float] = None,
                 classifier_dropout: Optional[float] = None,
                 classifier_degree: Optional[int] = None,
                 **kwargs: Any
                 ):
        classifier_factory: Optional[Callable[..., nn.Module]] = None
        _kan_classifier_type_name = "None"
        final_classifier_dropout = dropout_linear if classifier_dropout is None else classifier_dropout


        if classifier_type in ['HiddenKAN', 'VGGKAN', 'KAN']:
            _kan_classifier_type = kan_classifier if kan_classifier else "KAN"

            classifier_kan_classifierc = MLP_KAN_FACTORY[_kan_classifier_type]

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
            classifier_kan_sig = signature(classifier_kan_classifierc)
            valid_classifier_args = {k: v for k, v in classifier_kan_args.items() if k in classifier_kan_sig.parameters}
            valid_extra_classifier_args = {k: v for k, v in kwargs.items() if k.startswith('classifier_') and k[len('classifier_'):] in classifier_kan_sig.parameters}
            valid_classifier_args.update({k[len('classifier_'):]: v for k, v in valid_extra_classifier_args.items()})


            classifier_factory = partial(classifier_kan_classifierc, **valid_classifier_args)


        kan_conv_suffix = f"_{kan_conv.upper()}" if conv_type == 'kanconv' else "_CONV"
        head_suffix = classifier_type
        if classifier_factory is not None:
             head_suffix += f"_{_kan_classifier_type.upper()}"
        arch_suffix = f"_{arch}"
        self.name = f"VGGKAN_{head_suffix}{kan_conv_suffix}{arch_suffix}"

        if arch not in cfgs:
            raise ValueError(f"Unknown arch: {arch}. Available types: {list(cfgs.keys())}")

        make_layers_kwargs = {k:v for k,v in kwargs.items() if not k.startswith('classifier_')}

        features, head = self.make_layers(
            cfg=cfgs[arch],
            conv_type=conv_type,
            kan_conv=kan_conv,
            classifier_type=classifier_type,
            kan_classifier=classifier_factory,
            spline_order=spline_order,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=grid_range,
            dropout=dropout, 
            l1_decay=l1_decay,
            groups=groups,
            std_conv_kernel_size=std_conv_kernel_size,
            std_conv_padding=std_conv_padding,
            std_conv_bias=std_conv_bias,
            expected_feature_shape=expected_feature_shape,
            num_input_features=input_channels,
            num_classes=num_classes,
            width_scale=width_scale,
            classifier_dropout=final_classifier_dropout, 
            affine=affine,
            norm_layer=norm_layer,
            kan_norm_layer=kan_norm_layer,
            degree=degree,
            conv_dropout=conv_dropout, 
            **make_layers_kwargs
        )

        super().__init__(features, head, expected_feature_shape)


def vggkan(input_channels: int,
           num_classes: int,
           conv_type: str = 'kanconv',
           kan_conv: Optional[str] = "KAN",
           kan_classifier: Optional[str] = "KAN",
           classifier_type: str = 'Linear',
           groups: int = 1,
           spline_order: int = 3,
           grid_size: int = 5,
           base_activation: Optional[Callable[..., nn.Module]] = nn.SiLU,
           grid_range: List = [-1, 1],
           dropout: float = 0.0,
           l1_decay: float = 0.0,
           dropout_linear: float = 0.5,
           arch: str = 'VGG16',
           expected_feature_shape: Tuple[int, int] = (1, 1),
           width_scale: int = 1,
           affine: bool = False,
           norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
           kan_norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
           std_conv_kernel_size: int = 3,
           std_conv_padding: int = 1,
           std_conv_bias: bool = True,
           degree: int = 3,
           conv_dropout: float = 0.0,
           **kwargs: Any
           ):
    return VGGKAN(input_channels=input_channels, num_classes=num_classes, conv_type=conv_type,
                  kan_conv=kan_conv, kan_classifier=kan_classifier,
                  groups=groups, spline_order=spline_order, grid_size=grid_size,
                  base_activation=base_activation, grid_range=grid_range, dropout=dropout,
                  l1_decay=l1_decay, dropout_linear=dropout_linear, arch=arch,
                  classifier_type=classifier_type, expected_feature_shape=expected_feature_shape,
                  width_scale=width_scale, affine=affine, norm_layer=norm_layer, kan_norm_layer=kan_norm_layer,
                  std_conv_kernel_size=std_conv_kernel_size, std_conv_padding=std_conv_padding,
                  std_conv_bias=std_conv_bias, degree=degree, conv_dropout=conv_dropout,
                  **kwargs)