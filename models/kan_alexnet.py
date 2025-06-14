from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union
from inspect import signature
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from layers.kan_conv import CONV_KAN_FACTORY, conv, _calculate_same_padding
from .kans import MLP_KAN_FACTORY

class AlexNetKAN(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        num_classes: int = 1000,
        dropout: float = 0.5,
        input_channels: int = 3,
        arch: str = "default",
        conv_type: str = 'kanconv',
        kan_conv: Optional[str] = "KAN", 
        kan_classifier: Optional[str] = "KAN",
        classifier_type: str = 'Linear',
        groups: int = 1,
        spline_order: int = 3,
        grid_size: int = 5,
        base_activation: Optional[Callable[..., nn.Module]] = nn.SiLU,
        grid_range: List = [-1, 1],
        degree: Optional[int] = 3,
        l1_decay: float = 0.0,
        affine: bool = True, 
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d, 
        kan_norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d, 
        classifier_spline_order: Optional[int] = None,
        classifier_grid_size: Optional[int] = None,
        classifier_base_activation: Optional[Callable[..., nn.Module]] = None,
        classifier_grid_range: Optional[List] = None,
        classifier_l1_decay: Optional[float] = None,
        classifier_dropout: Optional[float] = None, 
        classifier_degree: Optional[int] = None,
        conv_dropout: float=0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.arch = arch
        if kan_norm_layer is None:
            kan_norm_layer = norm_layer 

        conv_layer_factory: Callable[..., nn.Module]

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
            def std_conv_wrapper(
                in_planes: int,
                out_planes: int,
                kernel_size: Union[int, Tuple[int, int]],
                stride: Union[int, Tuple[int, int]] = 1,
                padding: Union[int, Tuple[int, int]] = 0,
                groups: int = 1, 
                dilation: Union[int, Tuple[int, int]] = 1,
                norm_layer: Optional[Callable[..., nn.Module]] = norm_layer,
                activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU, 
                affine: bool = affine,
                dropout: int=dropout,
                spline_order=spline_order, grid_size=grid_size, base_activation=base_activation, grid_range=grid_range, l1_decay=l1_decay, degree=degree, **kwargs
            ) -> nn.Sequential:
                if padding is None:
                     if isinstance(kernel_size, int): 
                        padding = _calculate_same_padding(kernel_size, dilation)
                     elif isinstance(kernel_size, tuple):
                        padding = _calculate_same_padding(kernel_size, (dilation, dilation))
                        
                bias = norm_layer is None or not affine
                layers = [
                    conv(
                        in_planes=in_planes,
                        out_planes=out_planes,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                        base_activation=activation_layer,
                        norm_layer=norm_layer,
                        dropout=dropout,
                        l1_decay=l1_decay,
                    )
                ]
                if norm_layer is not None:
                    layers.append(norm_layer(out_planes, affine=affine))
                if activation_layer is not None:
                    layers.append(activation_layer(inplace=True))
                return nn.Sequential(*layers)

            conv_layer_factory = std_conv_wrapper


        features_layers = []
        if self.arch == "default":
            features_layers.append(conv_layer_factory(input_channels, 64, kernel_size=11, stride=4, padding=2, groups=groups))
            features_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            features_layers.append(conv_layer_factory(64, 192, kernel_size=5, padding=2, groups=groups))
            features_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            features_layers.append(conv_layer_factory(192, 384, kernel_size=3, padding=1, groups=groups))
            features_layers.append(conv_layer_factory(384, 256, kernel_size=3, padding=1, groups=groups))
            features_layers.append(conv_layer_factory(256, 256, kernel_size=3, padding=1, groups=groups))
            features_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        elif self.arch == "small":
            features_layers.append(conv_layer_factory(input_channels, 64, kernel_size=5, stride=1, padding=2, groups=groups)) 
            features_layers.append(nn.MaxPool2d(kernel_size=3, stride=2)) 
            features_layers.append(conv_layer_factory(64, 192, kernel_size=5, padding=2, groups=groups))
            features_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
            features_layers.append(conv_layer_factory(192, 384, kernel_size=3, padding=1, groups=groups))
            features_layers.append(conv_layer_factory(384, 256, kernel_size=3, padding=1, groups=groups))
            features_layers.append(conv_layer_factory(256, 256, kernel_size=3, padding=1, groups=groups))
            features_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.features = nn.Sequential(*features_layers)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6)) 
        classifier_factory: Callable[..., nn.Module]
        feat_dimension = 256 * 6 * 6 
        classifier_spline_order = classifier_spline_order if classifier_spline_order is not None else spline_order
        classifier_grid_size = classifier_grid_size if classifier_grid_size is not None else grid_size
        classifier_base_activation = classifier_base_activation if classifier_base_activation is not None else base_activation
        classifier_grid_range = classifier_grid_range if classifier_grid_range is not None else grid_range
        classifier_l1_decay = classifier_l1_decay if classifier_l1_decay is not None else l1_decay
        classifier_dropout = classifier_dropout if classifier_dropout is not None else dropout 
        classifier_degree = classifier_degree if classifier_degree is not None else degree 

        if classifier_type in ['KAN', 'AlexNetKAN']:
            _kan_classifier_type = kan_classifier if kan_classifier is not None else "KAN"
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
            classifier_factory = partial(classifier_kan_func, **valid_classifier_args)
        else:
            def linear_factory(layers_hidden: List[int]) -> nn.Linear:
                return nn.Linear(layers_hidden[0], layers_hidden[1])
            classifier_factory = linear_factory


        self.classifier = nn.Sequential()
        self.classifier.add_module("head_dropout1", nn.Dropout(p=classifier_dropout))
        if arch == 'default':
            if classifier_type == 'Linear' or classifier_type == 'AlexNet':
                self.classifier.add_module("fc1", nn.Linear(feat_dimension, 4096))
                self.classifier.add_module("relu1", nn.ReLU(True))
                self.classifier.add_module("head_dropout2", nn.Dropout(p=classifier_dropout))
                self.classifier.add_module("fc2", nn.Linear(4096, 4096))
                self.classifier.add_module("relu2", nn.ReLU(True))
                self.classifier.add_module("fc3", nn.Linear(4096, num_classes))
            elif classifier_type == 'KAN':
                self.classifier.add_module("fc1", nn.Linear(feat_dimension, 4096))
                # self.classifier.add_module("kan_fc1", nn.Linear([feat_dimension, 4096]))
                self.classifier.add_module("relu1", nn.ReLU(True))
                self.classifier.add_module("head_dropout2", nn.Dropout(p=classifier_dropout))
                self.classifier.add_module("fc2", nn.Linear(4096, 4096))
                # self.classifier.add_module("kan_fc2", classifier_factory([4096, 4096]))
                self.classifier.add_module("relu2", nn.ReLU(True))
                self.classifier.add_module("kan_fc3", classifier_factory(layers_hidden=[4096, num_classes]))
            else:
                self.classifier.add_module("fc1", nn.Linear(feat_dimension, 4096))
                self.classifier.add_module("relu1", nn.ReLU(True))
                self.classifier.add_module("head_dropout2", nn.Dropout(p=classifier_dropout))
                self.classifier.add_module("fc2", nn.Linear(4096, 4096))
                self.classifier.add_module("relu2", nn.ReLU(True))
                self.classifier.add_module("fc3", nn.Linear(4096, num_classes))
        elif self.arch == "small":
            if classifier_type == 'Linear' or classifier_type == 'AlexNet':
                self.classifier.add_module("fc1", nn.Linear(feat_dimension, 1024))
                self.classifier.add_module("relu1", nn.ReLU(True))
                self.classifier.add_module("head_dropout2", nn.Dropout(p=classifier_dropout))
                self.classifier.add_module("fc2", nn.Linear(1024, 1024))
                self.classifier.add_module("relu2", nn.ReLU(True))
                self.classifier.add_module("fc3", nn.Linear(1024, num_classes))
            elif classifier_type == 'KAN':
                self.classifier.add_module("fc1", nn.Linear(feat_dimension, 1024))
                # self.classifier.add_module("kan_fc1", nn.Linear([feat_dimension, 1024]))
                self.classifier.add_module("relu1", nn.ReLU(True))
                self.classifier.add_module("head_dropout2", nn.Dropout(p=classifier_dropout))
                self.classifier.add_module("fc2", nn.Linear(1024, 1024))
                # self.classifier.add_module("kan_fc2", classifier_factory([1024, 1024]))
                self.classifier.add_module("relu2", nn.ReLU(True))
                self.classifier.add_module("kan_fc3", classifier_factory(layers_hidden=[1024, num_classes]))
            else:
                self.classifier.add_module("fc1", nn.Linear(feat_dimension, 1024))
                self.classifier.add_module("relu1", nn.ReLU(True))
                self.classifier.add_module("head_dropout2", nn.Dropout(p=classifier_dropout))
                self.classifier.add_module("fc2", nn.Linear(1024, 1024))
                self.classifier.add_module("relu2", nn.ReLU(True))
                self.classifier.add_module("fc3", nn.Linear(1024, num_classes))
                
        self._initialize_weights()

        kan_conv_suffix = f"_{kan_conv.upper()}" if conv_type == 'kanconv' else "_CONV"
        head_suffix = classifier_type
        if classifier_type in MLP_KAN_FACTORY or classifier_type in ['KAN', 'AlexNetKAN']:
            kanclassifier_name = kan_classifier if kan_classifier else 'KAN'
            head_suffix += f"_{kanclassifier_name.upper()}"

        self.name = f"AlexNet_{head_suffix}{kan_conv_suffix}"


    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                 if m.weight is not None:
                     nn.init.constant_(m.weight, 1)
                 if m.bias is not None:
                     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def alexnet_kan(
        num_classes: int = 1000,
        input_channels: int = 3,
        dropout: float = 0.5,
        arch: str = "default",
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
        affine: bool = True,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        kan_norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
        classifier_spline_order: Optional[int] = None,
        classifier_grid_size: Optional[int] = None,
        classifier_base_activation: Optional[Callable[..., nn.Module]] = None,
        classifier_grid_range: Optional[List] = None,
        classifier_l1_decay: Optional[float] = None,
        classifier_dropout: Optional[float] = None,
        degree: Optional[int] = 3,
        **kwargs: Any
    ) -> AlexNetKAN:
    model = AlexNetKAN(
        num_classes=num_classes,
        dropout=dropout,
        input_channels=input_channels,
        arch=arch,
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
        affine=affine,
        norm_layer=norm_layer,
        kan_norm_layer=kan_norm_layer,
        classifier_spline_order=classifier_spline_order,
        classifier_grid_size=classifier_grid_size,
        classifier_base_activation=classifier_base_activation,
        classifier_grid_range=classifier_grid_range,
        classifier_l1_decay=classifier_l1_decay,
        classifier_dropout=classifier_dropout,
        degree=degree,
        **kwargs,
    )
    return model
