# taken from and based on https://github.com/1ssb/torchkan/blob/main/torchkan.py
# and https://github.com/1ssb/torchkan/blob/main/LegendreKANet.py
# and https://github.com/ZiyaoLi/fast-kan/blob/master/fastkan/fastkan.py
# Copyright 2024 Li, Ziyao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# and https://github.com/SynodicMonth/ChebyKAN/blob/main/ChebyKANLayer.py
# and https://github.com/Khochawongwat/GRAMKAN/blob/main/model.py
# and https://github.com/zavareh1/Wav-KAN
# and https://github.com/quiqi/relu_kan/issues/2
from typing import List, Type

import torch.nn as nn

# Assume these imports work relative to the file location
# If not, adjust the paths accordingly
from utils.regularization import L1
from layers import BersnsteinKANLayer, BesselKANLayer, ChebyKANLayer, FastKANLayer, FibonacciKANLayer, \
    FourierKANLayer, GegenbauerKANLayer, GRAMKANLayer, HermiteKANLayer, JacobiKANLayer, KANLayer, LaguerreKANLayer, \
        LegendreKANLayer, LucasKANLayer, ReLUKANLayer, TaylorKANLayer, WavKANLayer

class BersnsteinKAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float = 0.0, l1_decay: float = 0.0, degree=3,
                 base_activation: Type[nn.Module] = nn.SiLU, first_dropout: bool = True, **kwargs):
        super(BersnsteinKAN, self).__init__()
        self.layers_hidden = layers_hidden
        self.polynomial_order = degree
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = BersnsteinKANLayer(in_features, out_features, degree, base_activation=base_activation)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class BesselKAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float=0.0, l1_decay: float=0.0, degree: int=3,
                 first_dropout: bool = True, **kwargs):
        super(BesselKAN, self).__init__()
        self.layers_hidden = layers_hidden
        self.polynomial_order = degree
        self.layers = nn.ModuleList([])
        self.num_layers = len(layers_hidden[:-1])

        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = BesselKANLayer(in_features, out_features, degree)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ChebyKAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float = 0.0, l1_decay: float = 0.0,
                 degree=3, first_dropout: bool = True, **kwargs):
        super(ChebyKAN, self).__init__()
        self.layers_hidden = layers_hidden
        self.polynomial_order = degree
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = ChebyKANLayer(in_features, out_features, degree)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class FastKAN(nn.Module):
    def __init__(
            self,
            layers_hidden: List[int],
            dropout: float = 0.0,
            l1_decay: float = 0.0,
            grid_range: List[float] = [-2, 2],
            grid_size: int = 8,
            use_base_update: bool = True,
            base_activation: Type[nn.Module] = nn.SiLU,
            spline_weight_init_scale: float = 0.1,
            first_dropout: bool = True, **kwargs
    ) -> None:
        super().__init__()
        self.layers_hidden = layers_hidden
        self.grid_min = grid_range[0]
        self.grid_max = grid_range[1]
        self.use_base_update = use_base_update
        self.base_activation = base_activation
        self.spline_weight_init_scale = spline_weight_init_scale
        self.num_layers = len(layers_hidden[:-1])

        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = FastKANLayer(in_features, out_features,
                                 grid_min=self.grid_min,
                                 grid_max=self.grid_max,
                                 num_grids=grid_size,
                                 use_base_update=use_base_update,
                                 base_activation=base_activation,
                                 spline_weight_init_scale=spline_weight_init_scale)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class FibonacciKAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float=0.0, l1_decay: float=0.0, degree: int=3,
                 first_dropout: bool = True, **kwargs):
        super(FibonacciKAN, self).__init__()
        self.layers_hidden = layers_hidden
        self.polynomial_order = degree
        self.layers = nn.ModuleList([])
        self.num_layers = len(layers_hidden[:-1])

        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = FibonacciKANLayer(in_features, out_features, degree)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class FourierKAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float=0.0, l1_decay: float=0.0,
                 grid_size:int=3, add_bias: bool=True, smooth_initialization:bool = False,
                 first_dropout: bool = True, **kwargs):
        super(FourierKAN, self).__init__()
        self.layers_hidden = layers_hidden
        self.layers = nn.ModuleList([])
        self.num_layers = len(layers_hidden[:-1])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = FourierKANLayer(in_features, out_features, grid_size, add_bias, smooth_initialization)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class GegenbauerKAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float=0.0, l1_decay: float=0.0,
                 degree: int=3, alpha_param: float=0.0,
                 first_dropout: bool = True, **kwargs):
        super(GegenbauerKAN, self).__init__()
        self.polynomial_order = degree
        self.layers_hidden = layers_hidden
        self.layers = nn.ModuleList([])
        self.num_layers = len(layers_hidden[:-1])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = GegenbauerKANLayer(in_features, out_features, degree, alpha_param)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class GRAMKAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float = 0.0, l1_decay: float = 0.0, degree=3,
                 base_activation: Type[nn.Module] = nn.SiLU, first_dropout: bool = True, **kwargs):
        super(GRAMKAN, self).__init__()
        self.layers_hidden = layers_hidden
        self.polynomial_order = degree
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = GRAMKANLayer(in_features, out_features, degree, act=base_activation)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class HermiteKAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float=0.0, l1_decay: float=0.0,
                 degree: int=3,
                 first_dropout: bool = True, **kwargs):
        super(HermiteKAN, self).__init__()
        self.polynomial_order = degree
        self.layers_hidden = layers_hidden
        self.layers = nn.ModuleList([])
        self.num_layers = len(layers_hidden[:-1])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = HermiteKANLayer(in_features, out_features, degree)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class JacobiKAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float = 0.0, l1_decay: float = 0.0, degree=3, a: float = 1, b: float = 1,
                 base_activation: Type[nn.Module] = nn.SiLU, first_dropout: bool = True, **kwargs):
        super(JacobiKAN, self).__init__()
        self.layers_hidden = layers_hidden
        self.polynomial_order = degree
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = JacobiKANLayer(in_features, out_features, degree, a=a, b=b, base_activation=base_activation)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class KAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float = 0.0, grid_size=5, spline_order=3, base_activation: Type[nn.Module] = nn.GELU,
                 grid_range: List = [-1, 1], l1_decay: float = 0.0, first_dropout: bool = True, **kwargs):
        super(KAN, self).__init__()
        self.layers_hidden = layers_hidden
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation
        self.grid_range = grid_range
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = KANLayer(in_features, out_features, grid_size=grid_size, spline_order=spline_order,
                             base_activation=base_activation, grid_range=grid_range)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class LaguerreKAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float=0.0, l1_decay: float=0.0,
                 degree: int=3, alpha: float=0.0,
                 first_dropout: bool = True, **kwargs):
        super(LaguerreKAN, self).__init__()
        self.polynomial_order = degree
        self.alpha = alpha
        self.layers_hidden = layers_hidden
        self.layers = nn.ModuleList([])
        self.num_layers = len(layers_hidden[:-1])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = LaguerreKANLayer(in_features, out_features, degree, alpha)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class LegendreKAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float = 0.0, l1_decay: float = 0.0, degree=3,
                 base_activation: Type[nn.Module] = nn.SiLU, first_dropout: bool = True, **kwargs):
        super(LegendreKAN, self).__init__()
        self.layers_hidden = layers_hidden
        self.polynomial_order = degree
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = LegendreKANLayer(in_features, out_features, degree, base_activation=base_activation)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class LucasKAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float=0.0, l1_decay: float=0.0,
                 degree: int=3,
                 first_dropout: bool = True, **kwargs):
        super(LucasKAN, self).__init__()
        self.polynomial_order = degree
        self.layers_hidden = layers_hidden
        self.layers = nn.ModuleList([])
        self.num_layers = len(layers_hidden[:-1])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = LucasKANLayer(in_features, out_features, degree)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ReLUKAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float = 0.0, l1_decay: float = 0.0, g: int = 1, k: int = 1,
                 train_ab: bool = True,
                 first_dropout: bool = True, **kwargs):
        super(ReLUKAN, self).__init__()
        self.layers_hidden = layers_hidden
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = ReLUKANLayer(in_features, g, k, out_features, train_ab=train_ab)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TaylorKAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float=0.0, l1_decay: float=0.0,
                 degree: int=3, add_bias: bool=False,
                 first_dropout: bool = True, **kwargs):
        super(TaylorKAN, self).__init__()
        self.polynomial_order = degree
        self.add_bias = add_bias
        self.layers_hidden = layers_hidden
        self.layers = nn.ModuleList([])
        self.num_layers = len(layers_hidden[:-1])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = TaylorKANLayer(in_features, out_features, degree, add_bias)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class WavKAN(nn.Module):
    def __init__(self, layers_hidden, dropout: float = 0.0, l1_decay: float = 0.0,
                 first_dropout: bool = True, wavelet_type: str = 'mexican_hat', **kwargs):
        super(WavKAN, self).__init__()
        assert wavelet_type in ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon'], \
            ValueError(f"Unsupported wavelet type: {wavelet_type}")
        self.layers_hidden = layers_hidden
        self.wavelet_type = wavelet_type
        self.layers = nn.ModuleList([])
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.num_layers = len(layers_hidden[:-1])

        for i, (in_features, out_features) in enumerate(zip(layers_hidden[:-1], layers_hidden[1:])):
            layer = WavKANLayer(in_features, out_features, wavelet_type=wavelet_type)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def mlp_kan(layers_hidden: List[int], dropout: float = 0.0, grid_size: int = 5, spline_order: int = 3,
            base_activation: Type[nn.Module] = nn.GELU,
            grid_range: List = [-1, 1], l1_decay: float = 0.0, first_dropout: bool = True) -> KAN:
    return KAN(layers_hidden, dropout=dropout, grid_size=grid_size, spline_order=spline_order,
               base_activation=base_activation, grid_range=grid_range, l1_decay=l1_decay, first_dropout=first_dropout)

def mlp_fastkan(layers_hidden: List[int], dropout: float = 0.0, grid_size: int = 8, base_activation: Type[nn.Module] = nn.SiLU,
                grid_range: List = [-2, 2], l1_decay: float = 0.0, use_base_update: bool = True,
                spline_weight_init_scale: float = 0.1, first_dropout: bool = True) -> FastKAN:
    return FastKAN(layers_hidden, dropout=dropout, grid_size=grid_size,
                   base_activation=base_activation, grid_range=grid_range, l1_decay=l1_decay,
                   use_base_update=use_base_update, spline_weight_init_scale=spline_weight_init_scale,
                   first_dropout=first_dropout)

def mlp_legendrekan(layers_hidden: List[int], dropout: float = 0.0, degree: int = 3,
             base_activation: Type[nn.Module] = nn.SiLU, l1_decay: float = 0.0, first_dropout: bool = True) -> LegendreKAN:
    return LegendreKAN(layers_hidden, dropout=dropout, base_activation=base_activation, degree=degree, l1_decay=l1_decay, first_dropout=first_dropout)


def mlp_bersnsteinkan(layers_hidden: List[int], dropout: float = 0.0, degree: int = 3,
             base_activation: Type[nn.Module] = nn.SiLU, l1_decay: float = 0.0, first_dropout: bool = True) -> BersnsteinKAN:
    return BersnsteinKAN(layers_hidden, dropout=dropout, base_activation=base_activation, degree=degree, l1_decay=l1_decay, first_dropout=first_dropout)


def mlp_chebykan(layers_hidden: List[int], dropout: float = 0.0, degree: int = 3, l1_decay: float = 0.0, first_dropout: bool = True) -> ChebyKAN:
    return ChebyKAN(layers_hidden, dropout=dropout, degree=degree, l1_decay=l1_decay, first_dropout=first_dropout)


def mlp_jacobikan(layers_hidden: List[int], dropout: float = 0.0, degree: int = 3, l1_decay: float = 0.0,
             a: float = 1.0, b: float = 1.0, base_activation: Type[nn.Module] = nn.SiLU, first_dropout: bool = True) -> JacobiKAN:
    return JacobiKAN(layers_hidden, dropout=dropout, degree=degree, l1_decay=l1_decay, a=a, b=b, base_activation=base_activation, first_dropout=first_dropout)


def mlp_gramkan(layers_hidden: List[int], dropout: float = 0.0, degree: int = 3,
             base_activation: Type[nn.Module] = nn.SiLU, l1_decay: float = 0.0, first_dropout: bool = True) -> GRAMKAN:
    return GRAMKAN(layers_hidden, dropout=dropout, base_activation=base_activation,
                degree=degree, l1_decay=l1_decay, first_dropout=first_dropout)
    
def mlp_besselkan(layers_hidden: List[int], dropout: float = 0.0, degree: int = 3, l1_decay: float = 0.0, first_dropout: bool = True) -> BesselKAN:
    return BesselKAN(layers_hidden, dropout=dropout, degree=degree, l1_decay=l1_decay, first_dropout=first_dropout)

def mlp_fibonaccikan(layers_hidden: List[int], dropout: float = 0.0, degree: int = 3, l1_decay: float = 0.0, first_dropout: bool = True) -> FibonacciKAN:
    return FibonacciKAN(layers_hidden, dropout=dropout, degree=degree, l1_decay=l1_decay, first_dropout=first_dropout)

def mlp_fourierkan(layers_hidden: List[int], dropout: float = 0.0, grid_size: int = 3, l1_decay: float = 0.0,
                   add_bias: bool = True, smooth_initialization: bool = False, first_dropout: bool = True) -> FourierKAN:
    return FourierKAN(layers_hidden, dropout=dropout, grid_size=grid_size, l1_decay=l1_decay,
                      add_bias=add_bias, smooth_initialization=smooth_initialization, first_dropout=first_dropout)

def mlp_gegenbauerkan(layers_hidden: List[int], dropout: float = 0.0, degree: int = 3, l1_decay: float = 0.0,
               alpha_param: float = 0.0, first_dropout: bool = True) -> GegenbauerKAN:
    return GegenbauerKAN(layers_hidden, dropout=dropout, degree=degree, l1_decay=l1_decay, alpha_param=alpha_param, first_dropout=first_dropout)

def mlp_hermitekan(layers_hidden: List[int], dropout: float = 0.0, degree: int = 3, l1_decay: float = 0.0, first_dropout: bool = True) -> HermiteKAN:
    return HermiteKAN(layers_hidden, dropout=dropout, degree=degree, l1_decay=l1_decay, first_dropout=first_dropout)

def mlp_laguerrekan(layers_hidden: List[int], dropout: float = 0.0, degree: int = 3, l1_decay: float = 0.0,
               alpha: float = 0.0, first_dropout: bool = True) -> LaguerreKAN:
    return LaguerreKAN(layers_hidden, dropout=dropout, degree=degree, l1_decay=l1_decay, alpha=alpha, first_dropout=first_dropout)

def mlp_lucaskan(layers_hidden: List[int], dropout: float = 0.0, degree: int = 3, l1_decay: float = 0.0, first_dropout: bool = True) -> LucasKAN:
    return LucasKAN(layers_hidden, dropout=dropout, degree=degree, l1_decay=l1_decay, first_dropout=first_dropout)

def mlp_relukan(layers_hidden: List[int], dropout: float = 0.0, l1_decay: float = 0.0, g: int = 1, k: int = 1,
                train_ab: bool = True, first_dropout: bool = True) -> ReLUKAN:
    return ReLUKAN(layers_hidden, dropout=dropout, l1_decay=l1_decay, g=g, k=k, train_ab=train_ab, first_dropout=first_dropout)

def mlp_taylorkan(layers_hidden: List[int], dropout: float = 0.0, degree: int = 3, l1_decay: float = 0.0,
             add_bias: bool = False, first_dropout: bool = True) -> TaylorKAN:
    return TaylorKAN(layers_hidden, dropout=dropout, degree=degree, l1_decay=l1_decay, add_bias=add_bias, first_dropout=first_dropout)

def mlp_wavkan(layers_hidden: List[int], dropout: float = 0.0, l1_decay: float = 0.0,
               wavelet_type: str = 'mexican_hat', first_dropout: bool = True) -> WavKAN:
    return WavKAN(layers_hidden, dropout=dropout, l1_decay=l1_decay, wavelet_type=wavelet_type, first_dropout=first_dropout)

MLP_KAN_FACTORY = {
    "KAN": mlp_kan,
    "FastKAN": mlp_fastkan,
    "LegendreKAN": mlp_legendrekan,
    "BersnsteinKAN": mlp_bersnsteinkan,
    "BesselKAN": mlp_besselkan,
    "ChebyKAN": mlp_chebykan,
    "FibonacciKAN": mlp_fibonaccikan,
    "FourierKAN": mlp_fourierkan,
    "GegenbauerKAN": mlp_gegenbauerkan,
    "GRAMKAN": mlp_gramkan,
    "HermiteKAN": mlp_hermitekan,
    "JacobiKAN": mlp_jacobikan,
    "LaguerreKAN": mlp_laguerrekan,
    "LucasKAN": mlp_lucaskan,
    "ReLUKAN": mlp_relukan,
    "TaylorKAN": mlp_taylorkan,
    "WavKAN": mlp_wavkan,
}