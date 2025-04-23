import torch
import torch.nn as nn

# Code from https://github.com/IvanDrokin/torch-conv-kan/blob/main/kans/utils.py

class SplineLinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 init_scale: float=0.1,
                 **kwargs):
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kwargs)
    
    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)
        # nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')

class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            denominator: float = None, 
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)