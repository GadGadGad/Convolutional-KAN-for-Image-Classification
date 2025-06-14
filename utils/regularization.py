# Based on this implementations: https://github.com/szymonmaszke/torchlayers/blob/master/torchlayers/regularization.py
import abc

import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseInjection(nn.Module):
    def __init__(self, p: float = 0.0, alpha: float = 0.05):
        super(NoiseInjection, self).__init__()
        self.p = p
        self.alpha = alpha

    def get_noise(self, x):
        dims = tuple(i for i in range(len(x.shape)) if i != 1)
        std = torch.std(x, dim=dims, keepdim=True)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype) * std
        return noise

    def forward(self, x):
        if self.training:
            mask = torch.rand(x.shape, device=x.device, dtype=x.dtype)
            mask = (mask < self.p).float() * 1
            x = x + self.alpha * mask * self.get_noise(x)
            return x
        return x


class NoiseMultiplicativeInjection(nn.Module):
    def __init__(self, p: float = 0.05, alpha: float = 0.05, betta: float = 0.01):
        super(NoiseMultiplicativeInjection, self).__init__()
        self.p = p
        self.alpha = alpha
        self.betta = betta

    def get_noise(self, x):
        dims = tuple(i for i in range(len(x.shape)) if i != 1)
        std = torch.std(x, dim=dims, keepdim=True)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype) * std
        return noise

    def get_m_noise(self, x):
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype) * self.betta + 1
        return noise

    def forward(self, x):
        if self.training:
            mask = torch.rand(x.shape, device=x.device, dtype=x.dtype)
            mask = (mask < self.p).float() * 1
            mask_m = torch.rand(x.shape, device=x.device, dtype=x.dtype)
            mask_m = (mask_m < self.p).float() * 1
            x = x + x * mask_m * self.get_m_noise(x) + self.alpha * mask * self.get_noise(x)
            return x
        return x


class WeightDecay(nn.Module):
    def __init__(self, module, weight_decay, name: str = None):
        if weight_decay < 0.0:
            raise ValueError(
                "Regularization's weight_decay should be greater than 0.0, got {}".format(
                    weight_decay
                )
            )

        super().__init__()
        self.module = module
        self.weight_decay = weight_decay
        self.name = name

        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    def remove(self):
        self.hook.remove()

    def _weight_decay_hook(self, *_):
        if self.name is None:
            for param in self.module.parameters():
                if param.grad is None or torch.all(param.grad == 0.0):
                    param.grad = self.regularize(param)
        else:
            for name, param in self.module.named_parameters():
                if self.name in name and (
                    param.grad is None or torch.all(param.grad == 0.0)
                ):
                    param.grad = self.regularize(param)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def extra_repr(self) -> str:
        representation = "weight_decay={}".format(self.weight_decay)
        if self.name is not None:
            representation += ", name={}".format(self.name)
        return representation

    @abc.abstractmethod
    def regularize(self, parameter):
        pass


class L2(WeightDecay):
    r"""Regularize module's parameters using L2 weight decay.

    Example::

        import torchlayers as tl

        # Regularize only weights of Linear module
        regularized_layer = tl.L2(tl.Linear(30), weight_decay=1e-5, name="weight")

    .. note::
            Backward hook will be registered on `module`. If you wish
            to remove `L2` regularization use `remove()` method.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters will be regularized.
    weight_decay : float
        Strength of regularization (has to be greater than `0.0`).
    name : str, optional
        Name of parameter to be regularized (if any).
        Default: all parameters will be regularized (including "bias").

    """

    def regularize(self, parameter):
        return self.weight_decay * parameter.data


class L1(WeightDecay):
    """Regularize module's parameters using L1 weight decay.

    Example::

        import torchlayers as tl

        # Regularize all parameters of Linear module
        regularized_layer = tl.L1(tl.Linear(30), weight_decay=1e-5)

    .. note::
            Backward hook will be registered on `module`. If you wish
            to remove `L1` regularization use `remove()` method.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters will be regularized.
    weight_decay : float
        Strength of regularization (has to be greater than `0.0`).
    name : str, optional
        Name of parameter to be regularized (if any).
        Default: all parameters will be regularized (including "bias").

    """

    def regularize(self, parameter):
        return self.weight_decay * torch.sign(parameter.data)


# Regularization for KAN-Based Model
class SmoothnessRegularization(nn.Module):
    """
    Computes the smoothness regularization loss for a KANLinear layer.

    This loss penalizes the integral of the squared second derivative of the
    spline functions implicitly defined by the KANLinear layer's weights.

    Args:
        kan_layer (KANLinear): The KANLinear layer instance to regularize.
        lambda_smooth (float): The regularization strength (lambda from the paper).
                                If 0.0, this module returns a zero loss.
        grid_points (int): Number of points to sample for numerical integration
                           of the second derivative.
        h (float): Step size for numerical differentiation (computing the
                   second derivative).
    """
    def __init__(self,
                 lambda_smooth: float
                 ):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        
    def forward(self, *kan_layers):
        """
        Calculates the scalar smoothness loss value.
        """
        total_smoothness_loss = 0.0
        for i, layer in enumerate(kan_layers):
            # Shape: (out, in, coeff)
            coeffs = layer.scaled_spline_weight
            # Sum over all dimensions (out, in, coeff-2)
            diff2 = coeffs[..., 2:] - 2 * coeffs[..., 1:-1] + coeffs[..., :-2]
            
            layer_loss = torch.sum(diff2 * 2)
            total_smoothness_loss += layer_loss
        
        return self.lambda_smooth * total_smoothness_loss
    
    def __repr__(self):
        return f"{self.__class__.__name__}(lambda_smooth={self.lambda_smooth})"
    
    
class SegmentDeactivation(nn.Module):
    """
    Wraps a KANLinear layer to apply Segment Deactivation during training.

    During training, with probability p_deactivate, the contribution of each
    individual spline (input i to output j) is replaced by a linear function
    connecting the spline's values at the defined grid boundaries.

    Args:
        kan_layer (KANLinear): The KANLinear instance to wrap.
        p_deactivate (float): Probability of deactivating a spline segment
                               during training (must be between 0 and 1).
    """
    def __init__(self, 
                 kan_layer: nn.Module,
                 p_deactivate: float):
        super().__init__()
        self.kan_layer = kan_layer
        self.p_deactivate = p_deactivate
        
        grid = self.kan_layer.grid
        if hasattr(self.kan_layer, 'grid_min_max'):
            grid_min = self.kan_layer.grid_min_max[0]
            grid_max = self.kan_layer.grid_min_max[1]
        else:
            grid_min = grid[:, self.kan_layer.spline_order].min() 
            grid_max = grid[:, -self.kan_layer.spline_order-1].max()
            
        self.register_buffer("grid_endpoints", torch.tensor([grid_min, grid_max]))
        
    def _evaluate_wrapped_spline_at_points(self, x_points: torch.Tensor):
            """
            Internal helper to evaluate wrapped KANLinear's splines at specific points.

            Args:
                x_points (torch.Tensor): Shape (num_points, in_features). Points to evaluate at.

            Returns:
                torch.Tensor: Shape (num_points, out_features, in_features).
                            Value of Spline_ji at x_points[k, i].
            """
            bases = self.kan_layer.b_splines(x_points) # (num_points, in_features, coeff)
            weights = self.kan_layer.scaled_spline_weight # (out_features, in_features, coeff)
            # einsum: Sum_c (weights_oic * bases_nic) -> (n, o, i)
            spline_values = torch.einsum('nic,oic->noi', bases, weights)
            return spline_values
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass with conditional Segment Deactivation during training.
        """
        if not self.training or self.p_deactivate == 0:
            return self.kan_layer(x)

        original_shape = x.shape
        batch_size = x.shape[0] if x.dim() > 1 else 1
        in_features = self.kan_layer.in_features
        out_features = self.kan_layer.out_features
        x = x.reshape(-1, in_features) # (batch_size, in_features)

        base_output = F.linear(self.kan_layer.base_activation(x), self.kan_layer.base_weight)

        bases = self.kan_layer.b_splines(x) # (batch, in, coeff)
        weights = self.kan_layer.scaled_spline_weight # (out, in, coeff)
        # Sum_c (bases_bic * weights_oic) -> (b, o, i)
        spline_contrib_orig = torch.einsum('bic,oic->boi', bases, weights)

        deactivate_mask = torch.rand(out_features, in_features, device=x.device) < self.p_deactivate

        if torch.any(deactivate_mask):
            grid_ends_tensor = self.grid_endpoints.unsqueeze(1).expand(2, in_features) # (2, in)
            y_vals_at_ends = self._evaluate_wrapped_spline_at_points(grid_ends_tensor)
            y_start = y_vals_at_ends[0] # (out, in)
            y_end = y_vals_at_ends[1] # (out, in)

            # Calculate slope (a) and intercept (b)
            x_start = self.grid_endpoints[0] 
            x_end = self.grid_endpoints[1]   
            delta_x = x_end - x_start

            if abs(delta_x) < 1e-8: # Avoid division by zero
                 a = torch.zeros_like(y_start)
                 b = y_start # Constant function if grid is single point
            else:
                 a = (y_end - y_start) / delta_x # (out, in)
                 b = y_start - a * x_start        # (out, in)

            # Calculate linear replacement: a_oi * x_bi + b_oi -> (batch, out, in)
            linear_replacement = a.unsqueeze(0) * x.unsqueeze(1) + b.unsqueeze(0)

            # Expand mask for batch dimension: (1, out, in)
            final_spline_contrib = torch.where(
                deactivate_mask.unsqueeze(0), # (1, out, in)
                linear_replacement,           # (batch, out, in)
                spline_contrib_orig           # (batch, out, in)
            )
        else:
            final_spline_contrib = spline_contrib_orig

        spline_output = final_spline_contrib.sum(dim=2) # (batch, out)

        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], out_features)
        return output

    def __getattr__(self, name):
        return super().__getattr__(name)    

    def __repr__(self):
        return f"{self.__class__.__name__}(p_deactivate={self.p_deactivate}, wrapped_layer={repr(self.kan_layer)})"