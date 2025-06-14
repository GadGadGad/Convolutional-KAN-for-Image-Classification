#Credits to: https://github.com/detkov/Convolution-From-Scratch/
import torch
import numpy as np
from typing import List, Tuple, Union
import time # Added for timing
import logging # Added for logging

logger = logging.getLogger(__name__) # Use the logger

def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    batch_size,n_channels,n, m = matrix.shape

    h_out =  np.floor((n + 2 * padding[0] - kernel_side - (kernel_side - 1) * (dilation[0] - 1)) / stride[0]).astype(int) + 1
    w_out = np.floor((m + 2 * padding[1] - kernel_side - (kernel_side - 1) * (dilation[1] - 1)) / stride[1]).astype(int) + 1
    # b = [kernel_side // 2, kernel_side// 2] # This variable 'b' was unused
    return h_out, w_out, batch_size, n_channels

def multiple_convs_kan_conv2d(matrix, #but as torch tensors. Kernel side asume q el kernel es cuadrado
             kernels,
             kernel_side,
             out_channels,
             stride= (1, 1),
             dilation= (1, 1),
             padding= (0, 0),
             device= "cuda"
             ) -> torch.Tensor:
    """Makes a 2D convolution with the kernel over matrix using defined stride, dilation and padding along axes.

    Args:
        matrix (batch_size, colors, n, m]): 2D matrix to be convolved.
        kernel  (function]): 2D odd-shaped matrix (e.g. 3x3, 5x5, 13x9, etc.).
        stride (Tuple[int, int], optional): Tuple of the stride along axes. With the `(r, c)` stride we move on `r` pixels along rows and on `c` pixels along columns on each iteration. Defaults to (1, 1).
        dilation (Tuple[int, int], optional): Tuple of the dilation along axes. With the `(r, c)` dilation we distancing adjacent pixels in kernel by `r` along rows and `c` along columns. Defaults to (1, 1).
        padding (Tuple[int, int], optional): Tuple with number of rows and columns to be padded. Defaults to (0, 0).

    Returns:
        np.ndarray: 2D Feature map, i.e. matrix after convolution.
    """
    func_start_time = time.time()
    h_out, w_out, batch_size, n_channels = calc_out_dims(matrix, kernel_side, stride, dilation, padding)
    n_convs = len(kernels)
    logger.debug(f"[multiple_convs_kan_conv2d] Input shape: {matrix.shape}")
    logger.debug(f"[multiple_convs_kan_conv2d] Calculated output dims: H={h_out}, W={w_out}, BS={batch_size}, InChannels={n_channels}")
    logger.debug(f"[multiple_convs_kan_conv2d] Num kernels provided: {n_convs}, Target out_channels: {out_channels}")

    matrix_out = torch.zeros((batch_size, out_channels, h_out, w_out), device=device) #estamos asumiendo que no existe la dimension de rgb
    unfold = torch.nn.Unfold((kernel_side, kernel_side), dilation=dilation, padding=padding, stride=stride)

    unfold_start_time = time.time()
    unfolded_matrix = unfold(matrix) 
    unfold_time = time.time() - unfold_start_time
    logger.debug(f"[multiple_convs_kan_conv2d] Unfold output shape: {unfolded_matrix.shape}, Time: {unfold_time:.4f}s")

    view_start_time = time.time()
    conv_groups_prep = unfolded_matrix.view(batch_size, n_channels, kernel_side*kernel_side, h_out*w_out)
    conv_groups = conv_groups_prep.transpose(2, 3)
    view_time = time.time() - view_start_time
    logger.debug(f"[multiple_convs_kan_conv2d] Reshaped/Transposed conv_groups shape: {conv_groups.shape}, Time: {view_time:.4f}s")

    kern_per_out = n_convs // out_channels 
    logger.debug(f"[multiple_convs_kan_conv2d] Kernels per output channel: {kern_per_out} (Effective InChannels per group)")

    loop_start_time = time.time()
    total_kan_forward_time = 0
    
    in_channels_per_group = n_channels // getattr(kernels[0], 'groups', 1) if len(kernels)>0 and hasattr(kernels[0], 'groups') else n_channels 
    num_groups = n_channels // in_channels_per_group


    for c_out in range(out_channels):
        out_channel_accum = torch.zeros((batch_size, h_out, w_out), device=device)
        
        group_idx = c_out // (out_channels // num_groups)
        
        start_in_channel = group_idx * in_channels_per_group
        end_in_channel = start_in_channel + in_channels_per_group

        for k_idx_in_group in range(in_channels_per_group):
            actual_in_channel_idx = start_in_channel + k_idx_in_group
            kernel_list_idx = c_out * in_channels_per_group + k_idx_in_group

            if kernel_list_idx >= n_convs:
                 logger.error(f"Calculated kernel index {kernel_list_idx} is out of bounds (num kernels: {n_convs}). Check group/channel logic.")
                 continue # Skip if index is wrong

            kernel = kernels[kernel_list_idx]

            input_patches = conv_groups[:, actual_in_channel_idx, :, :]

            flattened_patches = input_patches.flatten(0, 1)

            kan_fwd_start_time = time.time()
            conv_result = kernel.conv.forward(flattened_patches) 
            kan_fwd_time = time.time() - kan_fwd_start_time
            total_kan_forward_time += kan_fwd_time

            out_channel_accum += conv_result.view(batch_size, h_out, w_out)

        matrix_out[:, c_out, :, :] = out_channel_accum

    loop_time = time.time() - loop_start_time
    func_time = time.time() - func_start_time
    return matrix_out


def add_padding(matrix: np.ndarray,
                padding: Tuple[int, int]) -> np.ndarray:
    """Adds padding to the matrix.

    Args:
        matrix (np.ndarray): Matrix that needs to be padded. Type is List[List[float]] casted to np.ndarray.
        padding (Tuple[int, int]): Tuple with number of rows and columns to be padded. With the `(r, c)` padding we addding `r` rows to the top and bottom and `c` columns to the left and to the right of the matrix

    Returns:
        np.ndarray: Padded matrix with shape `n + 2 * r, m + 2 * c`.
    """
    n, m = matrix.shape
    r, c = padding

    padded_matrix = np.zeros((n + r * 2, m + c * 2))
    padded_matrix[r : n + r, c : m + c] = matrix

    return padded_matrix