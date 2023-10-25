import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import torch.nn as nn
from tensor_decomp.VBMF import VBMF

from typing import Optional
from torch import Tensor

class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: Optional[nn.Module] = None):
        """
        Initialize a ResidualAdd block.

        Args:
            block (nn.Module): The main block of the residual.
            shortcut (Optional[nn.Module]): An optional shortcut connection.

        This class defines a residual block that consists of a main block and an
        optional shortcut connection. The main block is typically a neural network
        layer or module that performs some transformations on the input data.
        """
        super().__init__()
        self.block = block
        self.shortcut = shortcut

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ResidualAdd block.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the residual block.

        This method defines the forward pass of the residual block. It first saves
        the input tensor as 'res', then applies the main block to 'x'. If a shortcut
        connection is provided, it also applies the shortcut to 'res'. Finally,
        it adds 'res' to the output of the main block and returns the result.
        """
        res = x  # Save the input as 'res'

        x = self.block(x)  # Apply the main block to 'x'

        if self.shortcut:
            res = self.shortcut(res)  # Apply the shortcut connection to 'res'

        x += res  # Add 'res' to the output of the main block
        return x


def get_param(in_c, out, rank):
  # define the input and output channels, filter size, and stride
  in_channels = in_c
  out_channels = rank[1]
  kernel_size = 1
  stride = 1

  # create a convolutional layer
  conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

  # count the number of parameters
  num_params1 = sum(p.numel() for p in conv_layer.parameters())

  # define the input and output channels, filter size, and stride
  in_channels = rank[1]
  out_channels = rank[0]
  kernel_size = 3
  stride = 1
  padding = 1
  # create a convolutional layer
  conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

  # count the number of parameters
  num_params2 = sum(p.numel() for p in conv_layer.parameters())

  # define the input and output channels, filter size, and stride
  in_channels = rank[0]
  out_channels = out
  kernel_size = 1
  stride = 1

  # create a convolutional layer
  conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

  # count the number of parameters
  num_params3 = sum(p.numel() for p in conv_layer.parameters())

  in_channels = in_c
  out_channels = out
  kernel_size = 3
  stride = 1

  # create a convolutional layer
  conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

  # count the number of parameters
  num_params = sum(p.numel() for p in conv_layer.parameters())

  return num_params, num_params1+num_params2+num_params3



def cp_decomposition_conv_layer(layer, rank, res=False):
    """
    Perform CP decomposition on a convolutional layer's weight tensor.

    Args:
        layer: The input convolutional layer.
        rank (int): The target rank for the decomposition.
        res (bool): Whether to use a residual connection.

    Returns:
        Tuple[float, nn.Module]: A tuple containing the compression ratio and the
        decomposed convolutional layer(s) as an nn.Sequential object.

    This function performs CP decomposition on the weight tensor of a convolutional
    layer and returns the decomposed layers. It also optionally includes a residual
    connection if 'res' is True.
    """
    # Perform CP decomposition on the layer's weight tensor using tensorly
    l, f, v, h = parafac(np.asarray(layer.weight.data), rank=rank)[1]
    l, f, v, h = torch.tensor(l), torch.tensor(f), torch.tensor(v), torch.tensor(h)
    # Calculate the approximation ratio
    appro = tl.cp_to_tensor(parafac(np.asarray(layer.weight.data), rank=rank))
    ratio = tl.norm(appro) / tl.norm(np.asarray(layer.weight.data))

    # Create pointwise convolution layer to reduce spatial dimensions
    pointwise_s_to_r_layer = torch.nn.Conv2d(
        in_channels=f.shape[0],
        out_channels=f.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=False)

    # Create depthwise vertical convolution layer
    depthwise_vertical_layer = torch.nn.Conv2d(
        in_channels=v.shape[1],
        out_channels=v.shape[1],
        kernel_size=(v.shape[0], 1),
        stride=1,
        padding=(layer.padding[0], 0),
        dilation=layer.dilation,
        groups=v.shape[1],
        bias=False)

    # Create depthwise horizontal convolution layer
    depthwise_horizontal_layer = torch.nn.Conv2d(
        in_channels=h.shape[1],
        out_channels=h.shape[1],
        kernel_size=(1, h.shape[0]),
        stride=layer.stride,
        padding=(0, layer.padding[0]),
        dilation=layer.dilation,
        groups=h.shape[1],
        bias=False)

    # Create pointwise convolution layer to expand spatial dimensions
    pointwise_r_to_t_layer = torch.nn.Conv2d(
        in_channels=l.shape[1],
        out_channels=l.shape[0],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=True)

    # Copy bias data from the original layer to the pointwise_r_to_t_layer
    pointwise_r_to_t_layer.bias.data = layer.bias.data

    # Set weights for each decomposed layer
    depthwise_horizontal_layer.weight.data = torch.transpose(h, 1, 0).unsqueeze(1).unsqueeze(1)
    depthwise_vertical_layer.weight.data = torch.transpose(v, 1, 0).unsqueeze(1).unsqueeze(-1)
    pointwise_s_to_r_layer.weight.data = torch.transpose(f, 1, 0).unsqueeze(-1).unsqueeze(-1)
    pointwise_r_to_t_layer.weight.data = l.unsqueeze(-1).unsqueeze(-1)

    # Create a list of the new layers
    new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer,
                  depthwise_horizontal_layer, pointwise_r_to_t_layer]

    if res:
        # Include a residual connection if 'res' is True
        return ratio, ResidualAdd(nn.Sequential(*new_layers), shortcut=nn.Conv2d(f.shape[0], l.shape[0], kernel_size=1))
    else:
        return ratio, nn.Sequential(*new_layers)


def estimate_ranks(layer, method, threshold):
    """
    Estimate the ranks of unfolded matrices using VBMF or SVD.

    Args:
        layer: The layer whose weights will be used for rank estimation.
        method (str): The method to use for rank estimation ('VBMF' or 'SVD').
        threshold (float): The threshold value for SVD-based rank estimation.

    Returns:
        List[int]: A list containing the estimated ranks for each unfolded matrix.

    This function unfolds the weight tensor of a layer along its modes and estimates
    the ranks of the unfolded matrices using either VBMF or SVT methods.
    """
    weights = layer.weight.data

    # Unfold the weight tensor along mode 0 and mode 1
    unfold_0 = tl.base.unfold(np.asarray(weights), 0)
    unfold_1 = tl.base.unfold(np.asarray(weights), 1)

    ranks = []

    if method == 'VBMF':
        # Estimate ranks using VBMF (Variational Bayesian Matrix Factorization)
        _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
        _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
        ranks = [diag_0.shape[0], diag_1.shape[1]]

    elif method == 'SVD':
        # Estimate ranks using SVT (Singular Value Thresholding)
        U, S, V = torch.svd(torch.tensor(unfold_0))
        U1, S1, V1 = torch.svd(torch.tensor(unfold_1))
        rank0 = (S > threshold).sum().item()
        rank1 = (S1 > threshold).sum().item()
        ranks = [rank0, rank1]

    return ranks

def estimate_threshold(layer):
    """
    Estimate the singular values of unfolded matrices using SVD.

    Args:
        layer: The layer whose weights will be used for singular value estimation.

    Returns:
        torch.Tensor: The singular values of the unfolded matrices.

    This function unfolds the weight tensor of a layer along its modes and estimates
    the singular values of the unfolded matrices using Singular Value Decomposition
    for the first mode.
    """
    weights = layer.weight.data

    # Unfold the weight tensor along mode 0 and mode 1
    unfold_0 = tl.base.unfold(np.asarray(weights), 0)
    unfold_1 = tl.base.unfold(np.asarray(weights), 1)

    # Compute SVD for unfolded matrices
    U, S, V = torch.svd(torch.tensor(unfold_0))
    U1, S1, V1 = torch.svd(torch.tensor(unfold_1))

    # Return the singular values
    return S

def tucker_decomposition_conv_layer(layer, method, target_ratio_):
    """
    Perform Tucker decomposition on a convolutional layer's weight tensor.

    Args:
        layer: The input convolutional layer.
        method (str): The method to use for Tucker decomposition ('SVD' or 'VBMF').
        target_ratio_ (float): The target compression ratio.

    Returns:
        Tuple[int, int, float, nn.Module]: A tuple containing the number of parameters
        before and after decomposition, the compression ratio, and the decomposed
        convolutional layer(s) as an nn.Sequential object.

    This function performs Tucker decomposition on the weight tensor of a convolutional
    layer and returns the decomposed layers along with compression-related information.
    """
    if method == 'SVD':
        ratio = 0
        target_ratio = target_ratio_

        # Estimate the singular values of the unfolded matrices
        S = estimate_threshold(layer)

        # Binary search for the optimal rank
        left, right = 0, len(S) - 1
        while left <= right:
            mid = (left + right) // 2

            ranks = estimate_ranks(layer, method, threshold=S[mid])
            # Perform Partial Tucker decomposition
            core, [last, first] = partial_tucker(np.asarray(layer.weight.data),
                                                 modes=[0, 1], rank=ranks, init='svd')[0]
            core, last, first = torch.tensor(core), torch.tensor(last), torch.tensor(first)  # Convert to PyTorch tensors

            # Compute the approximation ratio
            appro = tl.tucker_tensor.tucker_to_tensor(
                partial_tucker(np.asarray(layer.weight.data),
                               modes=[0, 1], rank=ranks, init='svd')[0])
            ratio = tl.norm(appro) / tl.norm(np.asarray(layer.weight.data))

            # Check if the achieved compression ratio is close to the target
            if abs(target_ratio - ratio) < 0.02:
                break
            elif ratio < target_ratio:
                left = mid + 1
            else:
                right = mid - 1
    else:
        # Rank selection with VBMF
        ranks = estimate_ranks(layer, method, threshold=0.8)

        core, [last, first] = partial_tucker(np.asarray(layer.weight.data),
                                             modes=[0, 1], rank=ranks, init='svd')[0]
        core, last, first = torch.tensor(core), torch.tensor(last), torch.tensor(first)

        appro = tl.tucker_tensor.tucker_to_tensor(
            partial_tucker(np.asarray(layer.weight.data),
                           modes=[0, 1], rank=ranks, init='svd')[0])
        ratio = tl.norm(appro) / tl.norm(np.asarray(layer.weight.data))

    print(method + " Estimated ranks: ", ranks)

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0],
                                  out_channels=first.shape[1], kernel_size=1,
                                  stride=1, padding=0, dilation=layer.dilation, bias=False)

    # A regular 2D convolution layer with R3 input channels
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1],
                                 out_channels=core.shape[0], kernel_size=layer.kernel_size,
                                 stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
                                 bias=False)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1],
                                 out_channels=last.shape[0], kernel_size=1, stride=1,
                                 padding=0, dilation=layer.dilation, bias=True)

    # last_layer.bias.data = layer.bias.data

    first_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    num_param, num_param_decomp = get_param(first.shape[0], last.shape[0], ranks)

    new_layers = [first_layer, core_layer, last_layer]
    return num_param, num_param_decomp, ratio, nn.Sequential(*new_layers)
