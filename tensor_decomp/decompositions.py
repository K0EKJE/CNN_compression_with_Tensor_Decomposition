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
        super().__init__()
        self.block = block
        self.shortcut = shortcut
        "Res applied"
        
    def forward(self, x: Tensor) -> Tensor:
        res = x
        
        x = self.block(x)
        
        if self.shortcut:
            res = self.shortcut(res)

        x += res
        return x



def cp_decomposition_conv_layer(layer, rank, res = False):
    """ Gets a conv layer and a target rank, 
        returns a nn.Sequential object with the decomposition """

    # Perform CP decomposition on the layer weight tensorly. 
    l, f, v, h = parafac(np.asarray(layer.weight.data), rank=rank)[1]
    l, f, v, h = torch.tensor(l),torch.tensor(f),torch.tensor(v), torch.tensor(h)
    appro = tl.cp_to_tensor(parafac(np.asarray(layer.weight.data), rank=rank))
    ratio = tl.norm(appro)/tl.norm(np.asarray(layer.weight.data))
    pointwise_s_to_r_layer = torch.nn.Conv2d(
            in_channels=f.shape[0], 
            out_channels=f.shape[1], 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            dilation=layer.dilation, 
            bias=False)

    depthwise_vertical_layer = torch.nn.Conv2d(
            in_channels=v.shape[1], 
            out_channels=v.shape[1], 
            kernel_size=(v.shape[0], 1),
            stride=1, padding=(layer.padding[0], 0), 
            dilation=layer.dilation,
            groups=v.shape[1], 
            bias=False)

    depthwise_horizontal_layer = torch.nn.Conv2d(
            in_channels=h.shape[1], 
            out_channels=h.shape[1], 
            kernel_size=(1, h.shape[0]), 
            stride=layer.stride,
            padding=(0, layer.padding[0]), 
            dilation=layer.dilation, 
            groups=h.shape[1], 
            bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv2d(
            in_channels=l.shape[1], 
            out_channels=l.shape[0], 
            kernel_size=1, 
            stride=1,
            padding=0, 
            dilation=layer.dilation, 
            bias=True)
    
    pointwise_r_to_t_layer.bias.data = layer.bias.data
    depthwise_horizontal_layer.weight.data = torch.transpose(h, 1, 0).unsqueeze(1).unsqueeze(1)
    depthwise_vertical_layer.weight.data = torch.transpose(v, 1, 0).unsqueeze(1).unsqueeze(-1)
    pointwise_s_to_r_layer.weight.data = torch.transpose(f, 1, 0).unsqueeze(-1).unsqueeze(-1)
    pointwise_r_to_t_layer.weight.data = l.unsqueeze(-1).unsqueeze(-1)

    new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer, 
                  depthwise_horizontal_layer, pointwise_r_to_t_layer]

    if res:
      return ratio, ResidualAdd(nn.Sequential(*new_layers), shortcut=nn.Conv2d(f.shape[0],l.shape[0], kernel_size=1)) #
    else: return ratio, nn.Sequential(*new_layers)


def estimate_ranks(layer):
    """ Unfold the 2 modes of the Tensor the decomposition will 
    be performed on, and estimates the ranks of the matrices using VBMF 
    """

    weights = layer.weight.data
    unfold_0 = tl.base.unfold(weights, 0) 
    unfold_1 = tl.base.unfold(weights, 1)
    _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
    _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
    ranks = [diag_0.shape[0], diag_1.shape[1]]
    return ranks

def tucker_decomposition_conv_layer(layer):
    """ Gets a conv layer, 
        returns a nn.Sequential object with the Tucker decomposition.
        The ranks are estimated with a Python implementation of VBMF
        https://github.com/CasvandenBogaard/VBMF
    """

    ranks = estimate_ranks(layer)
    print(layer, "VBMF Estimated ranks", ranks)
    core, [last, first] = \
        partial_tucker(layer.weight.data, \
            modes=[0, 1], rank=ranks, init='svd')

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0], \
            out_channels=first.shape[1], kernel_size=1,
            stride=1, padding=0, dilation=layer.dilation, bias=False)

    # A regular 2D convolution layer with R3 input channels 
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1], \
            out_channels=core.shape[0], kernel_size=layer.kernel_size,
            stride=layer.stride, padding=layer.padding, dilation=layer.dilation,
            bias=False)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
        out_channels=last.shape[0], kernel_size=1, stride=1,
        padding=0, dilation=layer.dilation, bias=True)

    last_layer.bias.data = layer.bias.data

    first_layer.weight.data = \
        torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)
