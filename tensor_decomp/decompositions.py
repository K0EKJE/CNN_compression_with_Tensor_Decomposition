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


def estimate_ranks(layer, method, threshold):
    """ Unfold the 2 modes of the Tensor the decomposition will 
    be performed on, and estimates the ranks of the matrices using VBMF 
    """

    weights = layer.weight.data
    
    unfold_0 = tl.base.unfold(np.asarray(weights), 0) 
    unfold_1 = tl.base.unfold(np.asarray(weights), 1)

    

    if method == 'VBMF':
      _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
      _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
      ranks = [diag_0.shape[0], diag_1.shape[1]]
 
    if method == 'SVD':
      
      U, S, V = torch.svd(torch.tensor(unfold_0))
      U1, S1, V1 = torch.svd(torch.tensor(unfold_1))
      rank0 = (S > threshold).sum().item()
      rank1 = (S1 >threshold).sum().item()

      ranks = [rank0, rank1]

    if method == 'QR':
      # compute the QR decomposition
      threshold=.1
      Q, R = torch.linalg.qr(torch.tensor(unfold_0))
      Q1, R1 = torch.linalg.qr(torch.tensor(unfold_1))
      # compute the rank of the matrix
      rank0 = (torch.abs(torch.diag(R)) > threshold).sum().item()
      rank1 = (torch.abs(torch.diag(R1)) > threshold).sum().item()
      min_rank = max(torch.tensor(unfold_0).shape[0]//10,torch.tensor(unfold_1).shape[0]//10)
      while (rank0<min_rank) or (rank1<min_rank):
        threshold-=0.01
        rank0 = (torch.abs(torch.diag(R)) > threshold).sum().item()
        rank1 = (torch.abs(torch.diag(R1)) > threshold).sum().item()
      print("Threshold = ", threshold, torch.tensor(unfold_0).shape[0])
      ranks = [rank0, rank1]
    

    return ranks

def estimate_threshold(layer):

    weights = layer.weight.data

    unfold_0 = tl.base.unfold(np.asarray(weights), 0) 
    unfold_1 = tl.base.unfold(np.asarray(weights), 1)
 
    U, S, V = torch.svd(torch.tensor(unfold_0))
    U1, S1, V1 = torch.svd(torch.tensor(unfold_1))
    # channel = max(layer.in_channels,layer.out_channels)/50
    # threshold = max(torch.quantile(S, 0.97), torch.quantile(S, 0.97))
    # step_size = min(torch.std(S), torch.std(S1))
    # IQR = (torch.quantile(S, 0.75)-torch.quantile(S, 0.25))
    # IQR1 = (torch.quantile(S1, 0.75)-torch.quantile(S1, 0.25))
    # step_size = min(IQR,IQR1)
    # channel = max(layer.in_channels,layer.out_channels)

    # return threshold-0.00000001, step_size*(1/channel)

    return S

def tucker_decomposition_conv_layer(layer, method):
    """ Gets a conv layer, 
        returns a nn.Sequential object with the Tucker decomposition.

    """
    if method =='SVD':
      ratio = 0 
      target_ratio = 0.6
      # threshold, step_size = estimate_threshold(layer)
      #print("=== ", threshold, step_size)
      # while(ratio<target_ratio):
        
      #   ranks = estimate_ranks(layer, method, threshold=threshold)
      #   while ranks[0]<1 or ranks[1]<1:
      #     ranks = estimate_ranks(layer, method, threshold)
      #     threshold -= step_size
      #   core, [last, first] = \
      #       partial_tucker(np.asarray(layer.weight.data), \
      #           modes=[0, 1], rank=ranks, init='svd')[0]
      #   core, last, first = torch.tensor(core),torch.tensor(last),torch.tensor(first)
      #   appro = tl.tucker_tensor.tucker_to_tensor(partial_tucker(np.asarray(layer.weight.data),\
      #           modes=[0, 1], rank=ranks, init='svd')[0])
      #   ratio = tl.norm(appro)/tl.norm(np.asarray(layer.weight.data))
      #   # if target_ratio - ratio < 0.1: possible adaptive way 
      #   #   step_size /= 5
      #   # print("===" ,threshold, ratio)
      #   threshold -= step_size

      S = estimate_threshold(layer)
      left, right = 0, len(S) - 1
      while left <= right:
        mid = (left + right) // 2
        
        ranks = estimate_ranks(layer, method, threshold=S[mid])
        core, [last, first] = \
            partial_tucker(np.asarray(layer.weight.data), \
                modes=[0, 1], rank=ranks, init='svd')[0]
        core, last, first = torch.tensor(core),torch.tensor(last),torch.tensor(first)
        appro = tl.tucker_tensor.tucker_to_tensor(partial_tucker(np.asarray(layer.weight.data),\
                modes=[0, 1], rank=ranks, init='svd')[0])
        ratio = tl.norm(appro)/tl.norm(np.asarray(layer.weight.data))

        if abs(target_ratio - ratio) < 0.02:

          break
        elif ratio < target_ratio:
          left = mid + 1
        else:
          right = mid - 1

    else:
      ranks = estimate_ranks(layer, method,threshold = 0.8)
      core, [last, first] = \
              partial_tucker(np.asarray(layer.weight.data), \
              modes=[0, 1], rank=ranks, init='svd')[0]
      core, last, first = torch.tensor(core),torch.tensor(last),torch.tensor(first)
      appro = tl.tucker_tensor.tucker_to_tensor(partial_tucker(np.asarray(layer.weight.data),\
              modes=[0, 1], rank=ranks, init='svd')[0])
      ratio = tl.norm(appro)/tl.norm(np.asarray(layer.weight.data))

    print(method+" Estimated ranks: ", ranks)
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

    num_param, num_param_decomp = get_param(first.shape[0],last.shape[0],ranks)



    new_layers = [first_layer, core_layer, last_layer]
    return num_param, num_param_decomp, ratio, nn.Sequential(*new_layers)
