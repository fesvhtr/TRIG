import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional

import warnings

import torch
import torch.nn as nn

try:
    from apex.normalization import FusedRMSNorm as RMSNorm
except ImportError:
    warnings.warn("Cannot import apex RMSNorm, switch to vanilla implementation")


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.
        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

            
def modulate(x, scale):
    return x * (1 + scale.unsqueeze(1))

class LLamaFeedForward(nn.Module):
    """ 
    Corresponds to the FeedForward layer in Next DiT.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float] = None,
        zeros_initialize: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.zeros_initialize = zeros_initialize
        self.dtype = dtype

        # Compute hidden_dim based on the given formula
        hidden_dim_calculated = int(2 * self.hidden_dim / 3)
        if self.ffn_dim_multiplier is not None:
            hidden_dim_calculated = int(self.ffn_dim_multiplier * hidden_dim_calculated)
        hidden_dim_calculated = self.multiple_of * ((hidden_dim_calculated + self.multiple_of - 1) // self.multiple_of)

        # Define linear layers
        self.w1 = nn.Linear(self.dim, hidden_dim_calculated, bias=False)
        self.w2 = nn.Linear(hidden_dim_calculated, self.dim, bias=False)
        self.w3 = nn.Linear(self.dim, hidden_dim_calculated, bias=False)

        # Initialize weights
        if self.zeros_initialize:
            nn.init.zeros_(self.w2.weight)
        else:
            nn.init.xavier_uniform_(self.w2.weight)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w3.weight)

    def _forward_silu_gating(self, x1, x3):
        return F.silu(x1) * x3

    def forward(self, x):
        return self.w2(self._forward_silu_gating(self.w1(x), self.w3(x)))

class FinalLayer(nn.Module):
    """
    The final layer of Next-DiT.
    """
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        # LayerNorm without learnable parameters (elementwise_affine=False)
        self.norm_final = nn.LayerNorm(self.hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(self.hidden_size, np.prod(self.patch_size) * self.out_channels, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        # Initialize the last layer with zeros
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        scale = self.adaLN_modulation(c)
        x = modulate(self.norm_final(x), scale)
        x = self.linear(x)
        return x