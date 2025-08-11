# Copyright (c) 2025 Hongjie Zhang
# Licensed under the MIT License. See LICENSE file in the project root for details.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class JointBilateralFilter(nn.Module):
    """
    ONNX-friendly joint bilateral filter module. No learnable parameters.
    Uses a dense `depth` map to guide propagation of `sparse` values.

    Args:
      sigma_s: float spatial stddev
      sigma_r: float range stddev
      kernel_size: int odd filter window size

    Forward inputs:
      - sparse: Tensor (B,1,H,W), missing entries == 1.0
      - depth:  Tensor (B,1,H,W)
    Returns:
      - filtered: Tensor (B,1,H,W)
    """
    def __init__(self, sigma_s: float, sigma_r: float, kernel_size: int):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        pad = kernel_size // 2
        coords = list(range(-pad, pad+1))
        self.offsets = [(i, j) for i in coords for j in coords]
        # precompute spatial weights as Python floats
        self.weights = [math.exp(-((i**2 + j**2) / (2 * sigma_s**2))) for i, j in self.offsets]
        self.sigma_r = sigma_r
        self.kernel_size = kernel_size
        self.pad = pad

    def forward(self, sparse: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        B, C, H, W = sparse.shape
        pad = self.pad
        # pad inputs
        sp = F.pad(sparse, (pad,pad,pad,pad), mode='constant', value=1.0)
        dp = F.pad(depth,  (pad,pad,pad,pad), mode='reflect')

        # accumulators
        num = torch.zeros_like(sparse)
        den = torch.zeros_like(sparse)

        # center depth for range weights
        center = depth

        for (dy, dx), w in zip(self.offsets, self.weights):
            sp_shift = sp[..., pad+dy:pad+dy+H, pad+dx:pad+dx+W]
            dp_shift = dp[..., pad+dy:pad+dy+H, pad+dx:pad+dx+W]
            valid = (sp_shift != 1.0).float()
            range_w = torch.exp(-((dp_shift - center)**2) / (2 * self.sigma_r**2))
            w_total = w * range_w * valid
            num += w_total * sp_shift
            den += w_total

        out = num / (den + 1e-8)
        out = torch.where(den < 1e-8, torch.ones_like(out), out)
        return out

