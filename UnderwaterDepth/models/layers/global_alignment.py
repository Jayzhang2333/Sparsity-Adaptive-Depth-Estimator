# Copyright (c) 2025 Hongjie Zhang
# Licensed under the MIT License. See LICENSE file in the project root for details.
import torch
import torch.nn as nn
from .bilateral_filter import JointBilateralFilter
import numpy as np
import torch.nn.functional as F

    
class LeastSquaresEstimatorTorch(nn.Module):
    """
    ONNX-friendly least-squares estimator module. Computes per-batch scale and shift
    to align `estimate` to `prompt_depth` under a validity mask. If the computed scale
    is negative, it reverts to a scale-only fit (no shift).
    """
    def __init__(self, max_pred: float, min_pred: float,
                 clamp_min: float = None, clamp_max: float = None):
        super().__init__()
        self.max_pred = max_pred
        self.min_pred = min_pred
        self.min_sparse = 0.1
        self.max_sparse = max_pred

        # reciprocal clamp bounds (for inverse-depth output)
        self.clamp_min_inv = 1.0 / float(clamp_min) if clamp_min is not None else None
        self.clamp_max_inv = 1.0 / float(clamp_max) if clamp_max is not None else None
        self.eps = 1e-12

    def forward(self,
                estimate:     torch.Tensor,  # (b,1,h,w)
                prompt_depth: torch.Tensor   # (b,1,h,w)
               ):
        # 1) build valid mask and inverse-depth target
        valid = ((prompt_depth > self.min_sparse) & (prompt_depth < self.max_sparse)).float()
        target = torch.where(valid > 0,
                             1.0 / prompt_depth,
                             torch.zeros_like(prompt_depth))

        # 2) sums over spatial dims
        sum_dims = (2, 3)
        a00 = torch.sum(valid * estimate * estimate, dim=sum_dims)  # Σ v·est²
        a01 = torch.sum(valid * estimate,          dim=sum_dims)  # Σ v·est
        a11 = torch.sum(valid,                     dim=sum_dims)  # Σ v
        b0  = torch.sum(valid * estimate * target, dim=sum_dims)  # Σ v·est·tgt
        b1  = torch.sum(valid * target,           dim=sum_dims)  # Σ v·tgt

        # 3) full 2×2 solve → scale_full, shift_full
        det = a00 * a11 - a01 * a01
        scale_full = (a11 * b0 - a01 * b1) / (det + self.eps)
        shift_full = (-a01 * b0 + a00 * b1) / (det + self.eps)

        # 4) simple scale-only fallback: scale_simple = Σ v·est·tgt / Σ v·est²
        if scale_full <0:
            # print("scale only")
            scale_simple = b0 / (a00 + self.eps)

            # 5) pick strategy per batch: if scale_full<0 → use simple, else full
            use_simple = scale_full < 0
            scale = torch.where(use_simple, scale_simple, scale_full)
            shift = torch.where(use_simple,
                                torch.zeros_like(shift_full),
                                shift_full)
        else:
            scale = scale_full
            shift = shift_full

        # 6) apply to estimate
        scale4 = scale.view(-1,1,1,1)
        shift4 = shift.view(-1,1,1,1)
        output = estimate * scale4 + shift4

        # 7) optional clamping
        if self.clamp_min_inv is not None and self.clamp_max_inv is not None:
            output = torch.clamp(output,
                                 min=self.clamp_max_inv,
                                 max=self.clamp_min_inv)
        elif self.clamp_min_inv is not None:
            output = torch.clamp(output, max=self.clamp_min_inv)
        elif self.clamp_max_inv is not None:
            output = torch.clamp(output, min=self.clamp_max_inv)

        return output, valid, target
    
class LaserAlignerTorch(nn.Module):
    """
    Align an affine-invariant inverse-depth map to metric scale using two laser points.

    Inputs to forward:
      - estimate:     (b,1,h_orig,w_orig) inverse-depth map
      - prompt_depth: (b,1,new_h,new_w) sparse map where only two pixels >0 indicating laser points
      - fx:           scalar or tensor focal length in px
      - cx:           scalar or tensor principal point x in px

    The module first resizes the estimate to (new_h,new_w), locates the two non-zero points
    in prompt_depth (already at new_h×new_w), computes the scale factor k so that their
    back-projected X separation equals the known baseline, and applies k to the entire estimate
    to produce a metric-depth map, capped at max_depth.

    Returns:
      - metric_depth: (b,1,new_h,new_w) in metres
    """
    def __init__(self,
                 new_h: int,
                 new_w: int,
                 baseline: float = 0.10,
                 max_pred: float = 12.0,
                 min_pred : float = 0.1,
                 eps: float = 1e-8):
        super().__init__()
        self.new_h = new_h
        self.new_w = new_w
        self.baseline = baseline
        self.max_pred = max_pred
        self.min_pred = min_pred
        self.max_sparse = max_pred
        self.min_sparse = 0.1
        self.eps = eps


    def forward(self,
                estimate: torch.Tensor,
                prompt_depth: torch.Tensor,
                fx, cx) -> torch.Tensor:
        valid = ((prompt_depth > self.min_sparse) & (prompt_depth < self.max_sparse)).float()
        target = torch.where(valid > 0,
                             1.0 / prompt_depth,
                             torch.zeros_like(prompt_depth))
        
        b,_,h_orig,w_orig = estimate.shape
        # 1) Resize estimate to target resolution
        est = F.interpolate(estimate, size=(self.new_h, self.new_w),
                            mode='bicubic', align_corners=False)
        # 2) Use prompt_depth as-is (shape b×1×new_h×new_w), no interpolation
        pts = prompt_depth

        # 3) Find two non-zero pixel coords per batch
        u1 = torch.zeros(b, dtype=torch.long, device=est.device)
        v1 = torch.zeros_like(u1)
        u2 = torch.zeros_like(u1)
        v2 = torch.zeros_like(u1)
        for i in range(b):
            nz = (pts[i,0] > 0).nonzero(as_tuple=False)
            if nz.size(0) < 2:
                raise RuntimeError(f"Batch {i}: need 2 laser points, found {nz.size(0)}")
            v1[i], u1[i] = nz[0]
            v2[i], u2[i] = nz[1]

            u1[i] = u1[i]/w_orig * self.new_w
            u2[i] = u2[i]/w_orig * self.new_w

            v1[i] = v1[i]/h_orig * self.new_h
            v2[i] = v2[i]/h_orig * self.new_h

        # 4) Sample inverse-depth at those points
        d1 = est[torch.arange(b), 0, v1, u1]
        d2 = est[torch.arange(b), 0, v2, u2]

        # 5) Prepare fx, cx as tensors of shape (b,)
        if not torch.is_tensor(fx):
            fx = torch.full((b,), float(fx), device=est.device)
        if not torch.is_tensor(cx):
            cx = torch.full((b,), float(cx), device=est.device)

        # 6) Compute scale k
        denom = ((u2.float() - cx) / (fx * d2)) - ((u1.float() - cx) / (fx * d1))
        denom = denom.clamp(min=self.eps)
        k = self.baseline / denom

        # 7) Apply scale to entire map
        k4 = k.view(b,1,1,1)
        metric = k4 / est

        # 8) Clamp invalid and cap
        metric = torch.where(torch.isfinite(metric), metric,
                             torch.tensor(self.max_pred, device=metric.device))
        metric = torch.clamp(metric, max=self.max_pred, min = self.min_pred)

        metric = F.interpolate(metric, size=(h_orig,w_orig),
                            mode='bicubic', align_corners=False)
        return (1.0 / metric).to(torch.float32), valid, target


class GAandFilter(nn.Module):
    def __init__(self, max_pred: float, min_pred: float, sigma_s: float, sigma_r: float, kernel_size: int, clamp_min: float = None, clamp_max: float = None):
        super().__init__()
        # Precompute reciprocal clamp bounds as python floats
        self.ga = LeastSquaresEstimatorTorch(max_pred=max_pred, min_pred=min_pred, clamp_min=clamp_min, clamp_max=clamp_max)
        self.laser_alignment = LaserAlignerTorch(608, 968, max_pred=max_pred, min_pred=min_pred)
        self.filter = JointBilateralFilter(sigma_s, sigma_r, kernel_size)

    def forward(self, estimate: torch.Tensor,prompt_depth: torch.Tensor, fx = None, cx = None) -> torch.Tensor:
        
        if fx is not None and cx is not None:
            ga_output, sparse_mask, sparse_depth_inv = self.laser_alignment(estimate, prompt_depth, fx, cx)
        else:
            ga_output, sparse_mask, sparse_depth_inv = self.ga(estimate, prompt_depth)
        
       
        d = ga_output.clone()
        ga_mask = (estimate >0) # where infinte area is, actually using the reltive map is better
        sparse_and_ga_mask = sparse_mask.bool() & ga_mask

        scale_residual = torch.zeros_like(sparse_depth_inv)
     
        scale_residual = torch.where(sparse_and_ga_mask,sparse_depth_inv / ga_output, torch.ones_like(sparse_depth_inv))
        
        dense_residual = self.filter(scale_residual, ga_output)
       

        if torch.onnx.is_in_onnx_export():
            ga_output = ga_output.half()

        if torch.onnx.is_in_onnx_export():
            dense_residual = dense_residual.half()
       
        return dense_residual, ga_output, d
    