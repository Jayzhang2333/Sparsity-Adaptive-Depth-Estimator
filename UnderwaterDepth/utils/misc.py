# Modified from: https://github.com/isl-org/ZoeDepth/blob/main/zoedepth/utils/misc.py
################Original License Notice###########################################
# Original work licensed under the MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat
#################################################################################

# Modifications Copyright (c) 2025 Hongjie Zhang
# Licensed under the MIT License. See LICENSE file in the project root for details.

"""Miscellaneous utility functions."""
from io import BytesIO
import matplotlib
import matplotlib.cm
import numpy as np
import requests
import torch
import torch.distributed as dist
import torch.nn
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor

class NormalAverage:
    """A class that computes the normal (simple) average."""
    def __init__(self):
        self.total = 0
        self.count = 0

    def append(self, value):
        self.total += value
        self.count += 1

    def get_value(self):
        if self.count == 0:
            return 0  # Return 0 or None if no values have been added
        return self.total / self.count

class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


def denormalize(x):
    """Reverses the imagenet normalization applied to the input.

    Args:
        x (torch.Tensor - shape(N,3,H,W)): input tensor

    Returns:
        torch.Tensor - shape(N,3,H,W): Denormalized input
    """
    mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    x = x * std + mean
    return torch.clamp(x, max=1.0)


from typing import Dict, Any, List, Optional

class RunningMedian:
    """
    Simple accumulate-then-median helper.
        • append(value)      – store new sample
        • get_value() -> med – median of all samples (or None)
    """
    def __init__(self) -> None:
        self._values: List[float] = []

    def append(self, value: float) -> None:
        self._values.append(float(value))

    def get_value(self) -> Optional[float]:
        n = len(self._values)
        if n == 0:
            return None
        # Copy-and-sort is fine if data volume is modest
        vals = sorted(self._values)
        mid = n // 2
        return vals[mid] if n & 1 else (vals[mid - 1] + vals[mid]) / 2.0


class RunningMedianDict:
    """
    Keeps a list of values for every key, then reports the per-key median.
    Usage pattern matches your original `RunningAverageDict`.
    """
    def __init__(self) -> None:
        self._dict: Optional[Dict[Any, RunningMedian]] = None

    def update(self, new_dict: Optional[Dict[Any, float]]) -> None:
        if new_dict is None:
            return

        if self._dict is None:
            # first call – create trackers for initial keys
            self._dict = {k: RunningMedian() for k in new_dict}

        for key, value in new_dict.items():
            # auto-create tracker for previously unseen keys
            self._dict.setdefault(key, RunningMedian()).append(value)

    def get_value(self) -> Optional[Dict[Any, float]]:
        if self._dict is None:
            return None
        return {k: tracker.get_value() for k, tracker in self._dict.items()}

class RunningAverageDict:
    """A dictionary of running averages."""
    def __init__(self):
        self._dict = None

    def update(self, new_dict):
        if new_dict is None:
            return

        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = RunningAverage()

        for key, value in new_dict.items():
            self._dict[key].append(value)

    def get_value(self):
        if self._dict is None:
            return None
        return {key: value.get_value() for key, value in self._dict.items()}


def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img


def count_parameters(model, include_all=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or include_all)

# import numpy as np

def rmse_silog(predicted, ground_truth):
    """
    Computes the RMSE_silog as described in the provided formula.

    Parameters:
    - predicted: np.array, predicted depth values (N,)
    - ground_truth: np.array, ground truth depth values (N,)

    Returns:
    - rmse_silog: scalar, RMSE_silog error
    """
    # Ensure no division by zero or log of zero
    epsilon = 1e-8
    predicted = np.maximum(predicted, epsilon)
    ground_truth = np.maximum(ground_truth, epsilon)

    # Compute log values
    log_predicted = np.log(predicted)
    log_ground_truth = np.log(ground_truth)

    # Compute alpha
    N = len(predicted)
    alpha = np.sum(log_ground_truth - log_predicted) / N

    # Compute RMSE_silog
    rmse_silog = np.sqrt(np.mean((log_predicted - log_ground_truth + alpha) ** 2))

    return rmse_silog

def compute_errors(gt, pred):
    """Compute depth–prediction metrics.

    Returns a dict with all previous scores plus:
        'mdae' : Median Absolute Error (metres)
        'p90'  : 90-th percentile absolute error (metres)
    """
    if gt.size == 0:
        return None

    # Threshold-based accuracy (δ) -------------------------------------------
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    # Absolute & squared relative -------------------------------------------
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel  = np.mean(((gt - pred) ** 2) / gt)

    # RMSE, RMSE-log, SILog ---------------------------------------------------
    rmse      = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log  = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    silog     = rmse_silog(pred, gt)            # assuming you defined this
    log_10    = np.mean(np.abs(np.log10(gt) - np.log10(pred)))
    mae       = np.mean(np.abs(gt - pred))

    # ----------- NEW: robust, metre-scaled error statistics -----------------
    abs_err = np.abs(gt - pred)
    mdae    = np.median(abs_err)                # Median Absolute Error
    p90     = np.percentile(abs_err, 90)        # 90-th percentile error

    # Inverse-depth variants --------------------------------------------------
    inv_gt   = 1.0 / gt
    inv_pred = 1.0 / pred
    i_rmse   = np.sqrt(np.mean((inv_gt - inv_pred) ** 2))
    i_mae    = np.mean(np.abs(inv_gt - inv_pred))
    i_absrel = np.mean(np.abs(inv_gt - inv_pred) / inv_gt)

    return dict(
        # δ-accuracy
        a1=a1, a2=a2, a3=a3,
        # standard mean metrics
        abs_rel=abs_rel, sq_rel=sq_rel,
        rmse=rmse, rmse_log=rmse_log, log_10=log_10, silog=silog, mae=mae,
        # NEW robust metrics
        mdae=mdae, p90=p90,
        # inverse-depth metrics
        iRMSE=i_rmse, iMAE=i_mae, iAbsRel=i_absrel,
    )

def rescale_mask(old_mask, new_shape):
    old_rows, old_cols = old_mask.shape
    new_rows, new_cols = new_shape
    new_mask = np.zeros(new_shape, dtype=old_mask.dtype)
    
    row_scale = new_rows / old_rows
    col_scale = new_cols / old_cols

    # Loop through all valid (i.e. == 1) points in the old mask.
    for i in range(old_rows):
        for j in range(old_cols):
            if old_mask[i, j] == 1:
                # Compute new indices using scaling factors.
                new_i = int(np.floor(i * row_scale))
                new_j = int(np.floor(j * col_scale))
                
                # Ensure the new indices are within bounds.
                new_i = min(new_i, new_rows - 1)
                new_j = min(new_j, new_cols - 1)
                
                new_mask[new_i, new_j] = 1
                
    return new_mask
    
def evaluation_on_ranges(gt, pred, interpolate=True, evaluation_range = [18.0, 18, 18, 18, 18]):
    """Compute metrics of predicted depth maps. Applies cropping and masking as necessary or specified via arguments. Refer to compute_errors for more details on metrics.
    """

    if gt.shape[-2:] != pred.shape[-2:] and interpolate:
        pred = nn.functional.interpolate(
            pred, gt.shape[-2:], mode='bilinear', align_corners=True)

    gt_depth = gt.squeeze().cpu().numpy()
    
    pred = pred.squeeze().cpu().numpy()
    min_depth_eval = 0.01
    result_dicts = []
    for i in evaluation_range:
        pred_copy = pred.copy()
        pred_copy[pred_copy < min_depth_eval] = min_depth_eval
        pred_copy[np.isinf(pred_copy)] = i
        pred_copy[np.isnan(pred_copy)] = min_depth_eval

        valid_mask = np.logical_and(
            gt_depth > min_depth_eval, gt_depth < i)
        
        dict = compute_errors(gt_depth[valid_mask], pred_copy[valid_mask])
        result_dicts.append(dict)
    return result_dicts
    


#################################### Model uilts ################################################


def parallelize(config, model, find_unused_parameters=True):

    if config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        model = model.cuda(config.gpu)

    config.multigpu = False
    if config.distributed:
        # Use DDP
        config.multigpu = True
        config.rank = config.rank * config.ngpus_per_node + config.gpu
        dist.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                world_size=config.world_size, rank=config.rank)
        config.batch_size = int(config.batch_size / config.ngpus_per_node)
        # config.batch_size = 8
        config.workers = int(
            (config.num_workers + config.ngpus_per_node - 1) / config.ngpus_per_node)
        print("Device", config.gpu, "Rank",  config.rank, "batch size",
              config.batch_size, "Workers", config.workers)
        torch.cuda.set_device(config.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(config.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu], output_device=config.gpu,
                                                          find_unused_parameters=find_unused_parameters)

    elif config.gpu is None:
        # Use DP
        config.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    return model


#################################################################################################


#####################################################################################################


class colors:
    '''Colors class:
    Reset all colors with colors.reset
    Two subclasses fg for foreground and bg for background.
    Use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.green
    Also, the generic bold, disable, underline, reverse, strikethrough,
    and invisible work with the main class
    i.e. colors.bold
    '''
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'

    class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'


def printc(text, color):
    print(f"{color}{text}{colors.reset}")

