# Modified from: https://github.com/isl-org/ZoeDepth/blob/main/evaluate.py
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
###############################################################################

# Modifications Copyright (c) 2025 Hongjie Zhang
# Licensed under the MIT License. See LICENSE file in the project root for details.
import argparse
from typing import Dict, List, Optional, Sequence
import os

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import numpy as np
from UnderwaterDepth.data.data_mono import DepthDataLoader
from UnderwaterDepth.models.builder import build_model
from UnderwaterDepth.utils.arg_utils import parse_unknown
from UnderwaterDepth.utils.config import get_config
from UnderwaterDepth.utils.misc import (
    RunningAverageDict,
    colors,
    evaluation_on_ranges,
    count_parameters,
    denormalize
)

def show_images_with_sparse_overlay(
    source_rgb,
    pred_depth,
    sparse_depth,
    *,
    vmin=None,
    vmax=None,
    point_size=5,
    cmap="viridis_r",
):
    def prep(img):
        if hasattr(img, 'detach'):
            img = img.detach().cpu().numpy()
        # Convert NCHW to NHWC
        if img.ndim == 4 and img.shape[1] in [1,3]:
            img = np.transpose(img, (0,2,3,1))
        return img
    
    rgb = prep(source_rgb)
    depth = prep(pred_depth)
    sparse_depth = prep(sparse_depth)
    
    batch = rgb.shape[0]
    assert sparse_depth.shape[:3] == (batch, rgb.shape[1], rgb.shape[2])
    # Figure layout: 3 columns (RGB+overlay | depth | colorbar) per batch row
    fig = plt.figure(figsize=(12, 4 * batch))
    gs = gridspec.GridSpec(
        nrows=batch,
        ncols=3,
        width_ratios=[1, 1, 0.05],
        wspace=0.05,
        hspace=0.05,
    )

    for i in range(batch):
        # Left: RGB with sparse points overlaid (colored by sparse depth)
        ax1 = fig.add_subplot(gs[i,0])
        ax1.imshow(rgb[i])
        # overlay sparse points
        sd = sparse_depth[i,...,0]
        mask = ~np.isnan(sd) & (sd != 0)
        ys, xs = np.where(mask)
        vals = sd[mask]
        ax1.scatter(xs, ys, c=vals, cmap=cmap, vmin=vmin, vmax=vmax, s=point_size)
        ax1.set_title("RGB + sparse points")
        ax1.axis('off')

        # Middle: predicted depth
        ax2 = fig.add_subplot(gs[i, 1])
        im = ax2.imshow(depth[i], cmap=cmap, vmin=vmin, vmax=vmax)
        ax2.set_title("Predicted depth")
        ax2.axis("off")

        # Right: colorbar for this row's depth
        cax = fig.add_subplot(gs[i, 2])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=9)
        cb.set_label("Depth (m)", fontsize=10, fontweight="bold")

    plt.show()
    return fig


def save_images_with_sparse_overlay(
    source_rgb,
    pred_depth,
    sparse_depth,
    output_dir: str,
    image_name: str,
    *,
    vmin=None,
    vmax=None,
    point_size=5,
    cmap="viridis_r",
    dpi=200,
):
    def prep(img):
        if hasattr(img, 'detach'):
            img = img.detach().cpu().numpy()
        # Convert NCHW to NHWC
        if img.ndim == 4 and img.shape[1] in [1,3]:
            img = np.transpose(img, (0,2,3,1))
        return img
    
    rgb = prep(source_rgb)
    depth = prep(pred_depth)
    sparse_depth = prep(sparse_depth)
    
    batch = rgb.shape[0]
    assert sparse_depth.shape[:3] == (batch, rgb.shape[1], rgb.shape[2])
    # Figure layout: 3 columns (RGB+overlay | depth | colorbar) per batch row
    fig = plt.figure(figsize=(12, 4 * batch))
    gs = gridspec.GridSpec(
        nrows=batch,
        ncols=3,
        width_ratios=[1, 1, 0.05],
        wspace=0.05,
        hspace=0.05,
    )

    for i in range(batch):
        # Left: RGB with sparse points overlaid (colored by sparse depth)
        ax1 = fig.add_subplot(gs[i,0])
        ax1.imshow(rgb[i])
        # overlay sparse points
        sd = sparse_depth[i,...,0]
        mask = ~np.isnan(sd) & (sd != 0)
        ys, xs = np.where(mask)
        vals = sd[mask]
        ax1.scatter(xs, ys, c=vals, cmap=cmap, vmin=vmin, vmax=vmax, s=point_size)
        ax1.set_title("RGB + sparse points")
        ax1.axis('off')

        # Middle: predicted depth
        ax2 = fig.add_subplot(gs[i, 1])
        im = ax2.imshow(depth[i], cmap=cmap, vmin=vmin, vmax=vmax)
        ax2.set_title("Predicted depth")
        ax2.axis("off")

        # Right: colorbar for this row's depth
        cax = fig.add_subplot(gs[i, 2])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=9)
        cb.set_label("Depth (m)", fontsize=10, fontweight="bold")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, image_name)
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return out_path


@torch.no_grad()
def infer(
    model,
    images: torch.Tensor,
    sparse_features: torch.Tensor,
    cx: Optional[torch.Tensor],
    fx: Optional[torch.Tensor],
) -> torch.Tensor:
    """Forward pass with optional prompt depth and intrinsics."""
    pred = model(
        images,
        prompt_depth=sparse_features,
        fx=fx,
        cx=cx,
    )
    return pred["metric_depth"]


def evaluate(
    model,
    test_loader,
    config,
    multi_range_evaluation: Sequence[float],
    round_vals: bool = True,
    round_precision: int = 3,
    visualize: bool = False,             # <- already added
    save_image: bool = False,            # <- new
    output_image_path: Optional[str] = None,  # <- new (can be a pattern)
) -> List[Dict[str, float]]:
    """Evaluate model over multiple ranges and return aggregated metrics."""
    model.eval()

    # One accumulator per evaluation range
    metrics_list = [RunningAverageDict() for _ in multi_range_evaluation]

    for sample in tqdm(test_loader, total=len(test_loader)):
        # Skip if no valid depth annotation
        if sample.get("has_valid_depth") is False:
            print("no valid depth")
            continue

        # Move data to GPU
        image = sample["image"].cuda()
        depth = sample["depth"].cuda()
        sparse_map = sample["sparse_map"].cuda().float()

        # Mask out invalid sparse points
        valid_mask = (sparse_map >= config.min_depth) & (sparse_map <= config.max_depth)
        if valid_mask.sum().item() < 1:
            print("less than 1 depth points")
            continue

        # Mask out invalid ground-truth depths
        gt_mask = (depth >= config.min_depth) & (depth <= config.max_depth)
        if gt_mask.sum().item() == 0:
            print("no valid gt")
            continue

        # Ensure depth is (1, 1, H, W)
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)

        # Intrinsics if laser scaler is used
        if sample.get("laser_scaler", False):
            fx = sample["fx"].cuda()
            cx = sample["cx"].cuda()
        else:
            fx = cx = None

        # Inference
        pred = infer(model, image, sparse_map, cx=cx, fx=fx)

        image_denorm = denormalize(image)

        if save_image:
            image_path = sample['image_path'][0]
            base = os.path.basename(image_path)
            name, _ = os.path.splitext(base)
            png_name = name + '.png'
            save_images_with_sparse_overlay(image_denorm, pred, sparse_map,output_image_path, png_name)
            # breakpoint()
        
        if visualize:
            show_images_with_sparse_overlay(image_denorm, pred, sparse_map, vmax = 10, cmap='turbo_r')
        
        # Compute metrics per range
        result = evaluation_on_ranges(
            depth, pred, evaluation_range=multi_range_evaluation
        )
        if result is None:
            continue

        # Update accumulators
        for idx, acc in enumerate(metrics_list):
            acc.update(result[idx])

    def _round(val: float) -> float:
        return round(val, round_precision) if round_vals else val

    # Finalize and round if requested
    final: List[Dict[str, float]] = []
    for acc in metrics_list:
        vals = acc.get_value() or {}
        final.append({k: _round(v) for k, v in vals.items()})
    return final


def main(
    config,
    ranges: Sequence[float],
    visualize: bool = False,
    save_image: bool = False,
    output_image_path: Optional[str] = None,
) -> List[Dict[str, float]]:
    """Build model/dataloader, evaluate over user-specified ranges, print and return metrics."""
    model = build_model(config).cuda()
    test_loader = DepthDataLoader(config, "eval").data

    # Evaluate over the provided ranges
    metrics_list = evaluate(model, test_loader, config, ranges, visualize=visualize, save_image=save_image,output_image_path=output_image_path )

    # Print colored results
    print(colors.fg.green)
    for r, metrics in zip(ranges, metrics_list):
        print(f"Range 0.01 - {r}")
        print("Mean")
        print(metrics)
        print()
    print(colors.reset)

    # Append model parameter count
    param_str = f"{round(count_parameters(model, include_all=True) / 1e6, 2)}M"
    for m in metrics_list:
        m["#params"] = param_str

    return metrics_list


def eval_model(
    model_name: str,
    pretrained_resource: Optional[str],
    dataset: str = "nyu",
    ranges: Optional[Sequence[float]] = None,
    visualize: bool = False,
    save_image: bool = False,                 # <- added
    output_image_path: Optional[str] = None,  # <- added
    **kwargs,
) -> List[Dict[str, float]]:
    """Load eval config, print status, and run main() with the specified ranges."""
    overrides = {k: v for k, v in kwargs.items()}
    config = get_config(
        model_name, "eval", dataset, **overrides, pretrained_resource=pretrained_resource
    )

    print(f"Evaluating {model_name} on {dataset}...")
    return main(config, ranges or [10, 5, 2], visualize=visualize, save_image=save_image, output_image_path=output_image_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate underwater depth estimation model"
    )
    parser.add_argument("-m", "--model", required=True, help="Name of the model to evaluate")
    parser.add_argument(
        "-p",
        "--pretrained_resource",
        default=None,
        help="Override pretrained resource for model weights",
    )
    parser.add_argument(
        "-d", "--dataset", default="nyu", help='Dataset to evaluate on (e.g., "nyu")'
    )
    parser.add_argument(
        "-r",
        "--ranges",
        type=float,
        nargs="+",
        default=[10, 5, 2],
        help="List of evaluation ranges in meters",
    )
    parser.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="Display plots interactively.",
    )

    parser.add_argument(
        "--save-image",
        action="store_true",
        help="Save visualization image(s) to disk.",
    )
    parser.add_argument(
        "--output-image-path",
        type=str,
        default="evaluation.png",
        help='Path to save visualization (e.g., "out.png" or "viz_{idx}.png").',
    )

    args, unknown_args = parser.parse_known_args()
    extra_kwargs = parse_unknown(unknown_args)

    eval_model(
        model_name=args.model,
        pretrained_resource=args.pretrained_resource,
        dataset=args.dataset,
        ranges=args.ranges,
        visualize = args.visualize,
        save_image = args.save_image,
        output_image_path = args.output_image_path,
        **extra_kwargs,
    )
