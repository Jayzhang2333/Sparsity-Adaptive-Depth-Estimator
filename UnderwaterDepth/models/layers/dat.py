# Modified from: https://github.com/LeapLabTHU/DAT/blob/main/models/dat.py
# Original work licensed under the Apache 2.0 License
# ################Original License Notice###########################################
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------
###############################################################################

# Modifications Copyright (c) 2025 Hongjie Zhang
# Licensed under the MIT License. See LICENSE file in the project root for details.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple
from .dat_blocks import *
from .fusion_layers import BasicBlock, conv_bn_relu


class LayerScale(nn.Module):

    def __init__(self,
                 dim: int,
                 inplace: bool = False,
                 init_values: float = 1e-5):
        super().__init__()
        self.inplace = inplace
        self.weight = nn.Parameter(torch.ones(dim) * init_values)

    def forward(self, x):
        if self.inplace:
            return x.mul_(self.weight.view(-1, 1, 1))
        else:
            return x * self.weight.view(-1, 1, 1)
        

class TransformerStage(nn.Module):
    def __init__(self, fmap_size, dim_in, dim_embed, depths,
                 n_groups, use_pe, heads, stride,
                 offset_range_factor, dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate,
                 use_dwc_mlp, ksize, layer_scale_value, use_lpu, log_cpb):
        """
        Simplified TransformerStage that only supports the 'D' (DAttentionBaseline) branch.

        Args:
            fmap_size (int or tuple): Feature map size.
            dim_in (int): Number of input channels.
            dim_embed (int): Embedding dimension.
            depths (int): Number of layers (or attention-MLP blocks).
            n_groups (int): Number of groups (for attention module).
            use_pe (bool): Whether to use positional encoding.
            heads (int): Number of attention heads.
            stride (int): Stride used in the attention module.
            offset_range_factor (int): Offset range factor for attention.
            dwc_pe (bool): Flag for depthwise conv positional encoding.
            no_off (bool): Flag for disabling offsets.
            fixed_pe (bool): Flag for fixed positional encoding.
            attn_drop (float): Dropout rate for attention.
            proj_drop (float): Dropout rate after projection.
            expansion (int): Expansion ratio in the MLP.
            drop (float): Dropout rate in the MLP.
            drop_path_rate (list[float]): Drop path rates for each layer.
            use_dwc_mlp (bool): Whether to use a conv-based MLP.
            ksize (int): Kernel size for the DAttentionBaseline.
            layer_scale_value (float): Initial layer scaling value.
            use_lpu (bool): Whether to use the Local Perception Unit.
            log_cpb (bool): Flag for using logarithmic CPB.
        """
        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc, "dim_embed must be divisible by heads"
        
        # Projection to embed dimension if necessary.
        self.proj = nn.Conv2d(dim_in, dim_embed, kernel_size=1) if dim_in != dim_embed else nn.Identity()
        
        # Two layer norms per depth (one for attention, one for MLP).
        self.layer_norms = nn.ModuleList([LayerNormProxy(dim_embed) for _ in range(2 * depths)])
        
        # Choose MLP module.
        mlp_fn = TransformerMLPWithConv if use_dwc_mlp else TransformerMLP
        self.mlps = nn.ModuleList([mlp_fn(dim_embed, expansion, drop) for _ in range(depths)])
        
        # Create the attention modules. Since stage_spec is always 'D', we only instantiate DAttentionBaseline.
        self.attns = nn.ModuleList([
            DAttentionBaseline(fmap_size, fmap_size, heads, hc, n_groups,
                               attn_drop, proj_drop, stride, offset_range_factor,
                               use_pe, dwc_pe, no_off, fixed_pe, ksize, log_cpb)
            for _ in range(depths)
        ])
        
        # Create drop path modules.
        self.drop_path = nn.ModuleList([
            DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity()
            for i in range(depths)
        ])
        
        # Layer scaling for both the attention and MLP parts.
        self.layer_scales = nn.ModuleList([
            LayerScale(dim_embed, init_values=layer_scale_value) if layer_scale_value > 0.0 else nn.Identity()
            for _ in range(2 * depths)
        ])
        
        # Local Perception Units if enabled.
        self.use_lpu = use_lpu
        self.local_perception_units = nn.ModuleList([
            nn.Conv2d(dim_embed, dim_embed, kernel_size=3, stride=1, padding=1, groups=dim_embed)
            if use_lpu else nn.Identity() for _ in range(depths)
        ])
    
    def forward(self, x):
        # print(x.shape)
        x = self.proj(x)
        for d in range(self.depths):
            if self.use_lpu:
                residual = x
                x = self.local_perception_units[d](x.contiguous())
                x = x + residual
            # Attention block.
            residual = x
            x, pos, ref = self.attns[d](self.layer_norms[2 * d](x))
            # print(pos.shape)
            # print(ref.shape)
            x = self.layer_scales[2 * d](x)
            x = self.drop_path[d](x) + residual
            
            # MLP block.
            residual = x
            x = self.mlps[d](self.layer_norms[2 * d + 1](x))
            x = self.layer_scales[2 * d + 1](x)
            x = self.drop_path[d](x) + residual
        return x
    


class FillConv(nn.Module):
    def __init__(self, input_guide_channel,input_target_channel, embedding_multiply = 16 ):
        super(FillConv, self).__init__()
        
        embedding_guide_channel = input_guide_channel * embedding_multiply
        embedding_target_channel = input_target_channel * embedding_multiply
        output_channel = embedding_guide_channel + embedding_target_channel
        self.conv_ga = conv_bn_relu(input_guide_channel, embedding_guide_channel, 3, 1, 1, bn=False)
        self.conv_sparse = conv_bn_relu(input_target_channel, 16, 3, 1, 1, bn=False)

        # self._trans = nn.Conv2d(33, 16, 1, 1, 0)
        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuse_conv2 = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.fuse_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fuse_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Sigmoid(),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        for m in self.modules():
           
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
               
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.3)
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0.1)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, ga, sparse):
        f_ga = self.conv_ga(ga)
        f_sparse = self.conv_sparse(sparse)
        f_sparse = torch.cat([f_ga, f_sparse], dim=1)
        f_sparse = self.fuse_conv1(f_sparse) * self.fuse_conv2(f_sparse)
        f = torch.cat([f_sparse, f_ga], dim=1)
        f = f + self.fuse_conv3(f) * self.fuse_conv4(f)

        return f

import matplotlib.pyplot as plt
def display_feature_maps(feature_maps):
    """
    Displays all feature maps in a batch.
    
    Args:
        feature_maps (torch.Tensor): Tensor of shape [B, C, H, W].
    """
    B, C, H, W = feature_maps.shape
    for b in range(B):
        # Create a row of subplots for the b-th sample in the batch
        fig, axes = plt.subplots(1, C, figsize=(C * 2, 2))
        
        # When there's only one channel, axes may not be iterable
        if C == 1:
            axes = [axes]
            
        for c in range(C):
            # Detach the tensor, move it to CPU, and convert to numpy array
            feature_map = feature_maps[b, c].cpu().detach().numpy()
            axes[c].imshow(feature_map, cmap='viridis')
            axes[c].set_title(f'Batch {b} - Channel {c}')
            axes[c].axis('off')
            
        plt.tight_layout()
        plt.show()


class DAT(nn.Module):
    def __init__(self, img_size=(336,448), patch_size=4, expansion=4,
                 dim_stem=96, dims=[96, 128, 256, 512], depths_conv=[1, 1, 2, 2], depths_trans=[1, 1, 2, 2], 
                 heads=[2, 4, 8, 16], stages = 4, conv_ratios = [8,8,8,8],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2, 
                 strides=[8, 4, 2, 1],
                 offset_range_factor=[-1, -1, -1, -1],
                 groups=[1, 2, 4, 8],
                 use_pes=[True, True, True, True], 
                 dwc_pes=[False, False, False, False],
                 lower_lr_kvs={},
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 use_dwc_mlps=[True, True, True, True],
                 use_conv_patches=True,
                 ksizes=[9, 7, 5, 3],
                 layer_scale_values=[-1, -1, -1, -1],
                 use_lpus=[True, True, True, True],
                 log_cpb=[False, False, False, False],
                 **kwargs):
        super().__init__()


        # GA nas sparse scale embedding layers
        self.conv1_ga = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                          bn=False)
        
        self.conv1_scale = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                          bn=False)

        self.conv1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1,
                                      bn=False)

        # --- Patch Embedding Modules ---
        # For the first stage, use patch_proj (as in the original code).
        self.patch_proj = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=patch_size // 2, padding=1),
            LayerNormProxy(64),
            nn.GELU(),
            nn.Conv2d(64, dim_stem, kernel_size=3, stride=patch_size // 2, padding=1),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            nn.Conv2d(3, dim_stem, kernel_size=patch_size, stride=patch_size, padding=0),
            LayerNormProxy(dim_stem)
        )

        # --- Down Projection Modules ---
        # For stages 1,2,3 we use a down projection to both change resolution and adjust channel dimension.
        self.down_projs = nn.ModuleList()
        for i in range(stages-1):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=2, padding=1, bias=False),
                    LayerNormProxy(dims[i+1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2, padding=0, bias=False),
                    LayerNormProxy(dims[i+1])
                )
            )

        # --- Compute drop path rate for each stage ---
        # Here we use one drop path value per stage.
        dpr_ts = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_trans))]

        # --- Build Big Blocks ---
        # Each big block consists of:
        #   [BasicBlock stack] -> [patch embedding] -> [TransformerStage (depth=1)]
        self.big_blocks = nn.ModuleList()
        # The initial patch projection produces a feature map of size img_size // patch_size.
        current_img_size = tuple(x // patch_size for x in img_size)

        # Stage 0: works on the raw image.
        stage0 = nn.Sequential(
            # BasicBlock stack (assumed to work on 3-channel input)
            *[BasicBlock(64, 64, ratio=conv_ratios[0]) for _ in range(depths_conv[0])],
            # Patch embedding for stage 0
            self.patch_proj,
            # TransformerStage: for stage0, project from dim_stem -> dims[0]
            TransformerStage(
                fmap_size=current_img_size,
                dim_in=dim_stem,
                dim_embed=dims[0],
                depths=depths_trans[0],
                n_groups=groups[0],
                use_pe=use_pes[0],
                heads=heads[0],
                stride=strides[0], #this stride is sued in the deformable process in tne attention module
                offset_range_factor=offset_range_factor[0],
                dwc_pe=dwc_pes[0],
                no_off=no_offs[0],
                fixed_pe=fixed_pes[0],
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                expansion=expansion,
                drop=drop_rate,
                drop_path_rate=dpr_ts[sum(depths_trans[:0]):sum(depths_trans[:0 + 1])],
                use_dwc_mlp=use_dwc_mlps[0],
                ksize=ksizes[0],
                layer_scale_value=layer_scale_values[0],
                use_lpu=use_lpus[0],
                log_cpb=log_cpb[0]
            )
        )
        self.big_blocks.append(stage0)
        # current_img_size = current_img_size // 2  # Update resolution for next stage. because the patch embedding happens first, so this // by 2 for the transformer module
        current_img_size = tuple(x // 2 for x in current_img_size)

        # Stages 1, 2, 3: process feature maps from the previous stage.
        for i in range(1, stages):
            basic_in = dims[i - 1]  # input channels from previous stage.
            stage = nn.Sequential(
                # BasicBlock stack (assumed to work on feature maps with basic_in channels)
                *[BasicBlock(basic_in, basic_in, ratio=conv_ratios[i]) for _ in range(depths_conv[i])],
                # Down projection for patch embedding
                self.down_projs[i - 1],
                # TransformerStage: for later stages, project from basic_in*2 -> dims[i]
                TransformerStage(
                    fmap_size=current_img_size,
                    # dim_in=basic_in * 2, # don't really knwo why it needs to be like this
                    dim_in = dims[i],
                    dim_embed=dims[i],
                    depths=depths_trans[i],
                    n_groups=groups[i],
                    use_pe=use_pes[i],
                    heads=heads[i],
                    stride=strides[i],
                    offset_range_factor=offset_range_factor[i],
                    dwc_pe=dwc_pes[i],
                    no_off=no_offs[i],
                    fixed_pe=fixed_pes[i],
                    attn_drop=attn_drop_rate,
                    proj_drop=drop_rate,
                    expansion=expansion,
                    drop=drop_rate,
                    drop_path_rate=dpr_ts[sum(depths_trans[:i]):sum(depths_trans[:i + 1])],
                    use_dwc_mlp=use_dwc_mlps[i],
                    ksize=ksizes[i],
                    layer_scale_value=layer_scale_values[i],
                    use_lpu=use_lpus[i],
                    log_cpb=log_cpb[i]
                )
            )
            self.big_blocks.append(stage)
            # current_img_size = current_img_size // 2
            current_img_size = tuple(math.ceil(x / 2) for x in current_img_size)
            

        self.lower_lr_kvs = lower_lr_kvs
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, ga, scale, featrues):
        ga_embedding = self.conv1_ga(ga)
        scale_embedding = self.conv1_scale(scale)
        x = torch.cat((ga_embedding, scale_embedding, featrues), dim=1)
        x = self.conv1(x)
        outputs = []
        for stage in self.big_blocks:
            x = stage(x)
            outputs.append(x)
        return outputs