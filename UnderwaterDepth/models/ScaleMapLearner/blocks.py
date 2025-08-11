# Modified from: https://github.com/isl-org/VI-Depth/blob/main/modules/midas/base_model.py
# Original work licensed under the MIT License
# Copyright (c) 2023 Intelligent Systems Lab Org
# Modifications Copyright (c) 2025 Hongjie Zhang
import torch
import torch.nn as nn
# from UnderwaterDepth.models.layers.fusion_layers import conv_bn_relu
import torch.nn.functional as F
from UnderwaterDepth.models.layers.dat import DAT


def _make_encoder_DAT( features, img_size,patch_size,dat_stem_features, dims, depths_conv,depths_trans,stages, groups=1, expand=False):
    
    encoder = DAT(img_size=img_size,patch_size=patch_size, dim_stem=dat_stem_features, dims=dims, depths_conv=depths_conv, 
                  depths_trans=depths_trans, stages = stages)
    scratch = _make_scratch(dims, features, groups=groups, expand=expand)  # efficientnet_lite3     
    
    return encoder, scratch

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand==True:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )

    return scratch

class ProgressiveFusionMulti(nn.Module):
    def __init__(self, num_features, channels, final_dim = None ):
        """
        Args:
            num_features (int): Number of feature maps to fuse.
            channels (int): Number of channels in each feature map.
        """
        super(ProgressiveFusionMulti, self).__init__()
        # We need (num_features - 1) fusion stages.
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(channels * 2, channels, kernel_size=1)
            for _ in range(num_features - 1)
        ])
        # self.final_size = final_size
        self.final_dim = final_dim
        
        if final_dim is not None:
            self.extra_conv = nn.Conv2d(channels, final_dim, kernel_size=3,padding=1)
        else:
            self.extra_conv = nn.Conv2d(channels, channels, kernel_size=3,padding=1)
       

    def forward(self, features, final_size = None):
        """
        Args:
            features (list of Tensors): A list of feature maps ordered from highest to lowest resolution.
                                         Each tensor has shape [B, C, H, W] (different H and W).
        Returns:
            Tensor: The fused feature map at the highest resolution with shape [B, C, H_high, W_high].
        """
        # Start with the lowest resolution feature map.
        fused = features[-1]
        fusion_idx = 0

        # Fuse progressively from lower resolution to higher resolution.
        # Iterate from the second-to-last feature to the first.
        for i in range(len(features) - 2, -1, -1):
            # Upsample the current fused map to the resolution of the next higher feature.
            fused = F.interpolate(fused, size=features[i].shape[2:], mode='bilinear', align_corners=False)
            # Concatenate along the channel dimension.
            fused = torch.cat([features[i], fused], dim=1)
            # Fuse via a 1x1 conv to maintain the number of channels.
            fused = self.fusion_convs[fusion_idx](fused)
            fusion_idx += 1

        if final_size is not None:
            fused = F.interpolate(fused, size=final_size, mode='bilinear', align_corners=False)

        if self.final_dim is not None:    
            fused = self.extra_conv(fused)
           

        return fused

class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom_original(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom_original, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs, size = None):
        """Forward pass.

        Returns:
            tensor: output
        """

        if (size is None):
            modifier = {"scale_factor": 2}
        else:
            modifier = {"size": size}

        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        
        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output

    
class OutputConv(nn.Module):
    """Output conv block.
    """

    def __init__(self, features, activation, non_negative):

        super(OutputConv, self).__init__()

        self.output_conv = nn.Sequential(
            nn.Conv2d(features, 32, kernel_size=3, stride=1, padding=1),
            activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

    def forward(self, x):        
        return self.output_conv(x)
    
class DepthUncertaintyHead(nn.Module):
    def __init__(self, features, groups, activation, non_negative):
        super(DepthUncertaintyHead, self).__init__()
        
        # Head for the per-pixel scale correction factor (depth output)
        self.depth_head = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=groups),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            # activation, # originally, this is an input activation function, not the leakyrelu below
            nn.LeakyReLU(0.2, inplace = False),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )
        
        # Head for the uncertainty map output
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1, groups=groups),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Softplus()  # alternative to ReLU; outputs are strictly positive
        )
    
    def forward(self, x):
        depth = self.depth_head(x)
        uncertainty = self.uncertainty_head(x)
        return depth, uncertainty

def weights_init(m):
    import math
    # initialize from normal (Gaussian) distribution
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    

