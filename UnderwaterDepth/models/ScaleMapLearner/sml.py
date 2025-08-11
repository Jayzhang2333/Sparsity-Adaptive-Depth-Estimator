# Copyright (c) 2025 Hongjie Zhang
# Licensed under the MIT License. See LICENSE file in the project root for details.

import torch.nn as nn
import torch
from torch.nn import functional as F
from .base_model import BaseModel
from .blocks import FeatureFusionBlock_custom_original, OutputConv, ProgressiveFusionMulti, _make_encoder_DAT


class SMLDeformableAttention(BaseModel):
    """Network for monocular depth estimation.
    """

    def __init__(self, img_size, dat_patch_size, dat_stem_features, dims, depths_conv,
                 depths_trans, stages, features=64, dino_fusion_features = 32, dino_feature_num = 4, non_negative=False, channels_last=False, align_corners=True,
        blocks={'expand': True}, min_pred=None, max_pred=None):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 64.
            backbone (str, optional): Backbone network for encoder. Defaults to efficientnet_lite3.
        """
        super(SMLDeformableAttention, self).__init__()
                
        self.channels_last = channels_last
        self.blocks = blocks

        self.groups = 1

        # for model output
        self.min_pred = min_pred
        self.max_pred = max_pred

        features1=features
        features2=features
        features3=features
        features4=features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1=features
            features2=features*2
            features3=features*4
            features4=features*8 

        self.encoder, self.scratch = _make_encoder_DAT( features, img_size = img_size, 
                                                       patch_size = dat_patch_size, dat_stem_features = dat_stem_features,
                                                       dims = dims, depths_conv = depths_conv,
                                                       depths_trans = depths_trans, stages = stages, 
                                                         groups=self.groups, expand=self.expand)

        self.multi_feature_fusion = ProgressiveFusionMulti(dino_feature_num, features, dino_fusion_features)

        self.scratch.activation = nn.ReLU(False)    

        self.scratch.refinenet4 = FeatureFusionBlock_custom_original(features4, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom_original(features3, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom_original(features2, self.scratch.activation, deconv=False, bn=False, expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom_original(features1, self.scratch.activation, deconv=False, bn=False, align_corners=align_corners)

        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1, groups=self.groups)
        self.scratch.output_conv = OutputConv(features // 2, self.scratch.activation, non_negative)
        


    def forward(self, ga, scale, d, features):
        """Forward pass.

        Args:
            x (tensor): input data (image)
            d (tensor): unalterated input depth

        Returns:
            tensor: depth
        """
        
        fused_feature = self.multi_feature_fusion(features, ga.shape[2:])
        
        layer_1, layer_2, layer_3, layer_4 = self.encoder(ga, scale, fused_feature)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out_feature = self.scratch.output_conv1(path_1)
        out_feature = F.interpolate(out_feature, ga.shape[2:], mode="bilinear", align_corners=True)
        
        out = self.scratch.output_conv(out_feature)
        
        scales = F.relu(1.0 + out)
        pred = d * scales

        if self.min_pred is not None:
            min_pred_inv = 1.0 / self.min_pred
            pred = torch.where(pred > min_pred_inv,
                            torch.full_like(pred, min_pred_inv),
                            pred)

        if self.max_pred is not None:
            max_pred_inv = 1.0 / self.max_pred
            pred = torch.where(pred < max_pred_inv,
                            torch.full_like(pred, max_pred_inv),
                            pred)

        return (pred, scales)
    
    