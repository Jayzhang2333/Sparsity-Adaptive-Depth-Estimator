# Copyright (c) 2025 Hongjie Zhang
# Licensed under the MIT License. See LICENSE file in the project root for details.

import torch
from UnderwaterDepth.models.DepthAnything.dpt import DPTHeadCustomised
from UnderwaterDepth.models.DepthAnything.dinov2 import DINOv2
from UnderwaterDepth.models.depth_model import DepthModel
from UnderwaterDepth.models.ScaleMapLearner.sml import SMLDeformableAttention
from UnderwaterDepth.models.model_io import load_state_from_resource
import os
from UnderwaterDepth.models.layers.global_alignment import GAandFilter



model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384], 'layer_idxs': [2, 5, 8, 11]},
    }

    
class SPADE(DepthModel):
    patch_size = 14  # patch size of the pretrained dinov2 model
    use_bn = False
    use_clstoken = False
    output_act = 'identity'

    def __init__(self,
                 **kwargs):
        super().__init__()

        self.min_pred = kwargs['min_pred']
        self.max_pred = kwargs['max_pred']
        self.ga_max = kwargs['ga_max']
        self.biliteral_kernal_size = kwargs['biliteral_kernal_size']
        self.min_pred_inv = 1.0/self.min_pred
        self.max_pred_inv = 1.0/self.max_pred
        self.model_config = model_configs[kwargs['da_model_type']]
        self.encoder = kwargs['da_model_type']
        self.sigma_s = kwargs['sigma_s']
        self.sigma_r = kwargs['sigma_r']
        self.image_size = kwargs['img_size']
        self.dat_patch_size = kwargs['dat_patch_size']
        self.dat_stem_features = kwargs['dat_stem_features']
        self.dat_dims = kwargs['dat_dims']
        self.dat_depths_conv = kwargs['dat_depths_conv']
        self.dat_depth_trans = kwargs['dat_depth_trans']
        self.dat_stages = kwargs['dat_stages']
        
        self.pretrained = DINOv2(model_name=kwargs['da_model_type'])
        # Frozen DINOv2
        for param in self.pretrained.parameters():
            param.requires_grad = False

        dim = self.pretrained.blocks[0].attn.qkv.in_features
        self.depth_head = DPTHeadCustomised(in_channels=dim,
                                  features=self.model_config['features'],
                                  out_channels=self.model_config['out_channels'],
                                  use_bn=self.use_bn,
                                  use_clstoken=self.use_clstoken,
                                  output_act=self.output_act)
        # Frozen DepthAnything Decoder
        for name, param in self.depth_head.named_parameters():
                param.requires_grad = False

        self.scale_map_learner = SMLDeformableAttention(self.image_size, self.dat_patch_size, self.dat_stem_features,
                                                        self.dat_dims,self.dat_depths_conv, self.dat_depth_trans, self.dat_stages, 
            features=self.model_config['features'], non_negative=False, channels_last=False, align_corners=True,
        blocks={'expand': True}, min_pred=self.min_pred, max_pred=self.max_pred)

        self.ga_and_filter = GAandFilter(max_pred=self.max_pred, min_pred=self.min_pred, clamp_min=self.min_pred, clamp_max=self.ga_max,sigma_s=self.sigma_s, sigma_r=self.sigma_r, kernel_size=self.biliteral_kernal_size )
       
        # load weights for DepthAnything
        self.load_checkpoint_depthanything(kwargs['da_pretrained_resource'])

    def load_checkpoint(self, ckpt_path):
        if os.path.exists(ckpt_path):
            print(f'Loading checkpoint from {ckpt_path}')
            checkpoint = torch.load(ckpt_path, map_location='cuda')
            self.load_state_dict(checkpoint, strict=False)
        else:
            print(f'Checkpoint {ckpt_path} not found')

    def load_checkpoint_depthanything(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print(f'Checkpoint {ckpt_path} not found')
            return

        print(f'Loading checkpoint from {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=True)
        # if your checkpoint is wrapped, e.g. {'state_dict': ...}
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']

        # Partition into two dicts
        pretrained_sd   = {}
        depth_head_sd   = {}
        for full_key, v in ckpt.items():
            if full_key.startswith('pretrained.'):
                # drop the 'pretrained.' prefix
                sub_key = full_key[len('pretrained.'):]
                pretrained_sd[sub_key] = v
            elif full_key.startswith('depth_head.'):
                sub_key = full_key[len('depth_head.'):]
                depth_head_sd[sub_key] = v

        # Load them separately
        if pretrained_sd:
            self.pretrained.load_state_dict(pretrained_sd, strict=True)
            print(f'  → loaded {len(pretrained_sd)} weights into pretrained')
        else:
            print('  → no pretrained weights found in checkpoint')

        if depth_head_sd:
            self.depth_head.load_state_dict(depth_head_sd, strict=True)
            print(f'  → loaded {len(depth_head_sd)} weights into depth_head')
        else:
            print('  → no depth_head weights found in checkpoint')

    def forward(self, x, prompt_depth, fx =None, cx = None):
        assert prompt_depth is not None, 'prompt_depth is required'
        h, w = x.shape[-2:]
        features = self.pretrained.get_intermediate_layers(x, self.model_config['layer_idxs'],return_class_token=True)
        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        rel_depth, extracted_features = self.depth_head(features, patch_h, patch_w, prompt_depth)
       
        dense_residual, ga_result, d = self.ga_and_filter(rel_depth, prompt_depth, fx = fx, cx = cx)
        pred, scales = self.scale_map_learner(ga_result, dense_residual, d, extracted_features)
       
        output = dict(metric_depth=1.0/pred)
        output['inverse_depth'] = pred
        
        return output

    @torch.no_grad()
    def predict(self,
                image: torch.Tensor,
                prompt_depth: torch.Tensor):
        return self.forward(image, prompt_depth)

    def get_lr_params(self, lr):
        param_conf = []
        trainable = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)
                trainable.append(param)
        if trainable:
            param_conf.append({
                'params': trainable,
                'lr':      lr,
                'name':    'trainable parameters'
            })
        return param_conf

    @staticmethod
    def build(**kwargs):
        
        model = SPADE(**kwargs)
        if kwargs['pretrained_resource']:
            assert isinstance(kwargs['pretrained_resource'], str), "pretrained_resource must be a string"
            model = load_state_from_resource(model, kwargs['pretrained_resource'])

        return model

    @staticmethod
    def build_from_config(config):
        return SPADE.build(**config)
