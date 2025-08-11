# Copyright (c) 2025 Hongjie Zhang
# Licensed under the MIT License. See LICENSE file in the project root for details.
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class ToTensor(object):
    def __init__(self, mode, do_normalize=True, size=None):
        self.mode = mode
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if do_normalize else nn.Identity()
        self.size = size
        if size is not None:
            self.resize = transforms.Resize(size=size,  antialias=False)
        else:
            self.resize = nn.Identity()

    def __call__(self, sample):
        image, sparse_feature_map = sample['image'],sample['sparse_map']
        image = self.to_tensor(image)
        image = self.normalize(image)
        image = self.resize(image)
      
        sparse_feature_map = self.to_tensor(sparse_feature_map)

        if self.mode == 'test':
            return {'image': image, 'sparse_map':sparse_feature_map}

        depth = sample['depth']
      
        return {**sample, 'image': image, 'depth': depth, 'sparse_map':sparse_feature_map,
                    'image_path': sample['image_path'], 'depth_path': sample['depth_path'], 'depth_path': sample['depth_path'], 'feature_path':sample['feature_path']}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img