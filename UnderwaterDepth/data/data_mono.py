# Modified from: https://github.com/isl-org/ZoeDepth/blob/main/zoedepth/data/data_mono.py
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
import itertools
import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .transforms import *


def generate_feature_map_for_ga(feature_fp, original_height=480, original_width=640, new_height=336, new_width=448, inverse_depth = False):
    # Check the file extension to determine the delimiter
    file_extension = os.path.splitext(feature_fp)[-1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(feature_fp)
    elif file_extension == '.txt':
        df = pd.read_csv(feature_fp, delimiter=' ', header=0, names=['row', 'column', 'depth'])
    else:
        raise ValueError("Unsupported file format. Only CSV and TXT files are supported.")
    
    # Initialize a blank depth map for the new image size with zeros
    sparse_depth_map = np.full((new_height, new_width), 0.0, dtype=np.float32)

    # Calculate scaling factors
    scale_y = new_height / original_height
    scale_x = new_width / original_width

    # Iterate through the dataframe and populate the depth map with scaled coordinates
    for index, row in df.iterrows():
        # Scale pixel coordinates to new image size
        pixel_row = int(row['row'] * scale_y)
        pixel_col = int(row.get('column', row.get('col', 0)) * scale_x)
        
        depth_value = float(row['depth'])
        if inverse_depth:
            depth_value = 1.0/depth_value

        # Ensure the scaled coordinates are within the bounds of the new image size
        if 0 <= pixel_row < new_height and 0 <= pixel_col < new_width:
            
            sparse_depth_map[pixel_row, pixel_col] = depth_value

    sparse_depth_map = sparse_depth_map[..., np.newaxis]
    return sparse_depth_map


def preprocessing_transforms(mode, **kwargs):
    return transforms.Compose([
        ToTensor(mode=mode,  **kwargs)
    ])


class DepthDataLoader(object):
    def __init__(self, config, mode, device='cpu', transform=None, **kwargs):

        self.config = config

        img_size = self.config.get("img_size", None)
        img_size = img_size if self.config.get(
            "do_input_resize", False) else None

        if transform is None:
            transform = preprocessing_transforms(mode, size=img_size)
     
        if mode == 'eval':
            self.testing_samples = DataLoadPreprocess(
                config, mode, transform=transform)
            self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=kwargs.get("shuffle_test", True),
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler=self.eval_sampler)
        else:
            print(
                'mode should be one of \'train, test, eval\'. Got {}'.format(mode))


def repetitive_roundrobin(*iterables):
    # Repetitive roundrobin
    iterables_ = [iter(it) for it in iterables]
    exhausted = [False] * len(iterables)
    while not all(exhausted):
        for i, it in enumerate(iterables_):
            try:
                yield next(it)
            except StopIteration:
                exhausted[i] = True
                iterables_[i] = itertools.cycle(iterables[i])
                # First elements may get repeated if one iterable is shorter than the others
                yield next(iterables_[i])


class RepetitiveRoundRobinDataLoader(object):
    def __init__(self, *dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        return repetitive_roundrobin(*self.dataloaders)

    def __len__(self):
        # First samples get repeated, thats why the plus one
        return len(self.dataloaders) * (max(len(dl) for dl in self.dataloaders) + 1)


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


class CachedReader:
    def __init__(self, shared_dict=None):
        if shared_dict:
            self._cache = shared_dict
        else:
            self._cache = {}

    def open(self, fpath):
        im = self._cache.get(fpath, None)
        if im is None:
            im = self._cache[fpath] = Image.open(fpath)
        return im


class ImReader:
    def __init__(self):
        pass

    # @cache
    def open(self, fpath):
        return Image.open(fpath)


class DataLoadPreprocess(Dataset):
    def __init__(self, config, mode, transform=None, is_for_eval=False, **kwargs):
        self.config = config
        if mode == 'eval':
            with open(config.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(config.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.is_for_eval = is_for_eval
        if config.use_shared_dict:
            self.reader = CachedReader(config.shared_dict)
        else:
            self.reader = ImReader()

    def postprocess(self, sample):
        return sample

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        if len(sample_path.split()) > 3:
            fx = float(sample_path.split()[3])
            cx = float(sample_path.split()[4])
       
        sample = {}

        if self.mode == 'eval':
            data_path = self.config.data_path_eval

        image_path = os.path.join(
            data_path, remove_leading_slash(sample_path.split()[0]))
        
        feature_path = os.path.join(
            data_path, remove_leading_slash(sample_path.split()[2]))
        
        image = np.asarray(self.reader.open(image_path),
                        dtype=np.float32) / 255.0
        
        laser_scaler = False
        if self.config.laser_scaler:
            laser_scaler = True
        
        if (feature_path.lower().endswith('.csv') or feature_path.lower().endswith('.txt')):
            inverse_depth = False
            
            sparse_feature_map = generate_feature_map_for_ga(feature_path, original_height=self.config.sparse_feature_height, original_width=self.config.sparse_feature_width, new_height=self.config.img_size[0], new_width=self.config.img_size[1], inverse_depth=inverse_depth)
            sparse_feature_map = sparse_feature_map
        
        else:
            raise TypeError("Input Sparse Depth must be a csv or txt file")
        
        if self.mode == 'eval':
            gt_path = self.config.gt_path_eval
            depth_path = os.path.join(
                gt_path, remove_leading_slash(sample_path.split()[1]))
            has_valid_depth = False
            try:
                if depth_path.endswith('.npy'):
                    depth_gt = np.load(depth_path)
                else:
                    depth_gt = self.reader.open(depth_path)
                has_valid_depth = True
            except IOError:
                depth_gt = False
                print('Missing gt for {}'.format(image_path))

            if has_valid_depth:
                depth_gt = np.asarray(depth_gt, dtype=np.float32)
                depth_gt = np.expand_dims(depth_gt, axis=2)
                depth_gt = depth_gt /1.0

                mask = np.logical_and(
                    depth_gt >= self.config.min_depth, depth_gt <= self.config.max_depth).squeeze()[None, ...]
                
            else:
                mask = False
        
        sample = {'image': image,'sparse_map': sparse_feature_map, 'depth': depth_gt, 'has_valid_depth': has_valid_depth,
                    'image_path': sample_path.split()[0], 'feature_path': sample_path.split()[-1],'depth_path': sample_path.split()[1],
                    'mask': mask, 'laser_scaler': laser_scaler}
        if laser_scaler:
            sample['fx'] = fx
            sample['cx'] = cx

        if ('has_valid_depth' in sample and sample['has_valid_depth']):
            sample['mask'] = mask

        if self.transform:
            sample = self.transform(sample)

        sample = self.postprocess(sample)
        sample['dataset'] = self.config.dataset
        
        return sample

    def __len__(self):
        return len(self.filenames)

        
