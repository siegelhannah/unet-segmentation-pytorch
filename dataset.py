"""
Modified dataset class for wildfire prediction:
One input sample consists of a stacked multiband image with all features (18),
and a ground-truth fire mask raster.
"""

import os
import random

import numpy as np
import torch
from skimage.io import imread
import rasterio
from torch.utils.data import Dataset

# from utils import crop_sample, pad_sample, resize_sample, normalize_volume
from utils import center_crop_sample # for making images smaller by cropping to a smaller box

from sklearn.model_selection import train_test_split



class WildfireDataset(Dataset):
    in_channels=18 # 18 features
    out_channels=1 # singleband fire mask

    def __init__(
        self, 
        data_dir,       # dir for all data- stacked files and masks
        image_size=256, 
        transform=None, 
        crop_size=None, 
        subset="train", 
        seed=42):

        assert subset in ["all", "train", "validation"]

        self.transform = transform
        self.image_size = image_size
        self.subset = subset

        # construct paths based on subset:
        if subset == "all":
            # if using all, assume data dir contains stacked/ and fire_final/
            stacked_dir = os.path.join(data_dir, 'stacked')
            masks_dir = os.path.join(data_dir, 'fire_final')
        else:
            # if using ALREADY-SEPARATED train/val, expect pre-split directory structure
            subset_dir = os.path.join(data_dir, subset)
            stacked_dir = os.path.join(subset_dir, 'stacked')
            masks_dir = os.path.join(subset_dir, 'fire_final')

        # load all data
        self.samples=[]
        # sort so inputs/outputs are in the same order based on date & coords
        stacked_files = sorted([f for f in os.listdir(stacked_dir) if f.endswith('.tif')])
        mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.tif')])

        assert len(self.image_paths) == len(self.mask_paths), "Mismatch between number of images and masks"

        # store file paths for loading as we go
        self.sample_paths = []
        for stacked_file, mask_file in zip(stacked_files, mask_files):
            stacked_path = os.path.join(stacked_dir, stacked_file)
            mask_path = os.path.join(masks_dir, mask_file)
            self.sample_paths.append((stacked_path, mask_path)) # list of tuples: (stacked, mask) file paths

    
    def load_tif(self, filepath, single_band=False):
        # for reading/loading data from tifs/rasters
        with rasterio.open(filepath) as dataset:
            if single_band:
                data = dataset.read(1) # shape (H, W)
            else:
                data = dataset.read()   # shape (C, H, W)
            return data.astype(np.float32)
                

    def normalize_channels(self, volume):
        """Normalize each channel of stacked image -  for (H, W, C) format"""
        normalized = np.zeros_like(volume)
        num_channels = volume.shape[2]  # We know it's (H, W, C) at this point
        
        for c in range(num_channels):
            channel = volume[:, :, c]
            mean = channel.mean()
            std = channel.std()
            if std > 0:
                normalized[:, :, c] = (channel - mean) / std
            else:
                normalized[:, :, c] = channel - mean
        
        return normalized



   def __getitem__(self, idx):
        # Load data on-demand
        stacked_path, mask_path = self.sample_paths[idx]
        
        # Load the data
        image = self.load_tif(stacked_path)  # Shape: (C, H, W)
        mask = self.load_tif(mask_path, single_band=True)  # Shape: (H, W)
        
        # Transpose to (H, W, C) format to match original brainsegmentation dataset & transform funcs
        image = np.transpose(image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        mask = np.expand_dims(mask, axis=2)  # (H, W) -> (H, W, 1)
        
        # Normalize (per-channel normalization)
        image = self.normalize_channels(image)
        
        # Apply transforms if provided (transforms expect HWC format)
        if self.transform is not None:
            image, mask = self.transform((image, mask))
        
        # Apply final crop if specified (LAST step) 50.01km -> ~15 km
        if self.crop_size_km is not None:
            image, mask = center_crop_sample((image, mask))
        
        # Transpose to (C, H, W) format for PyTorch
        image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        mask = np.transpose(mask, (2, 0, 1))    # (H, W, 1) -> (1, H, W)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()  # Shape: (C, H, W)
        mask_tensor = torch.from_numpy(mask).float()    # Shape: (1, H, W)
        
        return image_tensor, mask_tensor







# class BrainSegmentationDataset(Dataset):
#     """Brain MRI dataset for FLAIR abnormality segmentation"""

#     in_channels = 3
#     out_channels = 1

#     def __init__(
#         self,
#         images_dir,
#         transform=None,
#         image_size=256,
#         subset="train",
#         random_sampling=True,
#         validation_cases=10,
#         seed=42,
#     ):
#         assert subset in ["all", "train", "validation"]

#         # read images
#         volumes = {}
#         masks = {}
#         print("reading {} images...".format(subset))
#         for (dirpath, dirnames, filenames) in os.walk(images_dir):
#             image_slices = []
#             mask_slices = []
#             for filename in sorted(
#                 filter(lambda f: ".tif" in f, filenames),
#                 key=lambda x: int(x.split(".")[-2].split("_")[4]),
#             ):
#                 filepath = os.path.join(dirpath, filename)
#                 if "mask" in filename:
#                     mask_slices.append(imread(filepath, as_gray=True))
#                 else:
#                     image_slices.append(imread(filepath))
#             if len(image_slices) > 0:
#                 patient_id = dirpath.split("/")[-1]
#                 volumes[patient_id] = np.array(image_slices[1:-1])
#                 masks[patient_id] = np.array(mask_slices[1:-1])

#         self.patients = sorted(volumes)

#         # select cases to subset
#         if not subset == "all":
#             random.seed(seed)
#             validation_patients = random.sample(self.patients, k=validation_cases)
#             if subset == "validation":
#                 self.patients = validation_patients
#             else:
#                 self.patients = sorted(
#                     list(set(self.patients).difference(validation_patients))
#                 )

#         print("preprocessing {} volumes...".format(subset))
#         # create list of tuples (volume, mask)
#         self.volumes = [(volumes[k], masks[k]) for k in self.patients]

#         print("cropping {} volumes...".format(subset))
#         # crop to smallest enclosing volume
#         self.volumes = [crop_sample(v) for v in self.volumes]

#         print("padding {} volumes...".format(subset))
#         # pad to square
#         self.volumes = [pad_sample(v) for v in self.volumes]

#         print("resizing {} volumes...".format(subset))
#         # resize
#         self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]

#         print("normalizing {} volumes...".format(subset))
#         # normalize channel-wise
#         self.volumes = [(normalize_volume(v), m) for v, m in self.volumes]

#         # probabilities for sampling slices based on masks
#         self.slice_weights = [m.sum(axis=-1).sum(axis=-1) for v, m in self.volumes]
#         self.slice_weights = [
#             (s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights
#         ]

#         # add channel dimension to masks
#         self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]

#         print("done creating {} dataset".format(subset))

#         # create global index for patient and slice (idx -> (p_idx, s_idx))
#         num_slices = [v.shape[0] for v, m in self.volumes]
#         self.patient_slice_index = list(
#             zip(
#                 sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
#                 sum([list(range(x)) for x in num_slices], []),
#             )
#         )

#         self.random_sampling = random_sampling

#         self.transform = transform

#     def __len__(self):
#         return len(self.patient_slice_index)

#     def __getitem__(self, idx):
#         patient = self.patient_slice_index[idx][0]
#         slice_n = self.patient_slice_index[idx][1]

#         if self.random_sampling:
#             patient = np.random.randint(len(self.volumes))
#             slice_n = np.random.choice(
#                 range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
#             )

#         v, m = self.volumes[patient]
#         image = v[slice_n]
#         mask = m[slice_n]

#         if self.transform is not None:
#             image, mask = self.transform((image, mask))

#         # fix dimensions (C, H, W)
#         image = image.transpose(2, 0, 1)
#         mask = mask.transpose(2, 0, 1)

#         image_tensor = torch.from_numpy(image.astype(np.float32))
#         mask_tensor = torch.from_numpy(mask.astype(np.float32))

#         # return tensors
#         return image_tensor, mask_tensor


