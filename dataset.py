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
import pickle

# from utils import crop_sample, pad_sample, resize_sample, normalize_volume
from utils import center_crop_sample # for making images smaller by cropping to a smaller box



class WildfireDataset(Dataset):
    in_channels=18 # 18 features
    out_channels=1 # singleband fire mask

    def __init__(
        self, 
        data_dir,       # dir for all data- stacked files and masks
        records_pkl,     # pkl file for organizing the pairs of files for samples
        transform=None, 
        crop_size_km=None, 
        subset="train",     # 'train' or 'validation'
        seed=42):

        self.transform = transform
        self.crop_size_km = crop_size_km
        self.subset = subset


        # Load the records from the pickle file. Each record is a dict:
        # { 'date_t': ..., 'date_t_plus_one': ..., 'stacked_path': ..., 'fire_mask_path': ...,
        #   'fire_predict_path': ... ... }
        with open(records_pkl, 'rb') as f:
            self.records = pickle.load(f)

        # Build the list of file path PAIRS. (samples)
        # assuming 'stacked_path' and 'fire_predict_path' in the pickle are relative

        self.sample_paths = [] # list of tuple samples (stacked image, fire mask)
        for record in self.records: # go thru pkl file dicts

            # Adjust the paths so they join (depends on where the data is going to be stored)
            stacked_rel = record['stacked_path'].lstrip("./")
            mask_rel = record['fire_predict_path'].lstrip("./")
            stacked_path = os.path.join(data_dir, stacked_rel)
            mask_path = os.path.join(data_dir, mask_rel)
            self.sample_paths.append((stacked_path, mask_path))

        print("[DEBUG] Got sample paths, total:", len(self.sample_paths))

    def __len__(self):
        return len(self.sample_paths)

    
    def load_tif(self, filepath, single_band=False):
        # for reading/loading data from tifs/rasters
        # add nodata handling
        with rasterio.open(filepath) as dataset:
            if single_band:
                data = dataset.read(1) # shape (H, W)
            else:
                data = dataset.read()   # shape (C, H, W)
                if not single_band and data.shape[0] >= 8:
                    # Replace NaNs in band 8 (NDVI) with 0.0
                    data[7] = np.nan_to_num(data[7], nan=0.0)

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
        print(f"[DEBUG] Loading sample {idx} - {stacked_path}, {mask_path}")
        
        # Load the data
        image = self.load_tif(stacked_path)  # Shape: (C, H, W)
        mask = self.load_tif(mask_path, single_band=True)  # Shape: (H, W)

        # TODO: if not correct number of channels, skip
        # Check and adjust number of channels.
        if image.shape[0] != self.in_channels:
            print(f"[WARN] Sample {idx} has {image.shape[0]} channels; expected {self.in_channels}.")
            if image.shape[0] < self.in_channels:
                missing = self.in_channels - image.shape[0]
                pad = np.zeros((missing, image.shape[1], image.shape[2]), dtype=image.dtype)
                image = np.concatenate((image, pad), axis=0)
                print(f"[DEBUG] Sample {idx} padded to {image.shape[0]} channels.")
            
        
        print(f"[DEBUG] BEFORE transpose: image {image.shape}, mask {mask.shape}")
        # Transpose from (C,H,W) to (H, W, C) for easy use of transform functions
        image = np.transpose(image, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        mask = np.expand_dims(mask, axis=2)  # (H, W) -> (H, W, 1)
        print(f"[DEBUG] After transpose: image {image.shape}, mask {mask.shape}")
        
        # Normalize (per-channel normalization)
        image = self.normalize_channels(image)
        print("[DEBUG] After normalization, image mean", image.mean())
        
        # Apply transforms if provided (transforms expect HWC format)
        if self.transform is not None:
            image, mask = self.transform((image, mask))
            print(f"[DEBUG] After transforms: image {image.shape}, mask {mask.shape}")
        
        # Apply final crop if specified (LAST step) 50.01km -> ~15 km
        if self.crop_size_km is not None:
            image, mask = center_crop_sample((image, mask), crop_size_km=self.crop_size_km)
            print(f"[DEBUG] After center crop: image {image.shape}, mask {mask.shape}")

        # Ensure dimensions are multiples of 16
        if image.shape[0] % 16 != 0 or image.shape[1] % 16 != 0:
            raise ValueError(f"Image dimensions are not multiples of 16: {image.shape}")

        if mask.shape[0] % 16 != 0 or mask.shape[1] % 16 != 0:
            raise ValueError(f"Mask dimensions are not multiples of 16: {mask.shape}")

        
        # Transpose back to (C, H, W) format for PyTorch
        image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        mask = np.transpose(mask, (2, 0, 1))    # (H, W, 1) -> (1, H, W)
        print(f"[DEBUG] Before tensor conversion: image {image.shape}, mask {mask.shape}")
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()  # Shape: (C, H, W)
        mask_tensor = torch.from_numpy(mask).float()    # Shape: (1, H, W)
        print(f"[DEBUG] Sample {idx} processed, tensor shapes: image {image_tensor.shape}, mask {mask_tensor.shape}")
        
        return image_tensor, mask_tensor

