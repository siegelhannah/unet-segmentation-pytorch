"""
Simple script to split wildfire dataset into train/validation splits using the pickle file metadata
for when we only have a certain amount generated yet
"""

import os
import pickle
import shutil
from sklearn.model_selection import train_test_split
import argparse
import tarfile


def split_wildfire_data(pickle_path, tar_path, output_base_dir, validation_split=0.2, seed=42):
    """
    Split wildfire data into train/validation directories based on pickle metadata

    extract a downloaded .tar file of tifs and split those accordingly ^
    """
    extract_dir = os.path.join(output_base_dir, "extracted-data") # where to extract the tar file
    os.makedirs(extract_dir, exist_ok=True)
    
    with tarfile.open(tar_file_path, 'r') as tar: # extract
        tar.extractall(extract_dir)

    with open(pickle_path, 'rb') as f: # open pickle file
        data_info = pickle.load(f)
        
    # Create output directories
    train_dir = os.path.join(output_base_dir, 'train')
    val_dir = os.path.join(output_base_dir, 'validation')
    
    for dir_path in [train_dir, val_dir]:
        os.makedirs(os.path.join(dir_path, 'stacked'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'fire_masks'), exist_ok=True)
    
    # Split
    indices = list(range(len(data_info)))
    # train/validation split
    train_indices, val_indices = train_test_split(
        indices, 
        test_size=validation_split, 
        random_state=seed
    )
        
    # function to copy files for a given split
    # adjust the soure filepaths to point to the extracted directory
    def copy_files_for_split(indices, target_dir, split_name):

        for i, idx in enumerate(indices):
            sample = data_info[idx]

            # Get paths - adjusted to be relative to extracted dir
            stacked_rel_path = sample['stacked_path'].lstrip('./')
            fire_mask_rel_path = sample['fire_predict_path'].lstrip('./')

            stacked_src = os.path.join(extract_dir, stacked_rel_path)
            fire_mask_src - os.path.join(extract_dir, fire_mask_rel_path)
            
            # destination filenames (use in-order INDEX to avoid conflicts)
            # so that when we sort the files each list stays in the same order (pairs stay together)
            stacked_dst = os.path.join(target_dir, 'stacked', f'stacked_{idx:06d}.tif')
            fire_mask_dst = os.path.join(target_dir, 'fire_masks', f'fire_mask_{idx:06d}.tif')
            
            # Copy files if they exist
            if os.path.exists(stacked_src):
                shutil.copy2(stacked_src, stacked_dst)
            else:
                print(f"source file not found{stacked_src}")
            
            if os.path.exists(fire_mask_src):
                shutil.copy2(fire_mask_src, fire_mask_dst)
            else:
                print(f"source file not found {fire_mask_src}")
    
    # Copy files for each split
    copy_files_for_split(train_indices, train_dir, "train")
    copy_files_for_split(val_indices, val_dir, "validation") 
    
    # Save metadata for each split
    def save_split_metadata(indices, split_name, target_dir):
        split_metadata = [data_info[i] for i in indices]
        metadata_path = os.path.join(target_dir, f'{split_name}_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(split_metadata, f)
    
    save_split_metadata(train_indices, "train", train_dir)
    save_split_metadata(val_indices, "validation", val_dir)


def main():
    parser = argparse.ArgumentParser(description='Split wildfire dataset')
    parser.add_argument('pickle_path')
    parser.add_argument('tar_path')
    parser.add_argument('output_dir')
    
    args = parser.parse_args()
    
    split_wildfire_data(
        args.pickle_path,
        args.tar_path, 
        args.output_dir, 
        args.validation_split,
        args.seed
    )

if __name__ == "__main__":
    main()


