# U-Net for Wildfire Segmentation

This repository contains an implementation of the U-Net model in PyTorch, adapted for the segmentation of wildfire extents using multiband raster images. The original implementation was used for brain MRI segmentation and has been modified to handle wildfire prediction data.

## Overview

The goal of this project is to predict wildfire extent spread using U-Net segmentation models on remotely-sensed multiband raster images. The input to the model is a stacked image at a given timestep, and the output is a predicted wildfire segmentation mask at the next timestep.

## Dataset Preparation

The dataset should be organized into directories containing stacked multiband image samples and corresponding fire masks. These directories are accessed by the dataset.py script to load data for training and validation.

## Directory Structure
- data_dir/
    - fire_final/
    - stacked/
    - train_records.pkl
    - Test_records.pkl

fire_final/: Contains the wildfire segmentation .tif masks.
stacked/: Contains the stacked multiband .tif images.
train_records.pkl and test_records.pkl: Pickle files (lists of dictionaries) mapping the image and mask pairs for training and testing.

## DataLoader Initialization
The dataset.py script has been adapted to include dataloaders for wildfire data:

def __init__(
        self, 
        data_dir,       # Directory for all data- stacked files and masks
        records_pkl,    # Pickle file for organizing the pairs of files for samples
        transform=None, 
        crop_size_km=None, 
        subset="train", # 'train' or 'validation'
        seed=42):

## Model

A segmentation model implemented in this repository is U-Net as described in [Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm](https://doi.org/10.1016/j.compbiomed.2019.05.002) with added batch normalization.

![unet](./assets/unet.png)


## Training
To train the model:

Ensure your dataset is prepared and organized as described above.
Run the train.py script. Default paths and parameters can be adjusted as needed.
python train.py --data_dir path/to/data_dir --records_pkl path/to/train_records.pkl
For more options and help, run:

python train.py --help

## Inference
To perform inference using the trained model:

Ensure the dataset is prepared and the model weights are available (in ./weights).
Run the inference.py script with specified paths to weights and images.
python inference.py --weights path/to/weights.pt --data_dir path/to/data_dir --records_pkl path/to/test_records.pkl
For more options and help, run:

python inference.py --help


## Results
The primary metric currently returned is the Dice Similarity Coefficient (DSC), which is calculated as 1 − dice loss. This metric provides an indication of the overlap between the predicted wildfire segmentation mask and the ground truth mask. Higher DSC values indicate better model performance.



## Docker
To build and run the Docker container:

docker build -t wildfire-segmentation .

nvidia-docker run --rm --shm-size 8G -it -v `pwd`:/workspace wildfire-segmentation



This repository was adapted from an existing implementation: https://github.com/mateuszbuda/brain-segmentation-pytorch/
```
@article{buda2019association,
  title={Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm},
  author={Buda, Mateusz and Saha, Ashirbani and Mazurowski, Maciej A},
  journal={Computers in Biology and Medicine},
  volume={109},
  year={2019},
  publisher={Elsevier},
  doi={10.1016/j.compbiomed.2019.05.002}
}
```



