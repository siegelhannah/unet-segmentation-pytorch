import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# from dataset import BrainSegmentationDataset as Dataset
from dataset import WildfireDataset as Dataset
#from logger import Logger
from loss import DiceLoss
from transform import transforms
from unet import UNet
from utils import log_images#, dsc




def datasets(args):
    train = Dataset(
        data_dir=args.images,
        records_pkl=args.train_mapper,
        transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5, max_shift_fraction=0.2),
        crop_size_km=args.crop_size_km,
        subset="train"
    )
    valid = Dataset(
        data_dir=args.images,
        records_pkl=args.validation_mapper,
        transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5, max_shift_fraction=0.2),
        crop_size_km=args.crop_size_km,
        subset="validation"
    )
    return train, valid




def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid




# def log_loss_summary(logger, loss, step, prefix=""):
#     logger.scalar_summary(prefix + "loss", np.mean(loss), step)


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def snapshotargs(args):
    args_file = os.path.join(args.logs, "args.json")
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp)





def main(args):
    print("[DEBUG] Starting training, creating directories...")
    makedirs(args)
    snapshotargs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    # create dataloaders
    print("[DEBUG] Creating datasets and data loaders...")
    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}
    print("[DEBUG] Data loaders created")

    unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
    unet.to(device)

    dice_loss = DiceLoss() # define loss metric
    best_validation_loss = float('inf') # we want lower loss, so try to do better than this

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)

    # logger = Logger(args.logs)
    loss_train = []
    loss_valid = []

    step = 0

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            for i, data in enumerate(loaders[phase]): # iterate thru batches coming from DataLoader (either train batches or valid)
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad() # zero the gradient before computing the next one

                # Forward Pass and Loss Calculation
                with torch.set_grad_enabled(phase == "train"): # only compute gradients when in training phase
                    y_pred = unet(x)
                    loss = dice_loss(y_pred, y_true)

                    # Validation step
                    if phase == "valid":
                        loss_valid.append(loss.item()) # accumulate validation loss

                        # data viz stuff
                        if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
                            if i * args.batch_size < args.vis_images:
                                tag = "image/{}".format(i)
                                num_images = args.vis_images - i * args.batch_size
                                # logger.image_list_summary(
                                #     tag,
                                #     log_images(x, y_true, y_pred)[:num_images],
                                #     step,
                                # )

                    # training-specific operations: accumulate loss, backward pass, step
                    if phase == "train":
                        loss_train.append(loss.item()) # loss for current batch
                        loss.backward()
                        optimizer.step()

                # logging stuff
                if phase == "train" and (step + 1) % 10 == 0:
                    # log_loss_summary(logger, loss_train, step)
                    loss_train = []

            if phase == "valid":
                # log_loss_summary(logger, loss_valid, step, prefix="val_")
                mean_val_loss = np.mean(loss_valid) # mean of all validation losses
               
                # print dsc (1 - diceloss: higher is better)
                mean_val_dsc = 1.0 - mean_val_loss
                # logger.scalar_summary("val_dsc", mean_val_dsc, step)
                
                if mean_val_loss < best_validation_loss:
                    best_validation_loss = mean_val_loss
                    torch.save(unet.state_dict(), os.path.join(args.weights, "unet.pt"))
                loss_valid = []

    print(f"Best validation mean DSC: {1.0 - best_validation_loss}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of wildfire masks at t+1"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="initial learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--vis-images",
        type=int,
        default=200,
        help="number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=10,
        help="frequency of saving images to log file (default: 10)",
    )
    parser.add_argument(
        "--weights", type=str, default="./weights", help="folder to save weights"
    )
    parser.add_argument(
        "--logs", type=str, default="./logs", help="folder to save logs"
    )

    # DATA
    parser.add_argument(
        "--images", type=str, default="./kaggle_3m", help="root folder with images"
    )
    parser.add_argument(
        "--train_mapper", type=str, default="./train_records.pkl", help="path to mapper pkl file for training data subset"
    )
    parser.add_argument(
        "--validation_mapper", type=str, default="./test_records.pkl", help="path to mapper pkl file for validation data subset"
    )
  
    parser.add_argument(
        "--aug-scale",
        type=float,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )
    parser.add_argument(
        "--crop_size_km",
        type=int,
        default=15,
        help="final cropped size in kilometers sidelength (defualt: 15)",
    )
    args = parser.parse_args()
    main(args)