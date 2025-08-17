import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# from tqdm import tqdm

from dataset import WildfireDataset as Dataset # custom dataset

from logger import Logger
from loss import DiceLoss
from transform import transforms
from unet import UNet
from utils import log_images, dsc



def datasets(args):
    """ create train/validation datasets manually using filepaths from pre-generated samples_list
        
        we shuffle and split explicitly here to control exactly which samples go into training vs validation
        This avoids having the Dataset class handle splitting internally
    """
    # TODO: logic here to get all paths for stacked files and CORRESPONDING firemask files
    samples_list = []
    # structure should be:
    samples_list.append({
        "stacked_path": ..., 
        "fire_mask_path": ...
    })

    # shuffle and split into train & validation (80/20)
    np.random.seed(42)
    np.random.shuffle(samples_list)
    split_idx = int(len(samples_list) * 0.8)
    train_list = samples_list[:split_idx]
    valid_list = samples_list[:split_idx]

    # create datasets
    train_dataset = Dataset(
        train_list, 
        image_size=args.image_size, # to resize
        transform=transforms(scale=arge.aug_scale, angle=arge.aug_angle, flilp_prob=0.5) # to transform
    )

    valid_dataset = Dataset(
        valid_list,
        image_size=args.image_size,
        transform=None,
    )

    return train_dataset, valid_dataset




def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last = True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last = False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid



def dsc_per_sample():
    """ computed per wildfire SAMPLE"""
    return [dsc(y_pred, y_true) for y_pred, y_true in zip(pred_list, true_list)]




def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def snapshotargs(args):
    args_file = os.path.join(args.logs, "args.json")
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp)





def main(args):
    makedirs(args)
    snapshotargs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels) # 17 in, 1 out
    unet.to(device)

    dsc_loss = DiceLoss()

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)

    logger = Logger(args.logs)

    best_validation_dsc = 0.0
    step = 0
    loss_train = []
    loss_valid = []

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    loss = dsc_loss(y_pred, y_true)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )
                        if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
                            if i * args.batch_size < args.vis_images:
                                tag = "image/{}".format(i)
                                num_images = args.vis_images - i * args.batch_size
                                logger.image_list_summary(
                                    tag,
                                    log_images(x, y_true, y_pred)[:num_images],
                                    step,
                                )

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    log_loss_summary(logger, loss_train, step)
                    loss_train = []

            if phase == "valid":
                log_loss_summary(logger, loss_valid, step, prefix="val_")
                # simplified diceloss for wildfire prediction
                mean_dsc = np.mean([dsc(y_pred, y_true) for y_pred, y_true in zip(validation_pred, validation_true)])
                logger.scalar_summary("val_dsc", mean_dsc, step)
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(unet.state_dict(), os.path.join(args.weights, "unet.pt"))
                loss_valid = []

    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))








if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for wildfire prediction segmentation"
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
        help="initial learning rate (default: 0.001)",
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
    parser.add_argument(
        "--images", type=str, default="./kaggle_3m", help="root folder with images"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--aug-scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )
    args = parser.parse_args()
    main(args)
