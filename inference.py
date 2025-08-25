import argparse
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
# from medpy.filter.binary import largest_connected_component
from skimage.io import imsave
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import WildfireDataset as Dataset
from unet import UNet
from utils import gray2rgb, outline#, dsc
from loss import DiceLoss



def makedirs(args):
    os.makedirs(args.predictions, exist_ok=True)


def data_loader(args):
    dataset = Dataset(
        data_dir=args.images,
        records_pkl=args.validation_mapper,
        crop_size_km=args.crop_size_km, 
        subset="validation",
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, drop_last=False, num_workers=1
    )
    return loader


# def dsc_distribution(volumes):
#     dsc_dict = {}
#     for p in volumes:
#         y_pred = volumes[p][1]
#         y_true = volumes[p][2]
#         dsc_dict[p] = dsc(y_pred, y_true, lcc=False)
#     return dsc_dict


# def plot_dsc(dsc_dist):
#     y_positions = np.arange(len(dsc_dist))
#     dsc_dist = sorted(dsc_dist.items(), key=lambda x: x[1])
#     values = [x[1] for x in dsc_dist]
#     labels = [x[0] for x in dsc_dist]
#     labels = ["_".join(l.split("_")[1:-1]) for l in labels]
#     fig = plt.figure(figsize=(12, 8))
#     canvas = FigureCanvasAgg(fig)
#     plt.barh(y_positions, values, align="center", color="skyblue")
#     plt.yticks(y_positions, labels)
#     plt.xticks(np.arange(0.0, 1.0, 0.1))
#     plt.xlim([0.0, 1.0])
#     plt.gca().axvline(np.mean(values), color="tomato", linewidth=2)
#     plt.gca().axvline(np.median(values), color="forestgreen", linewidth=2)
#     plt.xlabel("Dice coefficient", fontsize="x-large")
#     plt.gca().xaxis.grid(color="silver", alpha=0.5, linestyle="--", linewidth=1)
#     plt.tight_layout()
#     canvas.draw()
#     plt.close()
#     s, (width, height) = canvas.print_to_buffer()
#     return np.fromstring(s, np.uint8).reshape((height, width, 4))



def plot_dice_loss_distribution(dice_losses):
    fig = plt.figure(figsize=(10, 6))
    canvas = FigureCanvasAgg(fig)

    plt.hist(dice_losses, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(dice_losses), color='tomato', linewidth=2, label=f'Mean: {np.mean(dice_losses):.3f}')
    plt.axvline(np.median(dice_losses), color='forestgreen', linewidth=2, label=f'Median: {np.median(dice_losses):.3f}')

    plt.xlabel('Dice Loss', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Dice Losses', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()
    return np.frombuffer(s, np.uint8).reshape((height, width, 4))




def save_prediction_visualization(input_img, pred_mask, true_mask, idx, save_dir, dsc_val):
    """
    Save a visualization showing input features, prediction, and ground truth
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Show some input channels (select a few representative ones)
    channel_names = ['Channel 0', 'Channel 1', 'Channel 2']  # Update with actual feature names
    for i, ax in enumerate(axes[0]):
        if i < min(3, input_img.shape[0]):
            im = ax.imshow(input_img[i], cmap='viridis')
            ax.set_title(f'{channel_names[i]}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')

    # Ground truth fire mask
    axes[1, 0].imshow(true_mask[0], cmap='Reds', alpha=0.8)
    axes[1, 0].set_title('Ground Truth Fire')
    axes[1, 0].axis('off')

    # Predicted fire mask
    pred_binary = (pred_mask[0] > 0.5).astype(float)
    axes[1, 1].imshow(pred_binary, cmap='Reds', alpha=0.8)
    axes[1, 1].set_title(f'Prediction (DSC: {dsc_val:.3f})')
    axes[1, 1].axis('off')

    # Overlay comparison
    overlay = np.zeros((true_mask.shape[1], true_mask.shape[2], 3))
    true_binary = (true_mask[0] > 0.5).astype(bool)
    pred_binary_bool = (pred_mask[0] > 0.5).astype(bool)

    overlay[true_binary & pred_binary_bool, 0] = 1.0  # True Positive (Red)
    overlay[~true_binary & pred_binary_bool, 2] = 1.0  # False Positive (Blue)
    overlay[true_binary & ~pred_binary_bool, 1] = 1.0  # False Negative (Green)

    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Overlay (R=TP, B=FP, G=FN)')
    axes[1, 2].axis('off')

    plt.tight_layout()
    filename = f"prediction_{str(idx).zfill(4)}_dsc_{dsc_val:.3f}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()





def main(args):
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader = data_loader(args)

    with torch.set_grad_enabled(False):
        unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
        state_dict = torch.load(args.weights, map_location=device)
        unet.load_state_dict(state_dict)
        unet.eval()
        unet.to(device)

        dice_loss = DiceLoss() # same loss metric as training

        input_list = []
        pred_list = []
        true_list = []

        dice_losses = []

        for i, data in tqdm(enumerate(loader)):
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)

            y_pred = unet(x)
            y_pred_np = y_pred.detach().cpu().numpy()
            pred_list.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])

            y_true_np = y_true.detach().cpu().numpy()
            true_list.extend([y_true_np[s] for s in range(y_true_np.shape[0])])

            x_np = x.detach().cpu().numpy()
            input_list.extend([x_np[s] for s in range(x_np.shape[0])])

            for j in range(y_pred_np.shape[0]):
                pred_tensor = torch.from_numpy(y_pred_np[j]).unsqueeze(0).to(device)
                true_tensor = torch.from_numpy(y_true_np[j]).unsqueeze(0).to(device)
                loss_val = dice_loss(pred_tensor, true_tensor)
                dice_losses.append(loss_val.item())

    # plot Dice Loss distribution
    dice_loss_dist_plot = plot_dice_loss_distribution(dice_losses)
    imsave(args.figure, dice_loss_dist_plot)

    # Save prediction visualizations
    print("Saving prediction visualizations...")
    for i in tqdm(range(min(len(input_list), args.max_vis))):
        save_prediction_visualization(
            input_list[i],
            pred_list[i],
            true_list[i],
            i,
            args.predictions,
            1.0 - dice_losses[i]  # Convert Dice Loss to Dice Coefficient
        )


    # Print summary statistics
    print(f"\nInference Results:")
    print(f"Total samples: {len(dice_losses)}")
    print(f"\nUsing DiceLoss:")
    mean_dsc = np.mean([1.0 - dl for dl in dice_losses])
    median_dsc = np.median([1.0 - dl for dl in dice_losses])
    std_dsc = np.std([1.0 - dl for dl in dice_losses])
    print(f"  Mean DSC: {mean_dsc:.4f}")
    print(f"  Median DSC: {median_dsc:.4f}")
    print(f"  Std DSC: {std_dsc:.4f}")





# def plot_dsc_distribution(dsc_list):
#     """Plot histogram of DSC values"""
#     fig = plt.figure(figsize=(10, 6))
#     canvas = FigureCanvasAgg(fig)
    
#     plt.hist(dsc_list, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
#     plt.axvline(np.mean(dsc_list), color='tomato', linewidth=2, label=f'Mean: {np.mean(dsc_list):.3f}')
#     plt.axvline(np.median(dsc_list), color='forestgreen', linewidth=2, label=f'Median: {np.median(dsc_list):.3f}')
    
#     plt.xlabel('Dice Coefficient', fontsize=12)
#     plt.ylabel('Frequency', fontsize=12)
#     plt.title('Distribution of Dice Coefficients', fontsize=14)
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
    
#     canvas.draw()
#     plt.close()
#     s, (width, height) = canvas.print_to_buffer()
#     return np.fromstring(s, np.uint8).reshape((height, width, 4))

  



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference for segmentation of brain MRI"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--weights", type=str, required=True, help="path to weights file"
    )
    parser.add_argument(
        "--images", type=str, default="./kaggle_3m", help="root folder with images"
    )
    parser.add_argument(
        "--validation-mapper", type=str, required=True, help="path to validation mapper pkl file"
    )
    parser.add_argument(
        "--crop-size-km",
        type=int,
        default=15,
        help="crop size in km (should match training)",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="./predictions",
        help="folder for saving prediction visualizations",
    )
    parser.add_argument(
        "--figure",
        type=str,
        default="./dice_loss_distribution.png",
        help="filename for Dice Loss distribution figure",
    )
    parser.add_argument(
        "--max-vis",
        type=int,
        default=10,
        help="maximum number of visualizations to save (default: 10)"
    )

    args = parser.parse_args()
    main(args)
