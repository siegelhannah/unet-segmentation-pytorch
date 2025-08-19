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
from utils import dsc, gray2rgb, outline
from loss import DiceLoss



def makedirs(args):
    os.makedirs(args.predictions, exist_ok=True)


def data_loader(args):
    dataset = Dataset(
        data_dir=args.images,
        crop_size_km=args.crop_size_km, 
        subset="validation",
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, drop_last=False, num_workers=1
    )
    return loader


def dsc_distribution(volumes):
    dsc_dict = {}
    for p in volumes:
        y_pred = volumes[p][1]
        y_true = volumes[p][2]
        dsc_dict[p] = dsc(y_pred, y_true, lcc=False)
    return dsc_dict


def plot_dsc(dsc_dist):
    y_positions = np.arange(len(dsc_dist))
    dsc_dist = sorted(dsc_dist.items(), key=lambda x: x[1])
    values = [x[1] for x in dsc_dist]
    labels = [x[0] for x in dsc_dist]
    labels = ["_".join(l.split("_")[1:-1]) for l in labels]
    fig = plt.figure(figsize=(12, 8))
    canvas = FigureCanvasAgg(fig)
    plt.barh(y_positions, values, align="center", color="skyblue")
    plt.yticks(y_positions, labels)
    plt.xticks(np.arange(0.0, 1.0, 0.1))
    plt.xlim([0.0, 1.0])
    plt.gca().axvline(np.mean(values), color="tomato", linewidth=2)
    plt.gca().axvline(np.median(values), color="forestgreen", linewidth=2)
    plt.xlabel("Dice coefficient", fontsize="x-large")
    plt.gca().xaxis.grid(color="silver", alpha=0.5, linestyle="--", linewidth=1)
    plt.tight_layout()
    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()
    return np.fromstring(s, np.uint8).reshape((height, width, 4))





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

        dsc_list = []

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

    # calculate Dice Loss for each sample
    dsc_list = []
    for i in range(len(pred_list)):
        pred_tensor = torch.from_numpy(pred_list[i]).unsqueeze(0)
        true_tensor = torch.from_numpy(true_list[i]).unsqueeze(0)
        loss_val = dice_loss(pred_tensor, true_tensor)
        dsc = 1.0 - loss_val.item() # dsc coeff/value
        dsc_list.append(dsc)

    # plot DSC distribution
    dsc_dist_plot = plot_dsc_distribution(dsc_list)
    imsave(args.figure, dsc_dist_plot)

    # Save prediction visualizations
    print("Saving prediction visualizations...")
    for i in tqdm(range(min(len(input_list), args.max_vis))):
        save_prediction_visualization(
            input_list[i], 
            pred_list[i], 
            true_list[i], 
            i, 
            args.predictions,
            dsc_list[i]
        )

    # Print summary statistics
    print(f"\nInference Results:")
    print(f"Total samples: {len(dsc_list)}")
    print(f"\nUsing DiceLoss (training-consistent, with smoothing):")
    print(f"  Mean DSC: {np.mean(dsc_list_smooth):.4f}")
    print(f"  Median DSC: {np.median(dsc_list_smooth):.4f}")
    print(f"  Std DSC: {np.std(dsc_list_smooth):.4f}")
    print(f"\nUsing utils.dsc (binary, no smoothing):")
    print(f"  Mean DSC: {np.mean(dsc_list_raw):.4f}")
    print(f"  Median DSC: {np.median(dsc_list_raw):.4f}")
    print(f"  Std DSC: {np.std(dsc_list_raw):.4f}")
    print(f"\nDifference (smooth - raw): {np.mean(dsc_list_smooth) - np.mean(dsc_list_raw):.4f}")




def save_prediction_visualization(input_img, pred_mask, true_mask, idx, save_dir, dsc_val):
    """
    Save a visualization showing input features, prediction, and ground truth
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show some input channels (select a few representative ones)
    # Assuming channels might include temperature, humidity, wind, etc.
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
    # Create RGB image: Red = True Positive, Blue = False Positive, Green = False Negative
    overlay = np.zeros((true_mask.shape[1], true_mask.shape[2], 3))
    true_binary = (true_mask[0] > 0.5).astype(bool)
    pred_binary_bool = (pred_mask[0] > 0.5).astype(bool)
    
    # True Positive (both predict and actual fire) - Red
    overlay[true_binary & pred_binary_bool, 0] = 1.0
    # False Positive (predict fire, no actual fire) - Blue  
    overlay[~true_binary & pred_binary_bool, 2] = 1.0
    # False Negative (actual fire, no prediction) - Green
    overlay[true_binary & ~pred_binary_bool, 1] = 1.0
    
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Overlay (R=TP, B=FP, G=FN)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    filename = f"prediction_{str(idx).zfill(4)}_dsc_{dsc_val:.3f}.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()


def plot_dsc_distribution(dsc_list):
    """Plot histogram of DSC values"""
    fig = plt.figure(figsize=(10, 6))
    canvas = FigureCanvasAgg(fig)
    
    plt.hist(dsc_list, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(dsc_list), color='tomato', linewidth=2, label=f'Mean: {np.mean(dsc_list):.3f}')
    plt.axvline(np.median(dsc_list), color='forestgreen', linewidth=2, label=f'Median: {np.median(dsc_list):.3f}')
    
    plt.xlabel('Dice Coefficient', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Dice Coefficients', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()
    return np.fromstring(s, np.uint8).reshape((height, width, 4))

  



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
        default="./dsc.png",
        help="filename for DSC distribution figure",
    )

    args = parser.parse_args()
    main(args)
