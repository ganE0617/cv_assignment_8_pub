"""Visualize feature maps from PyTorch CNN."""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from train_pytorch_cnn import CNN


def visualize_feature_maps(model, test_loader, dataset: str, device, class_idx: int = None):
    """
    Visualize feature maps from first convolutional layer.
    
    Args:
        model: Trained CNN model
        test_loader: Test data loader
        dataset: Dataset name ('mnist' or 'cifar10')
        device: Device to run on
        class_idx: Optional class index to find (e.g., 5 for digit 5, or None for any)
    """
    model.eval()
    
    # Find a sample
    sample_image = None
    sample_label = None
    
    for data, target in test_loader:
        if class_idx is not None:
            # Find sample with specific class
            mask = target == class_idx
            if mask.any():
                idx = torch.where(mask)[0][0]
                sample_image = data[idx:idx+1].to(device)
                sample_label = target[idx].item()
                break
        else:
            # Take first sample
            sample_image = data[0:1].to(device)
            sample_label = target[0].item()
            break
    
    if sample_image is None:
        print(f"Could not find sample with class {class_idx}")
        return
    
    print(f"Visualizing feature maps for class {sample_label}")
    
    # Extract feature maps from first conv layer
    with torch.no_grad():
        # Get first conv layer output
        if dataset == 'mnist':
            # First conv: Conv2d(1, 16, ...)
            conv1_output = model.features[0](sample_image)  # (1, 16, H, W)
            act1 = F.relu(conv1_output)  # (1, 16, H, W)
        else:  # CIFAR-10
            # First conv: Conv2d(3, 32, ...)
            conv1_output = model.features[0](sample_image)  # (1, 32, H, W)
            act1 = F.relu(conv1_output)  # (1, 32, H, W)
    
    # Convert to numpy
    act1_np = act1[0].cpu().numpy()  # (C, H, W)
    sample_np = sample_image[0].cpu().numpy()  # (C, H, W)
    
    # Visualize input image
    fig_input, ax_input = plt.subplots(1, 1, figsize=(4, 4))
    if dataset == 'mnist':
        ax_input.imshow(sample_np[0], cmap='gray')
    else:
        # CIFAR-10: (C, H, W) -> (H, W, C)
        img = np.transpose(sample_np, (1, 2, 0))
        ax_input.imshow(img)
    ax_input.set_title(f'Input Image (Class {sample_label})')
    ax_input.axis('off')
    
    input_path = Path(f"outputs/features/{dataset}_input.png")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(input_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Visualize feature maps (up to 8)
    num_maps = min(8, act1_np.shape[0])
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(num_maps):
        axes[i].imshow(act1_np[i], cmap='viridis')
        axes[i].set_title(f'Feature Map {i+1}')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_maps, 8):
        axes[i].axis('off')
    
    plt.suptitle(f'Feature Maps from First Conv Layer (Class {sample_label})', fontsize=14)
    plt.tight_layout()
    
    feat_path = Path(f"outputs/features/{dataset}_torch_conv1_featuremaps.png")
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(feat_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved input image to: {input_path}")
    print(f"Saved feature maps to: {feat_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize PyTorch CNN feature maps')
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10'],
                       help='Dataset to use')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to saved model checkpoint (optional)')
    parser.add_argument('--class_idx', type=int, default=None,
                       help='Class index to visualize (e.g., 5 for digit 5, or None for any)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = CNN(dataset=args.dataset).to(device)
    
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        # Try to load default checkpoint
        default_checkpoint = Path(f"outputs/logs/{args.dataset}_torch_model.pth")
        if default_checkpoint.exists():
            print(f"Loading default checkpoint from {default_checkpoint}...")
            checkpoint = torch.load(default_checkpoint, map_location=device)
            model.load_state_dict(checkpoint)
        else:
            print("Warning: No checkpoint provided and default checkpoint not found.")
            print(f"Train the model first with: python train_pytorch_cnn.py --dataset {args.dataset}")
            print("Using randomly initialized model for visualization.")
    
    # Load test data
    if args.dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Visualize
    visualize_feature_maps(model, test_loader, args.dataset, device, class_idx=args.class_idx)


if __name__ == '__main__':
    main()

