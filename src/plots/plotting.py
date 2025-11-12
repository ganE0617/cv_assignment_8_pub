"""Plotting utilities for visualizations."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from pathlib import Path


def plot_training_curves(
    train_losses: List[float],
    train_accs: List[float],
    test_losses: List[float],
    test_accs: List[float],
    save_path: str
) -> None:
    """
    Plot training curves (loss and accuracy).
    
    Args:
        train_losses: Training losses per epoch
        train_accs: Training accuracies per epoch
        test_losses: Test losses per epoch
        test_accs: Test accuracies per epoch
        save_path: Path to save plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Test Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Test Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str,
    class_names: Optional[List[str]] = None
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix (num_classes, num_classes)
        save_path: Path to save plot
        class_names: Optional list of class names
    """
    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_title('Confusion Matrix', fontsize=14, pad=20)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_misclassified_grid(
    X_mis: np.ndarray,
    y_pred_mis: np.ndarray,
    y_true_mis: np.ndarray,
    save_path: str,
    dataset: str = "mnist"
) -> None:
    """
    Plot grid of misclassified samples.
    
    Args:
        X_mis: Misclassified images (N, C, H, W)
        y_pred_mis: Predicted labels (N,)
        y_true_mis: True labels (N,)
        save_path: Path to save plot
        dataset: Dataset name ('mnist' or 'cifar10')
    """
    n_samples = len(X_mis)
    if n_samples == 0:
        print("No misclassified samples to plot.")
        return
    
    # Ensure all arrays have the same length
    min_len = min(len(X_mis), len(y_pred_mis), len(y_true_mis))
    if min_len < n_samples:
        print(f"Warning: Array length mismatch. Using {min_len} samples instead of {n_samples}.")
        n_samples = min_len
        X_mis = X_mis[:n_samples]
        y_pred_mis = y_pred_mis[:n_samples]
        y_true_mis = y_true_mis[:n_samples]
    
    # Determine grid size
    cols = 6
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif not isinstance(axes, np.ndarray):
        axes = np.array([[axes]])
    
    # Plot only the actual samples we have
    for idx in range(n_samples):
        row = idx // cols
        col = idx % cols
        img = X_mis[idx]
        if dataset == 'mnist':
            # (1, 28, 28) -> (28, 28)
            img = img[0] if img.shape[0] == 1 else img
            axes[row, col].imshow(img, cmap='gray')
        else:  # CIFAR-10
            # (3, 32, 32) -> (32, 32, 3)
            img = np.transpose(img, (1, 2, 0))
            axes[row, col].imshow(img)
        
        axes[row, col].set_title(
            f'True: {y_true_mis[idx]}\nPred: {y_pred_mis[idx]}',
            fontsize=8
        )
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for idx in range(n_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle('Misclassified Samples', fontsize=14)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_experiments_comparison(
    results: Dict[str, Dict[str, List[float]]],
    save_path: str
) -> None:
    """
    Plot side-by-side comparison of experiments.
    
    Args:
        results: Dict mapping experiment names to {'test_accs': [...], 'epochs': [...]}
        save_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for exp_name, data in results.items():
        epochs = data.get('epochs', range(1, len(data['test_accs']) + 1))
        ax.plot(epochs, data['test_accs'], label=exp_name, linewidth=2, marker='o', markersize=4)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Experiment Comparison: Test Accuracy', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

