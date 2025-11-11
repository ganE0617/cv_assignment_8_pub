"""Train NumPy CNN from scratch."""

import argparse
import json
import csv
from pathlib import Path
from typing import List
import numpy as np

from src.np_cnn.utils import set_seed, minibatches
from src.np_cnn.datasets import load_mnist, load_cifar10
from src.np_cnn.model import build_cnn
from src.np_cnn.losses import softmax_cross_entropy
from src.np_cnn.optim import SGD
from src.np_cnn.metrics import accuracy, compute_confusion_matrix, get_misclassified_samples
from src.plots.plotting import plot_training_curves, plot_confusion_matrix, plot_misclassified_grid


def train_epoch(
    model,
    optimizer,
    X_train,
    y_train,
    batch_size,
    seed,
    epoch_num=None
):
    """Train for one epoch."""
    train_losses = []
    train_accs = []
    
    # Calculate number of batches for progress
    n_samples = X_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    batch_iter = minibatches(X_train, y_train, batch_size, shuffle=True, seed=seed)
    for batch_idx, (X_batch, y_batch) in enumerate(batch_iter):
        # Forward pass
        logits = model.forward(X_batch)
        loss, grad_loss = softmax_cross_entropy(logits, y_batch, return_grad=True)
        
        # Backward pass
        model.backward(grad_loss)
        
        # Update parameters
        optimizer.step(model.get_trainable_layers())
        
        # Metrics
        acc = accuracy(logits, y_batch)
        train_losses.append(loss)
        train_accs.append(acc)
        
        # Progress indicator every 10 batches
        if epoch_num is not None and (batch_idx + 1) % 10 == 0:
            print(f"  Epoch {epoch_num}: Batch {batch_idx + 1}/{n_batches} - Loss: {loss:.4f}, Acc: {acc:.4f}")
    
    return np.mean(train_losses), np.mean(train_accs)


def evaluate(model, X, y, batch_size):
    """Evaluate on dataset."""
    losses = []
    accs = []
    all_logits = []
    all_labels = []
    
    for X_batch, y_batch in minibatches(X, y, batch_size, shuffle=False):
        logits = model.forward(X_batch)
        loss, _ = softmax_cross_entropy(logits, y_batch, return_grad=False)
        acc = accuracy(logits, y_batch)
        
        losses.append(loss)
        accs.append(acc)
        all_logits.append(logits)
        all_labels.append(y_batch)
    
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return np.mean(losses), np.mean(accs), all_logits, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train NumPy CNN')
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10'],
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--filters', type=str, default=None,
                       help='Comma-separated filter counts (e.g., "8,16")')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_pool', action='store_true', help='Remove pooling layers')
    parser.add_argument('--save_prefix', type=str, default=None, help='Save prefix')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Default filters
    if args.filters is None:
        if args.dataset == 'mnist':
            args.filters = "8,16"
        else:
            args.filters = "32,64"
    
    filters = [int(f) for f in args.filters.split(',')]
    
    # Load dataset
    print(f"Loading {args.dataset}...")
    if args.dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist(data_dir="./data", normalize=True)
        input_channels = 1
        num_classes = 10
    else:
        X_train, y_train, X_test, y_test = load_cifar10(data_dir="./data", normalize=True)
        input_channels = 3
        num_classes = 10
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Build model
    model = build_cnn(
        input_channels=input_channels,
        filters=filters,
        kernel_size=args.kernel_size,
        use_pooling=not args.no_pool,
        num_classes=num_classes
    )
    
    # Optimizer
    optimizer = SGD(lr=args.lr, weight_decay=args.weight_decay)
    
    # Training
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    # Setup logging
    save_prefix = args.save_prefix or f"{args.dataset}_numpy"
    log_file = Path(f"outputs/logs/{save_prefix}_log.csv")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
    
    print("\nStarting training...")
    print("Note: NumPy CNN from scratch is slow due to naive loop implementation.")
    print("This is expected - each epoch may take 5-10 minutes on CPU.\n")
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs} - Training...")
        # Train
        train_loss, train_acc = train_epoch(
            model, optimizer, X_train, y_train, args.batch_size, args.seed + epoch, epoch_num=epoch
        )
        
        # Evaluate
        test_loss, test_acc, test_logits, test_labels = evaluate(
            model, X_test, y_test, args.batch_size
        )
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Log to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
    
    # Final evaluation for confusion matrix and misclassified
    print("\nGenerating final predictions...")
    final_test_loss, final_test_acc, final_logits, final_labels = evaluate(
        model, X_test, y_test, args.batch_size
    )
    
    # Save plots
    print("\nSaving plots...")
    plot_path = Path(f"outputs/plots/{save_prefix}_loss_acc.png")
    plot_training_curves(train_losses, train_accs, test_losses, test_accs, str(plot_path))
    
    # Confusion matrix
    cm = compute_confusion_matrix(final_logits, final_labels, num_classes)
    cm_path = Path(f"outputs/confusion/{save_prefix}_confusion.png")
    class_names = [str(i) for i in range(num_classes)]
    plot_confusion_matrix(cm, str(cm_path), class_names)
    
    # Misclassified samples
    X_mis, y_pred_mis, y_true_mis = get_misclassified_samples(
        X_test, final_logits, final_labels, max_samples=24
    )
    mis_path = Path(f"outputs/misclassified/{save_prefix}_misclf.png")
    plot_misclassified_grid(X_mis, y_pred_mis, y_true_mis, str(mis_path), dataset=args.dataset)
    
    # Save summary
    summary = {
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'filters': filters,
        'kernel_size': args.kernel_size,
        'use_pooling': not args.no_pool,
        'seed': args.seed,
        'final_train_loss': float(train_losses[-1]),
        'final_train_acc': float(train_accs[-1]),
        'final_test_loss': float(test_losses[-1]),
        'final_test_acc': float(test_accs[-1]),
        'best_test_acc': float(max(test_accs))
    }
    
    summary_path = Path(f"outputs/logs/{save_prefix}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best test accuracy: {max(test_accs):.4f}")
    print(f"Final test accuracy: {test_accs[-1]:.4f}")
    print(f"\nOutputs saved to:")
    print(f"  - Logs: {log_file}")
    print(f"  - Summary: {summary_path}")
    print(f"  - Plots: {plot_path}")
    print(f"  - Confusion: {cm_path}")
    print(f"  - Misclassified: {mis_path}")


if __name__ == '__main__':
    main()

