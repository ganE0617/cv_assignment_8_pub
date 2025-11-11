"""Run multiple NumPy CNN experiments and compare results."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List
import numpy as np

from src.np_cnn.utils import set_seed
from src.plots.plotting import plot_experiments_comparison


def run_experiment(
    dataset: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    filters: List[int],
    kernel_size: int,
    use_pooling: bool,
    seed: int,
    exp_name: str
) -> Dict:
    """Run a single experiment and return results."""
    from train_numpy_cnn import train_epoch, evaluate
    from src.np_cnn.datasets import load_mnist, load_cifar10
    from src.np_cnn.model import build_cnn
    from src.np_cnn.optim import SGD
        
    set_seed(seed)
    
    # Load dataset
    if dataset == 'mnist':
        X_train, y_train, X_test, y_test = load_mnist(data_dir="./data", normalize=True)
        input_channels = 1
    else:
        X_train, y_train, X_test, y_test = load_cifar10(data_dir="./data", normalize=True)
        input_channels = 3
    
    # Build model
    model = build_cnn(
        input_channels=input_channels,
        filters=filters,
        kernel_size=kernel_size,
        use_pooling=use_pooling,
        num_classes=10
    )
    
    # Optimizer
    optimizer = SGD(lr=lr, weight_decay=weight_decay)
    
    # Training
    test_accs = []
    
    for epoch in range(1, epochs + 1):
        # Train
        train_epoch(model, optimizer, X_train, y_train, batch_size, seed + epoch)
        
        # Evaluate
        test_loss, test_acc, _, _ = evaluate(model, X_test, y_test, batch_size)
        test_accs.append(test_acc)
    
    # Final evaluation
    final_test_loss, final_test_acc, final_logits, final_labels = evaluate(
        model, X_test, y_test, batch_size
    )
    
    return {
        'name': exp_name,
        'test_accs': test_accs,
        'final_test_acc': final_test_acc,
        'final_test_loss': final_test_loss
    }


def main():
    parser = argparse.ArgumentParser(description='Run NumPy CNN experiments')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                       help='Dataset to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Define experiments
    experiments = []
    
    if args.dataset == 'mnist':
        # Experiment 1: Baseline (8,16 filters, kernel=3, pooling)
        experiments.append({
            'name': 'Baseline (8,16, k=3, pool)',
            'filters': [8, 16],
            'kernel_size': 3,
            'use_pooling': True,
            'lr': 1e-2,
            'weight_decay': 1e-4
        })
        
        # Experiment 2: More filters (16,32)
        experiments.append({
            'name': 'More filters (16,32, k=3, pool)',
            'filters': [16, 32],
            'kernel_size': 3,
            'use_pooling': True,
            'lr': 1e-2,
            'weight_decay': 1e-4
        })
        
        # Experiment 3: Larger kernel (k=5)
        experiments.append({
            'name': 'Larger kernel (8,16, k=5, pool)',
            'filters': [8, 16],
            'kernel_size': 5,
            'use_pooling': True,
            'lr': 1e-2,
            'weight_decay': 1e-4
        })
        
        # Experiment 4: No pooling
        experiments.append({
            'name': 'No pooling (8,16, k=3)',
            'filters': [8, 16],
            'kernel_size': 3,
            'use_pooling': False,
            'lr': 1e-2,
            'weight_decay': 1e-4
        })
        
        # Experiment 5: Higher LR
        experiments.append({
            'name': 'Higher LR (8,16, k=3, lr=0.1)',
            'filters': [8, 16],
            'kernel_size': 3,
            'use_pooling': True,
            'lr': 1e-1,
            'weight_decay': 1e-4
        })
        
        # Experiment 6: Lower LR
        experiments.append({
            'name': 'Lower LR (8,16, k=3, lr=0.001)',
            'filters': [8, 16],
            'kernel_size': 3,
            'use_pooling': True,
            'lr': 1e-3,
            'weight_decay': 1e-4
        })
        
        epochs = 10
    else:  # CIFAR-10
        # Experiment 1: Baseline (32,64 filters, kernel=3, pooling)
        experiments.append({
            'name': 'Baseline (32,64, k=3, pool)',
            'filters': [32, 64],
            'kernel_size': 3,
            'use_pooling': True,
            'lr': 1e-2,
            'weight_decay': 1e-4
        })
        
        # Experiment 2: More filters (64,128)
        experiments.append({
            'name': 'More filters (64,128, k=3, pool)',
            'filters': [64, 128],
            'kernel_size': 3,
            'use_pooling': True,
            'lr': 1e-2,
            'weight_decay': 1e-4
        })
        
        # Experiment 3: Larger kernel (k=5)
        experiments.append({
            'name': 'Larger kernel (32,64, k=5, pool)',
            'filters': [32, 64],
            'kernel_size': 5,
            'use_pooling': True,
            'lr': 1e-2,
            'weight_decay': 1e-4
        })
        
        # Experiment 4: No pooling
        experiments.append({
            'name': 'No pooling (32,64, k=3)',
            'filters': [32, 64],
            'kernel_size': 3,
            'use_pooling': False,
            'lr': 1e-2,
            'weight_decay': 1e-4
        })
        
        epochs = 15
    
    batch_size = 128
    
    # Run experiments
    print(f"Running {len(experiments)} experiments on {args.dataset}...")
    results = {}
    
    for i, exp in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(experiments)}: {exp['name']}")
        print(f"{'='*60}")
        
        result = run_experiment(
            dataset=args.dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=exp['lr'],
            weight_decay=exp['weight_decay'],
            filters=exp['filters'],
            kernel_size=exp['kernel_size'],
            use_pooling=exp['use_pooling'],
            seed=args.seed + i,
            exp_name=exp['name']
        )
        
        results[exp['name']] = {
            'test_accs': result['test_accs'],
            'epochs': list(range(1, epochs + 1)),
            'final_test_acc': result['final_test_acc'],
            'final_test_loss': result['final_test_loss']
        }
        
        print(f"Final test accuracy: {result['final_test_acc']:.4f}")
    
    # Save results to CSV
    csv_path = Path("outputs/logs/experiments_numpy.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['experiment', 'epoch', 'test_acc'])
        for exp_name, data in results.items():
            for epoch, acc in zip(data['epochs'], data['test_accs']):
                writer.writerow([exp_name, epoch, acc])
    
    # Plot comparison
    plot_path = Path("outputs/plots/numpy_experiments_acc.png")
    plot_experiments_comparison(results, str(plot_path))
    
    print(f"\n{'='*60}")
    print("Experiments complete!")
    print(f"Results saved to: {csv_path}")
    print(f"Comparison plot saved to: {plot_path}")
    print(f"\nFinal accuracies:")
    for exp_name, data in results.items():
        print(f"  {exp_name}: {data['final_test_acc']:.4f}")

if __name__ == '__main__':
    main()

