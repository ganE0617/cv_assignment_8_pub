# Programming Assignment 08: CNN Implementation

This repository contains a complete implementation of Convolutional Neural Networks (CNNs) from scratch using NumPy, as well as PyTorch implementations for comparison. The project includes training scripts, experiment runners, and visualization tools for MNIST and CIFAR-10 datasets.

## Repository Structure

```
.
├── src/
│   ├── np_cnn/              # NumPy CNN implementation from scratch
│   │   ├── layers.py        # Conv2D, ReLU, MaxPool2D, Flatten, Linear
│   │   ├── losses.py        # Softmax cross-entropy loss
│   │   ├── optim.py         # SGD optimizer with weight decay
│   │   ├── model.py         # Sequential model and CNN builder
│   │   ├── datasets.py      # MNIST & CIFAR-10 loaders (no torch)
│   │   ├── metrics.py       # Accuracy, confusion matrix helpers
│   │   └── utils.py         # Seeding, padding, minibatches, timer
│   └── plots/
│       └── plotting.py      # Visualization utilities
├── train_numpy_cnn.py       # NumPy CNN training script
├── experiments_numpy.py  # Run multiple experiments
├── train_pytorch_cnn.py     # PyTorch CNN training script
├── visualize_features.py    # Feature map visualization
├── reports/
│   └── template_report.md   # Report template
├── outputs/
│   ├── logs/                # CSV logs and summaries
│   ├── plots/               # Training curves
│   ├── confusion/           # Confusion matrices
│   ├── misclassified/      # Misclassified sample grids
│   └── features/            # Feature map visualizations
├── requirements.txt
├── README.md
└── LICENSE
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### NumPy CNN Training

**MNIST:**
```bash
python train_numpy_cnn.py --dataset mnist --epochs 10 --batch_size 128 --lr 0.01 --weight_decay 1e-4 --seed 42
```

**CIFAR-10:**
```bash
python train_numpy_cnn.py --dataset cifar10 --epochs 15 --batch_size 128 --lr 0.01 --weight_decay 5e-4 --seed 42
```

**Additional Options:**
- `--filters "8,16"`: Comma-separated filter counts (default: "8,16" for MNIST, "32,64" for CIFAR-10)
- `--kernel_size 3`: Kernel size (default: 3)
- `--no_pool`: Remove pooling layers
- `--save_prefix STR`: Custom prefix for output files

### Experiments

Run multiple experiment variants:
```bash
python experiments_numpy.py --dataset mnist --seed 42
```

This will run several experiments varying:
- Filter counts (e.g., 8,16 vs 16,32)
- Kernel sizes (3 vs 5)
- Pooling (with/without)
- Learning rates (0.1, 0.01, 0.001)

Results are saved to `outputs/logs/experiments_numpy.csv` and comparison plot to `outputs/plots/numpy_experiments_acc.png`.

### PyTorch CNN Training

**MNIST:**
```bash
python train_pytorch_cnn.py --dataset mnist --epochs 8 --batch_size 128 --lr 0.01 --seed 42
```

**CIFAR-10:**
```bash
python train_pytorch_cnn.py --dataset cifar10 --epochs 20 --batch_size 128 --lr 0.001 --seed 42 --adam
```

**Options:**
- `--adam`: Use Adam optimizer (default: SGD with momentum=0.9)
- `--weight_decay FLOAT`: Weight decay (default: 1e-4)

### Feature Map Visualization

**MNIST:**
```bash
python visualize_features.py --dataset mnist --class_idx 5
```

**CIFAR-10:**
```bash
python visualize_features.py --dataset cifar10 --class_idx 3
```

**Options:**
- `--checkpoint PATH`: Path to saved model checkpoint (optional)
- `--class_idx INT`: Class index to visualize (e.g., 5 for digit 5)

## Architecture Details

### NumPy CNN Baseline (MNIST)

```
Conv2D(1→8, k=3, padding='same') → ReLU → MaxPool2D(2)
Conv2D(8→16, k=3, padding='same') → ReLU → MaxPool2D(2)
Flatten → Linear(16*7*7 → 64) → ReLU → Linear(64 → 10)
```

### NumPy CNN Baseline (CIFAR-10)

```
Conv2D(3→32, k=3, padding='same') → ReLU → MaxPool2D(2)
Conv2D(32→64, k=3, padding='same') → ReLU → MaxPool2D(2)
Flatten → Linear(64*8*8 → 64) → ReLU → Linear(64 → 10)
```

### PyTorch CNN (MNIST)

```
Conv(1→16, 3, pad=1) → ReLU → MaxPool(2)
Conv(16→32, 3, pad=1) → ReLU → MaxPool(2)
Flatten → Linear(32*7*7→128) → ReLU → Linear(128→10)
```

### PyTorch CNN (CIFAR-10)

```
Conv(3→32, 3, pad=1) → ReLU → MaxPool(2)
Conv(32→64, 3, pad=1) → ReLU → MaxPool(2)
Flatten → Linear(64*8*8→256) → ReLU → Linear(256→10)
```

## Modifying Architecture

### Changing Filters

Use the `--filters` argument:
```bash
python train_numpy_cnn.py --dataset mnist --filters "16,32" --epochs 10
```

### Changing Kernel Size

```bash
python train_numpy_cnn.py --dataset mnist --kernel_size 5 --epochs 10
```

### Removing Pooling

```bash
python train_numpy_cnn.py --dataset mnist --no_pool --epochs 10
```

### Modifying Code

To change the architecture in code, edit `src/np_cnn/model.py`:
- Modify `build_cnn()` function to add/remove layers
- Adjust filter counts, kernel sizes, or add more conv blocks
- Change fully connected layer sizes

## Output Locations

All outputs are saved to the `outputs/` directory:

### Training Curves
- `outputs/plots/mnist_numpy_loss_acc.png`
- `outputs/plots/cifar10_numpy_loss_acc.png`
- `outputs/plots/mnist_torch_loss_acc.png`
- `outputs/plots/cifar10_torch_loss_acc.png`
- `outputs/plots/numpy_experiments_acc.png` (experiments comparison)

### Confusion Matrices
- `outputs/confusion/mnist_numpy_confusion.png`
- `outputs/confusion/cifar10_numpy_confusion.png`
- `outputs/confusion/mnist_torch_confusion.png`
- `outputs/confusion/cifar10_torch_confusion.png`

### Misclassified Samples
- `outputs/misclassified/mnist_numpy_misclf.png`
- `outputs/misclassified/cifar10_numpy_misclf.png`
- `outputs/misclassified/mnist_torch_misclf.png`
- `outputs/misclassified/cifar10_torch_misclf.png`

### Feature Maps
- `outputs/features/mnist_input.png`
- `outputs/features/mnist_torch_conv1_featuremaps.png`
- `outputs/features/cifar10_input.png`
- `outputs/features/cifar10_torch_conv1_featuremaps.png`

### Logs
- `outputs/logs/{dataset}_numpy_log.csv`
- `outputs/logs/{dataset}_numpy_summary.json`
- `outputs/logs/{dataset}_torch_log.csv`
- `outputs/logs/experiments_numpy.csv`

## Reproducibility

All scripts use a random seed (default: 42) for reproducibility:
- NumPy: `set_seed(seed)` sets NumPy random seed
- PyTorch: `torch.manual_seed(seed)` and `torch.cudnn.deterministic = True`

To change the seed, use the `--seed` argument:
```bash
python train_numpy_cnn.py --dataset mnist --seed 123
```

## Implementation Details

### NumPy CNN Components

1. **Layers**: All layers implement `forward()` and `backward()` methods
2. **Loss**: Numerically stable softmax cross-entropy (subtracts max per sample)
3. **Optimizer**: SGD with L2 weight decay (applied to weights only, not biases)
4. **Datasets**: MNIST and CIFAR-10 loaders without PyTorch dependencies

### Key Features

- **From Scratch**: NumPy CNN implemented without deep learning frameworks
- **Full Backpropagation**: All gradients computed manually
- **Numerical Stability**: Softmax uses max subtraction trick
- **Modular Design**: Easy to modify architecture and hyperparameters

## Requirements

- Python 3.10+
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0
- tqdm >= 4.62.0
- scikit-learn >= 1.0.0
- PyTorch >= 1.10.0
- torchvision >= 0.11.0

## License

MIT License - see LICENSE file for details.

## Author

[Your Name]  
Assignment 08: CNN Implementation

