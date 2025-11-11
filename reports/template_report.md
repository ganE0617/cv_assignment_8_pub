# Programming Assignment 08: CNN Implementation Report

**Student:** [Your Name]  
**Date:** [Date]  
**Assignment:** Hands-on Practice — Implementing a CNN

---

## 1. Model Architecture

### NumPy CNN (from scratch)

| Layer | Type | Parameters | Output Shape |
|-------|------|------------|--------------|
| Input | - | - | (N, C, H, W) |
| Conv1 | Conv2D | filters=8, kernel=3, padding='same' | (N, 8, H, W) |
| Act1 | ReLU | - | (N, 8, H, W) |
| Pool1 | MaxPool2D | kernel=2, stride=2 | (N, 8, H/2, W/2) |
| Conv2 | Conv2D | filters=16, kernel=3, padding='same' | (N, 16, H/2, W/2) |
| Act2 | ReLU | - | (N, 16, H/2, W/2) |
| Pool2 | MaxPool2D | kernel=2, stride=2 | (N, 16, H/4, W/4) |
| Flatten | Flatten | - | (N, 16*H/4*W/4) |
| FC1 | Linear | in=16*7*7, out=64 | (N, 64) |
| Act3 | ReLU | - | (N, 64) |
| FC2 | Linear | in=64, out=10 | (N, 10) |

**Note:** For MNIST (28x28), after pooling: 28→14→7. For CIFAR-10 (32x32), after pooling: 32→16→8.

### PyTorch CNN

**MNIST Architecture:**
- Conv(1→16, 3, pad=1) → ReLU → MaxPool(2)
- Conv(16→32, 3, pad=1) → ReLU → MaxPool(2)
- Flatten → Linear(32*7*7→128) → ReLU → Linear(128→10)

**CIFAR-10 Architecture:**
- Conv(3→32, 3, pad=1) → ReLU → MaxPool(2)
- Conv(32→64, 3, pad=1) → ReLU → MaxPool(2)
- Flatten → Linear(64*8*8→256) → ReLU → Linear(256→10)

---

## 2. Training Curves

### NumPy CNN

![MNIST NumPy Training Curves](outputs/plots/mnist_numpy_loss_acc.png)

![CIFAR-10 NumPy Training Curves](outputs/plots/cifar10_numpy_loss_acc.png)

### PyTorch CNN

![MNIST PyTorch Training Curves](outputs/plots/mnist_torch_loss_acc.png)

![CIFAR-10 PyTorch Training Curves](outputs/plots/cifar10_torch_loss_acc.png)

---

## 3. Final Test Accuracy

| Dataset | NumPy CNN | PyTorch CNN |
|---------|-------------|-------------|
| MNIST   | TODO: [%]   | TODO: [%]   |
| CIFAR-10| TODO: [%]   | TODO: [%]   |

---

## 4. Confusion Matrix and Misclassified Samples

### NumPy CNN

**MNIST Confusion Matrix:**
![MNIST NumPy Confusion Matrix](outputs/confusion/mnist_numpy_confusion.png)

**MNIST Misclassified Samples:**
![MNIST NumPy Misclassified](outputs/misclassified/mnist_numpy_misclf.png)

**CIFAR-10 Confusion Matrix:**
![CIFAR-10 NumPy Confusion Matrix](outputs/confusion/cifar10_numpy_confusion.png)

**CIFAR-10 Misclassified Samples:**
![CIFAR-10 NumPy Misclassified](outputs/misclassified/cifar10_numpy_misclf.png)

### PyTorch CNN

**MNIST Confusion Matrix:**
![MNIST PyTorch Confusion Matrix](outputs/confusion/mnist_torch_confusion.png)

**MNIST Misclassified Samples:**
![MNIST PyTorch Misclassified](outputs/misclassified/mnist_torch_misclf.png)

**CIFAR-10 Confusion Matrix:**
![CIFAR-10 PyTorch Confusion Matrix](outputs/confusion/cifar10_torch_confusion.png)

**CIFAR-10 Misclassified Samples:**
![CIFAR-10 PyTorch Misclassified](outputs/misclassified/cifar10_torch_misclf.png)

---

## 5. Experiments (≥2 variants)

### Experiment Results

![Experiment Comparison](outputs/plots/numpy_experiments_acc.png)

**Experiment 1: [Name]**
- **Changes:** TODO: Describe what changed (e.g., filters, kernel size, pooling, LR)
- **Results:** TODO: Final test accuracy
- **Interpretation:** TODO: Why did this change affect performance?

**Experiment 2: [Name]**
- **Changes:** TODO: Describe what changed
- **Results:** TODO: Final test accuracy
- **Interpretation:** TODO: Why did this change affect performance?

**Experiment 3: [Name]** (if applicable)
- **Changes:** TODO
- **Results:** TODO
- **Interpretation:** TODO

---

## 6. Implementation Notes

### Design Choices

1. **Numerical Stability:** Used max subtraction in softmax to prevent overflow/underflow.
2. **Weight Initialization:** 
   - Conv layers: He initialization (sqrt(2/fan_in))
   - Linear layers: Xavier initialization
3. **Backpropagation:** Implemented naive convolution for clarity; im2col could be added for optimization.
4. **MaxPool Backward:** Routes gradients only to max indices (correct gradient flow).

### Bugs Fixed

- TODO: List any bugs encountered and how they were fixed

### Numerical Tricks

- TODO: Any numerical tricks used (e.g., gradient clipping, learning rate scheduling)

---

## 7. Reproducibility

**Random Seed:** 42 (set for NumPy and PyTorch)

**Environment:**
- Python: 3.10+
- NumPy: [version]
- PyTorch: [version]
- Other dependencies: See `requirements.txt`

**Hyperparameters:**

**NumPy CNN (MNIST):**
- Epochs: 10
- Batch size: 128
- Learning rate: 0.01
- Weight decay: 1e-4
- Filters: [8, 16]
- Kernel size: 3
- Pooling: Yes

**NumPy CNN (CIFAR-10):**
- Epochs: 15
- Batch size: 128
- Learning rate: 0.01
- Weight decay: 5e-4
- Filters: [32, 64]
- Kernel size: 3
- Pooling: Yes

**PyTorch CNN (MNIST):**
- Epochs: 8
- Batch size: 128
- Learning rate: 0.01
- Weight decay: 1e-4
- Optimizer: SGD (momentum=0.9)

**PyTorch CNN (CIFAR-10):**
- Epochs: 20
- Batch size: 128
- Learning rate: 0.001
- Weight decay: 1e-4
- Optimizer: Adam (or SGD)

---

## 8. Feature Map Visualization

### MNIST

**Input Image:**
![MNIST Input](outputs/features/mnist_input.png)

**Feature Maps (First Conv Layer):**
![MNIST Feature Maps](outputs/features/mnist_torch_conv1_featuremaps.png)

**Interpretation:** TODO: Write 3-5 sentences describing what the feature maps show. What patterns are the filters detecting? How do they relate to the input digit?

### CIFAR-10

**Input Image:**
![CIFAR-10 Input](outputs/features/cifar10_input.png)

**Feature Maps (First Conv Layer):**
![CIFAR-10 Feature Maps](outputs/features/cifar10_torch_conv1_featuremaps.png)

**Interpretation:** TODO: Write 3-5 sentences describing what the feature maps show. How do CIFAR-10 feature maps differ from MNIST? What visual features are being detected?

### Comparison (Optional)

TODO: Compare MNIST vs CIFAR-10 feature maps. What differences do you observe?

---

## 9. Conclusion

TODO: Summarize key findings, challenges encountered, and lessons learned.

