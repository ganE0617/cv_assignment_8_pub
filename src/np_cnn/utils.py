"""Utility functions for NumPy CNN implementation."""

import numpy as np
from typing import Tuple, Optional, Union
import time


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def pad2d(X: np.ndarray, padding: int) -> np.ndarray:
    """
    Pad 2D images with zeros.
    
    Args:
        X: Input array of shape (N, C, H, W)
        padding: Padding size
        
    Returns:
        Padded array of shape (N, C, H+2*padding, W+2*padding)
    """
    if padding == 0:
        return X
    return np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')


def compute_padding_output_size(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: Union[str, int]
) -> Tuple[int, int]:
    """
    Compute padding and output size for convolution.
    
    Args:
        input_size: Input dimension (H or W)
        kernel_size: Kernel size
        stride: Stride
        padding: 'same', 'valid', or integer
        
    Returns:
        (padding_size, output_size)
    """
    if padding == 'same':
        padding_size = (kernel_size - 1) // 2
        output_size = (input_size + 2 * padding_size - kernel_size) // stride + 1
    elif padding == 'valid':
        padding_size = 0
        output_size = (input_size - kernel_size) // stride + 1
    else:
        padding_size = int(padding)
        output_size = (input_size + 2 * padding_size - kernel_size) // stride + 1
    
    return padding_size, output_size


def minibatches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate minibatches from dataset.
    
    Args:
        X: Features array (N, ...)
        y: Labels array (N,)
        batch_size: Batch size
        shuffle: Whether to shuffle
        seed: Random seed for shuffling
        
    Yields:
        (X_batch, y_batch) tuples
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        if seed is not None:
            rng = np.random.RandomState(seed)
            rng.shuffle(indices)
        else:
            np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        print(f"{self.name} took {self.elapsed:.2f} seconds")

