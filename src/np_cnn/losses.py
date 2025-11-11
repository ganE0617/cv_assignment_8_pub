"""Loss functions for NumPy CNN."""

import numpy as np
from typing import Tuple


def softmax_cross_entropy(
    logits: np.ndarray,
    y: np.ndarray,
    return_grad: bool = True
) -> Tuple[float, np.ndarray]:
    """
    Numerically stable softmax cross-entropy loss.
    
    Args:
        logits: Raw scores of shape (N, C)
        y: True labels of shape (N,) with values in [0, C-1]
        return_grad: Whether to return gradient
        
    Returns:
        (loss, grad) if return_grad else (loss, None)
    """
    N, C = logits.shape
    
    # Numerical stability: subtract max per sample
    logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
    
    # Compute softmax
    exp_logits = np.exp(logits_shifted)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Compute cross-entropy loss
    log_probs = logits_shifted - np.log(np.sum(exp_logits, axis=1, keepdims=True))
    loss = -np.mean(log_probs[np.arange(N), y])
    
    if not return_grad:
        return loss, None
    
    # Gradient: softmax - one_hot
    grad = softmax_probs.copy()
    grad[np.arange(N), y] -= 1.0
    grad /= N
    
    return loss, grad


def one_hot_encode(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    One-hot encode labels.
    
    Args:
        y: Labels of shape (N,)
        num_classes: Number of classes
        
    Returns:
        One-hot encoded array of shape (N, num_classes)
    """
    N = y.shape[0]
    one_hot = np.zeros((N, num_classes))
    one_hot[np.arange(N), y] = 1.0
    return one_hot

