"""Metrics for evaluation."""

import numpy as np
from typing import Tuple
from sklearn.metrics import confusion_matrix


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute accuracy.
    
    Args:
        y_pred: Predicted labels (N,) or logits (N, C)
        y_true: True labels (N,)
        
    Returns:
        Accuracy score
    """
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_pred == y_true)


def compute_confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_pred: Predicted labels (N,) or logits (N, C)
        y_true: True labels (N,)
        num_classes: Number of classes
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))


def get_misclassified_samples(
    X: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    max_samples: int = 24
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get misclassified samples.
    
    Args:
        X: Input images (N, C, H, W)
        y_pred: Predicted labels (N,) or logits (N, C)
        y_true: True labels (N,)
        max_samples: Maximum number of samples to return
        
    Returns:
        (X_mis, y_pred_mis, y_true_mis)
    """
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    mis_mask = y_pred != y_true
    X_mis = X[mis_mask]
    y_pred_mis = y_pred[mis_mask]
    y_true_mis = y_true[mis_mask]
    
    if len(X_mis) > max_samples:
        indices = np.random.choice(len(X_mis), max_samples, replace=False)
        X_mis = X_mis[indices]
        y_pred_mis = y_pred_mis[indices]
        y_true_mis = y_true_mis[indices]
    
    return X_mis, y_pred_mis, y_true_mis

