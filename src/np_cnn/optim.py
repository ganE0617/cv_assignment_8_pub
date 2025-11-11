"""Optimizer for NumPy CNN."""

import numpy as np
from typing import List, Any


class SGD:
    """Stochastic Gradient Descent optimizer with weight decay."""
    
    def __init__(self, lr: float, weight_decay: float = 0.0):
        """
        Initialize SGD optimizer.
        
        Args:
            lr: Learning rate
            weight_decay: L2 weight decay coefficient (applied to weights only, not biases)
        """
        self.lr = lr
        self.weight_decay = weight_decay
    
    def step(self, layers: List[Any]) -> None:
        """
        Update parameters for all layers.
        
        Args:
            layers: List of layer objects with weights, bias_weights, grad_weights, grad_bias
        """
        for layer in layers:
            # Update weights with weight decay
            if hasattr(layer, 'weights') and hasattr(layer, 'grad_weights'):
                # Apply weight decay (L2 regularization)
                if self.weight_decay > 0:
                    layer.weights -= self.weight_decay * layer.weights
                # Gradient descent step
                layer.weights -= self.lr * layer.grad_weights
            
            # Update bias (no weight decay)
            if hasattr(layer, 'bias_weights') and hasattr(layer, 'grad_bias'):
                layer.bias_weights -= self.lr * layer.grad_bias

