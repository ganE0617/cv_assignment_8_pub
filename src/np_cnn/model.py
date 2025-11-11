"""CNN model composition utilities."""

from typing import List, Any
import numpy as np


class Sequential:
    """Sequential model container."""
    
    def __init__(self, layers: List[Any]):
        """
        Initialize Sequential model.
        
        Args:
            layers: List of layers in forward order
        """
        self.layers = layers
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through all layers.
        
        Args:
            X: Input array
            
        Returns:
            Output array
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, grad_output: np.ndarray) -> None:
        """
        Backward pass through all layers (in reverse).
        
        Args:
            grad_output: Gradient w.r.t. output
        """
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
    
    def get_trainable_layers(self) -> List[Any]:
        """Get list of layers with trainable parameters."""
        trainable = []
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                trainable.append(layer)
        return trainable


def build_cnn(
    input_channels: int,
    filters: List[int],
    kernel_size: int = 3,
    use_pooling: bool = True,
    num_classes: int = 10
) -> Sequential:
    """
    Build CNN model architecture.
    
    Args:
        input_channels: Number of input channels (1 for MNIST, 3 for CIFAR-10)
        filters: List of filter counts for each conv layer (e.g., [8, 16])
        kernel_size: Kernel size for conv layers
        use_pooling: Whether to use pooling layers
        num_classes: Number of output classes
        
    Returns:
        Sequential model
    """
    from .layers import Conv2D, ReLU, MaxPool2D, Flatten, Linear
    
    layers = []
    
    # First conv block
    layers.append(Conv2D(filters[0], kernel_size, padding='same'))
    layers.append(ReLU())
    if use_pooling:
        layers.append(MaxPool2D(kernel_size=2, stride=2))
    
    # Additional conv blocks
    for i in range(1, len(filters)):
        layers.append(Conv2D(filters[i], kernel_size, padding='same'))
        layers.append(ReLU())
        if use_pooling:
            layers.append(MaxPool2D(kernel_size=2, stride=2))
    
    # Flatten
    layers.append(Flatten())
    
    # Compute flattened size (assumes 28x28 input for MNIST, 32x32 for CIFAR-10)
    if input_channels == 1:  # MNIST
        if use_pooling:
            # After 2 pooling layers: 28 -> 14 -> 7
            flattened_size = filters[-1] * 7 * 7
        else:
            # No pooling: 28 -> 28
            flattened_size = filters[-1] * 28 * 28
    else:  # CIFAR-10
        if use_pooling:
            # After 2 pooling layers: 32 -> 16 -> 8
            flattened_size = filters[-1] * 8 * 8
        else:
            # No pooling: 32 -> 32
            flattened_size = filters[-1] * 32 * 32
    
    # Fully connected layers
    layers.append(Linear(flattened_size, 64))
    layers.append(ReLU())
    layers.append(Linear(64, num_classes))
    
    return Sequential(layers)

