"""CNN layers implemented from scratch in NumPy."""

import numpy as np
from typing import Tuple, Union


class Conv2D:
    """2D Convolution layer."""
    
    def __init__(
        self,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: Union[str, int] = 'same',
        bias: bool = True
    ):
        """
        Initialize Conv2D layer.
        
        Args:
            out_channels: Number of output channels
            kernel_size: Kernel size (int or (H, W))
            stride: Stride
            padding: 'same', 'valid', or integer
            bias: Whether to use bias
        """
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding
        self.bias = bias
        
        self.in_channels = None
        self.weights = None
        self.bias_weights = None
        self.grad_weights = None
        self.grad_bias = None
        
        self.input = None
        self.output = None
    
    def initialize(self, in_channels: int, input_height: int, input_width: int) -> None:
        """Initialize weights based on input dimensions."""
        self.in_channels = in_channels
        
        # He initialization
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        limit = np.sqrt(2.0 / fan_in)
        self.weights = np.random.randn(
            self.out_channels,
            in_channels,
            self.kernel_size[0],
            self.kernel_size[1]
        ) * limit
        
        if self.bias:
            self.bias_weights = np.zeros(self.out_channels)
        
        self.grad_weights = np.zeros_like(self.weights)
        if self.bias:
            self.grad_bias = np.zeros_like(self.bias_weights)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            X: Input of shape (N, C_in, H, W)
            
        Returns:
            Output of shape (N, C_out, H', W')
        """
        if self.weights is None:
            _, in_channels, h, w = X.shape
            self.initialize(in_channels, h, w)
        
        self.input = X.copy()
        N, C_in, H, W = X.shape
        
        # Compute padding
        if self.padding_mode == 'same':
            pad_h = (self.kernel_size[0] - 1) // 2
            pad_w = (self.kernel_size[1] - 1) // 2
        elif self.padding_mode == 'valid':
            pad_h = pad_w = 0
        else:
            pad_h = pad_w = int(self.padding_mode)
        
        # Pad input
        if pad_h > 0 or pad_w > 0:
            X_padded = np.pad(
                X,
                ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode='constant'
            )
        else:
            X_padded = X
        
        # Compute output dimensions
        H_out = (H + 2 * pad_h - self.kernel_size[0]) // self.stride + 1
        W_out = (W + 2 * pad_w - self.kernel_size[1]) // self.stride + 1
        
        # Initialize output
        output = np.zeros((N, self.out_channels, H_out, W_out))
        
        # Convolution (naive implementation)
        for n in range(N):
            for c_out in range(self.out_channels):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * self.stride
                        w_start = w_out * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]
                        
                        patch = X_padded[n, :, h_start:h_end, w_start:w_end]
                        output[n, c_out, h_out, w_out] = np.sum(
                            patch * self.weights[c_out]
                        )
                
                if self.bias:
                    output[n, c_out] += self.bias_weights[c_out]
        
        self.output = output
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.
        
        Args:
            grad_output: Gradient w.r.t. output (N, C_out, H', W')
            
        Returns:
            Gradient w.r.t. input (N, C_in, H, W)
        """
        X = self.input
        N, C_in, H, W = X.shape
        
        # Compute padding
        if self.padding_mode == 'same':
            pad_h = (self.kernel_size[0] - 1) // 2
            pad_w = (self.kernel_size[1] - 1) // 2
        elif self.padding_mode == 'valid':
            pad_h = pad_w = 0
        else:
            pad_h = pad_w = int(self.padding_mode)
        
        # Pad input for gradient computation
        if pad_h > 0 or pad_w > 0:
            X_padded = np.pad(
                X,
                ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                mode='constant'
            )
        else:
            X_padded = X
        
        H_out, W_out = grad_output.shape[2], grad_output.shape[3]
        
        # Initialize gradients
        grad_input = np.zeros_like(X_padded)
        self.grad_weights.fill(0.0)
        if self.bias:
            self.grad_bias.fill(0.0)
        
        # Backward pass
        for n in range(N):
            for c_out in range(self.out_channels):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * self.stride
                        w_start = w_out * self.stride
                        h_end = h_start + self.kernel_size[0]
                        w_end = w_start + self.kernel_size[1]
                        
                        grad_val = grad_output[n, c_out, h_out, w_out]
                        
                        # Gradient w.r.t. input
                        grad_input[n, :, h_start:h_end, w_start:w_end] += (
                            self.weights[c_out] * grad_val
                        )
                        
                        # Gradient w.r.t. weights
                        self.grad_weights[c_out] += (
                            X_padded[n, :, h_start:h_end, w_start:w_end] * grad_val
                        )
                
                # Gradient w.r.t. bias
                if self.bias:
                    self.grad_bias[c_out] += np.sum(grad_output[n, c_out])
        
        # Remove padding from grad_input
        if pad_h > 0 or pad_w > 0:
            grad_input = grad_input[:, :, pad_h:-pad_h, pad_w:-pad_w]
        
        return grad_input


class ReLU:
    """ReLU activation layer."""
    
    def __init__(self):
        self.input = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: max(0, x)."""
        self.input = X.copy()
        return np.maximum(0, X)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass: gradient is 1 where input > 0, else 0."""
        return grad_output * (self.input > 0)


class MaxPool2D:
    """2D Max Pooling layer."""
    
    def __init__(self, kernel_size: int = 2, stride: int = 2):
        """
        Initialize MaxPool2D layer.
        
        Args:
            kernel_size: Pooling kernel size
            stride: Stride
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None
        self.max_indices = None  # Store indices of max values
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            X: Input of shape (N, C, H, W)
            
        Returns:
            Output of shape (N, C, H', W')
        """
        self.input = X.copy()
        N, C, H, W = X.shape
        
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        
        output = np.zeros((N, C, H_out, W_out))
        self.max_indices = np.zeros((N, C, H_out, W_out, 2), dtype=np.int32)
        
        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_start = h_out * self.stride
                        w_start = w_out * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        
                        patch = X[n, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(patch)
                        output[n, c, h_out, w_out] = max_val
                        
                        # Find index of max value (flattened)
                        flat_idx = np.argmax(patch)
                        h_idx = h_start + flat_idx // self.kernel_size
                        w_idx = w_start + flat_idx % self.kernel_size
                        self.max_indices[n, c, h_out, w_out] = [h_idx, w_idx]
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: route gradients only to max indices.
        
        Args:
            grad_output: Gradient w.r.t. output (N, C, H', W')
            
        Returns:
            Gradient w.r.t. input (N, C, H, W)
        """
        N, C, H, W = self.input.shape
        grad_input = np.zeros_like(self.input)
        
        H_out, W_out = grad_output.shape[2], grad_output.shape[3]
        
        for n in range(N):
            for c in range(C):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        h_idx, w_idx = self.max_indices[n, c, h_out, w_out]
                        grad_input[n, c, h_idx, w_idx] += grad_output[n, c, h_out, w_out]
        
        return grad_input


class Flatten:
    """Flatten layer."""
    
    def __init__(self):
        self.input_shape = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: flatten (N, C, H, W) -> (N, C*H*W).
        
        Args:
            X: Input of shape (N, C, H, W)
            
        Returns:
            Output of shape (N, C*H*W)
        """
        self.input_shape = X.shape
        N = X.shape[0]
        return X.reshape(N, -1)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: reshape back.
        
        Args:
            grad_output: Gradient w.r.t. output (N, C*H*W)
            
        Returns:
            Gradient w.r.t. input (N, C, H, W)
        """
        return grad_output.reshape(self.input_shape)


class Linear:
    """Fully connected (linear) layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize Linear layer.
        
        Args:
            in_features: Input feature size
            out_features: Output feature size
            bias: Whether to use bias
        """
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        # Xavier initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weights = np.random.uniform(
            -limit, limit, (in_features, out_features)
        )
        
        if bias:
            self.bias_weights = np.zeros(out_features)
        
        self.grad_weights = np.zeros_like(self.weights)
        if bias:
            self.grad_bias = np.zeros_like(self.bias_weights)
        
        self.input = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: X @ W + b.
        
        Args:
            X: Input of shape (N, in_features)
            
        Returns:
            Output of shape (N, out_features)
        """
        self.input = X.copy()
        output = X @ self.weights
        if self.bias:
            output += self.bias_weights
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.
        
        Args:
            grad_output: Gradient w.r.t. output (N, out_features)
            
        Returns:
            Gradient w.r.t. input (N, in_features)
        """
        # Gradient w.r.t. input
        grad_input = grad_output @ self.weights.T
        
        # Gradient w.r.t. weights
        self.grad_weights = self.input.T @ grad_output
        
        # Gradient w.r.t. bias
        if self.bias:
            self.grad_bias = np.sum(grad_output, axis=0)
        
        return grad_input

