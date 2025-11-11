"""Dataset loaders for MNIST and CIFAR-10 (NumPy only, no torch)."""

import numpy as np
import os
import gzip
import pickle
import urllib.request
from typing import Tuple
from pathlib import Path


def download_file(url: str, filepath: str, max_retries: int = 3) -> None:
    """Download a file from URL if it doesn't exist."""
    if os.path.exists(filepath):
        return
    
    print(f"Downloading {url}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded to {filepath}")
            return
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
            else:
                raise Exception(f"Failed to download {url} after {max_retries} attempts: {e}")


def load_mnist(data_dir: str = "./data", normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST dataset.
    
    Args:
        data_dir: Directory to store/load data
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        (X_train, y_train, X_test, y_test)
        X shapes: (N, 1, 28, 28)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    # URLs for MNIST - try multiple sources
    base_urls = [
        "https://storage.googleapis.com/cvdf-datasets/mnist/",  # Google Cloud mirror
        "http://yann.lecun.com/exdb/mnist/",  # Original (may not work)
    ]
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    # Download files - try multiple URLs
    for key, filename in files.items():
        filepath = data_dir / filename
        if filepath.exists():
            continue
        
        downloaded = False
        for base_url in base_urls:
            try:
                url = base_url + filename
                download_file(url, str(filepath))
                downloaded = True
                break
            except Exception as e:
                print(f"Failed to download from {url}: {e}")
                continue
        
        if not downloaded:
            raise Exception(f"Failed to download {filename} from all sources")
    
    def read_idx_images(filepath: str) -> np.ndarray:
        """Read IDX image file."""
        with gzip.open(filepath, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            n_images = int.from_bytes(f.read(4), 'big')
            n_rows = int.from_bytes(f.read(4), 'big')
            n_cols = int.from_bytes(f.read(4), 'big')
            
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(n_images, n_rows, n_cols)
            return images
    
    def read_idx_labels(filepath: str) -> np.ndarray:
        """Read IDX label file."""
        with gzip.open(filepath, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            n_labels = int.from_bytes(f.read(4), 'big')
            
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
    
    # Load data
    X_train = read_idx_images(str(data_dir / files['train_images']))
    y_train = read_idx_labels(str(data_dir / files['train_labels']))
    X_test = read_idx_images(str(data_dir / files['test_images']))
    y_test = read_idx_labels(str(data_dir / files['test_labels']))
    
    # Reshape to (N, 1, 28, 28)
    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]
    
    # Normalize
    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
    else:
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
    
    return X_train, y_train, X_test, y_test


def load_cifar10(data_dir: str = "./data", normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CIFAR-10 dataset.
    
    Args:
        data_dir: Directory to store/load data
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        (X_train, y_train, X_test, y_test)
        X shapes: (N, 3, 32, 32)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = data_dir / "cifar-10-python.tar.gz"
    extract_dir = data_dir / "cifar-10-batches-py"
    
    # Download if needed
    if not extract_dir.exists():
        if not tar_path.exists():
            download_file(url, str(tar_path))
        
        # Extract
        import tarfile
        print(f"Extracting {tar_path}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(data_dir)
        print("Extraction complete.")
    
    def load_batch(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a CIFAR-10 batch file."""
        with open(filepath, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        
        images = batch[b'data']
        labels = np.array(batch[b'labels'])
        
        # Reshape: (N, 3072) -> (N, 3, 32, 32)
        images = images.reshape(-1, 3, 32, 32)
        
        return images, labels
    
    # Load training batches
    X_train_list = []
    y_train_list = []
    
    for i in range(1, 6):
        batch_path = extract_dir / f"data_batch_{i}"
        X_batch, y_batch = load_batch(str(batch_path))
        X_train_list.append(X_batch)
        y_train_list.append(y_batch)
    
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    
    # Load test batch
    test_batch_path = extract_dir / "test_batch"
    X_test, y_test = load_batch(str(test_batch_path))
    
    # Normalize
    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
    else:
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
    
    return X_train, y_train, X_test, y_test

