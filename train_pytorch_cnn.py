"""Train PyTorch CNN for MNIST and CIFAR-10."""

import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from src.plots.plotting import plot_training_curves, plot_confusion_matrix, plot_misclassified_grid


class CNN(nn.Module):
    """CNN model for MNIST and CIFAR-10."""
    
    def __init__(self, dataset: str = 'mnist'):
        """
        Initialize CNN.
        
        Args:
            dataset: 'mnist' or 'cifar10'
        """
        super(CNN, self).__init__()
        self.dataset = dataset
        
        if dataset == 'mnist':
            # MNIST: Conv(1→16, 3, pad=1) → ReLU → MaxPool(2)
            #        Conv(16→32, 3, pad=1) → ReLU → MaxPool(2)
            #        Flatten → Linear(32*7*7→128) → ReLU → Linear(128→10)
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        else:  # CIFAR-10
            # CIFAR-10: Conv(3→32, 3, pad=1) → ReLU → MaxPool(2)
            #           Conv(32→64, 3, pad=1) → ReLU → MaxPool(2)
            #           Flatten → Linear(64*8*8→256) → ReLU → Linear(256→10)
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    return train_loss / len(train_loader), 100.0 * correct / total


def evaluate(model, data_loader, criterion, device):
    """Evaluate on dataset."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_logits.append(output.cpu().numpy())
    
    all_logits = np.concatenate(all_logits, axis=0)
    
    return test_loss / len(data_loader), 100.0 * correct / total, all_logits, np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description='Train PyTorch CNN')
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10'],
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--adam', action='store_true', help='Use Adam optimizer')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set defaults based on dataset
    if args.epochs is None:
        args.epochs = 8 if args.dataset == 'mnist' else 20
    if args.lr is None:
        args.lr = 0.01 if args.dataset == 'mnist' else 0.001
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading
    print(f"Loading {args.dataset}...")
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Model
    model = CNN(dataset=args.dataset).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.adam:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # Training
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    # Setup logging
    log_file = Path(f"outputs/logs/{args.dataset}_torch_log.csv")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
    
    print("\nStarting training...")
    best_test_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        test_loss, test_acc, test_logits, test_labels = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        print(f"Epoch {epoch}/{args.epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Log to CSV
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
    
    # Final evaluation for confusion matrix and misclassified
    print("\nGenerating final predictions...")
    final_test_loss, final_test_acc, final_logits, final_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    # Get misclassified samples
    model.eval()
    misclassified_images = []
    misclassified_preds = []
    misclassified_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            mis_mask = predicted != target
            if mis_mask.any():
                misclassified_images.append(data[mis_mask].cpu())
                misclassified_preds.extend(predicted[mis_mask].cpu().numpy())
                misclassified_labels.extend(target[mis_mask].cpu().numpy())
            
            if len(misclassified_images) >= 24:
                break
    
    if misclassified_images:
        X_mis = torch.cat(misclassified_images[:24], dim=0).numpy()
        y_pred_mis = np.array(misclassified_preds[:24])
        y_true_mis = np.array(misclassified_labels[:24])
    else:
        X_mis = np.array([])
        y_pred_mis = np.array([])
        y_true_mis = np.array([])
    
    # Save plots
    print("\nSaving plots...")
    plot_path = Path(f"outputs/plots/{args.dataset}_torch_loss_acc.png")
    plot_training_curves(train_losses, train_accs, test_losses, test_accs, str(plot_path))
    
    # Confusion matrix
    cm = confusion_matrix(final_labels, np.argmax(final_logits, axis=1))
    cm_path = Path(f"outputs/confusion/{args.dataset}_torch_confusion.png")
    class_names = [str(i) for i in range(10)]
    plot_confusion_matrix(cm, str(cm_path), class_names)
    
    # Misclassified samples
    if len(X_mis) > 0:
        mis_path = Path(f"outputs/misclassified/{args.dataset}_torch_misclf.png")
        plot_misclassified_grid(X_mis, y_pred_mis, y_true_mis, str(mis_path), dataset=args.dataset)
    
    # Save model checkpoint
    checkpoint_path = Path(f"outputs/logs/{args.dataset}_torch_model.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to: {checkpoint_path}")
    
    print(f"\nTraining complete!")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Final test accuracy: {final_test_acc:.2f}%")
    print(f"\nOutputs saved to:")
    print(f"  - Logs: {log_file}")
    print(f"  - Plots: {plot_path}")
    print(f"  - Confusion: {cm_path}")
    if len(X_mis) > 0:
        print(f"  - Misclassified: {mis_path}")
    print(f"\nTo visualize feature maps, run:")
    print(f"  python visualize_features.py --dataset {args.dataset} --checkpoint {checkpoint_path}")


if __name__ == '__main__':
    main()

