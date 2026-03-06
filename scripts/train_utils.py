# train_utils.py

import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms as T
from tqdm import tqdm
from scripts.data_augmentation import RuntimeAugmenter

# Initialize the random augmenter
augmenter = None

# Statistics of CIFAR-100 dataset
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

def load_transforms(train=False):
    """Load transforms for CIFAR-100 dataset."""
    if train:
        # Training: Basic augmentations + RandAugment
        # NO normalization here - happens in RuntimeAugmenter
        return T.Compose([
            T.RandomCrop(32, padding=4, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
        ])
    else:
        # Validation: Just ToTensor + Normalize
        return T.Compose([
            T.ToTensor(),
            T.Normalize(CIFAR100_MEAN, CIFAR100_STD)
        ])

def load_data(data_dir, batch_size=256, max_epochs=200):
    """Load CIFAR-100 dataset with DataLoader."""
    global augmenter
    augmenter = RuntimeAugmenter(total_epochs=max_epochs)

    train_dataset = datasets.ImageFolder(
        root=data_dir + '/raw/train', 
        transform=load_transforms(train=True)
    )
    val_dataset = datasets.ImageFolder(
        root=data_dir + '/raw/val', 
        transform=load_transforms(train=False)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )

    print(f"Dataset loaded from: {data_dir}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    return train_loader, val_loader

def define_loss_and_optimizer(model, lr, weight_decay, max_epochs):
    """Define the loss function and optimizer for training."""
    
    # CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    
    # AdamW optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay, 
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with warmup
    warmup_epochs = 20
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=warmup_epochs
            ),
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs - warmup_epochs, eta_min=5e-7
            )
        ],
        milestones=[warmup_epochs]
    )
    
    print(f"Learning rate: {lr:.6f}")
    print(f"Weight decay: {weight_decay}")
    
    return criterion, optimizer, scheduler

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train the model for one epoch."""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply runtime data augmentation
        inputs, labels, lam = augmenter(inputs, labels, epoch)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        label_a, label_b = labels if isinstance(labels, tuple) else (labels, None)
        if label_b is not None:
            loss = lam * criterion(outputs, label_a) + (1 - lam) * criterion(outputs, label_b)
        else:
            loss = criterion(outputs, label_a)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (helps stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        predicted = outputs.max(1)[1]
        total += label_a.size(0)
        
        if label_b is not None:
            correct += (lam * predicted.eq(label_a).sum().item() + 
                       (1 - lam) * predicted.eq(label_b).sum().item())
        else:
            correct += predicted.eq(label_a).sum().item()
        
        progress_bar.set_postfix({
            "Loss": f"{loss.item():.4f}", 
            "Acc": f"{100.0 * correct / total:.2f}%"
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "Acc": f"{100.0 * correct / total:.2f}%"
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc

def save_checkpoint(state, filename):
    """Save model checkpoint."""
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file {filename} not found")
    
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    return checkpoint

def save_metrics(metrics: str, filename: str = "training_metrics.txt"):
    """Save training metrics to a file."""
    with open(filename, 'w') as f:
        f.write(metrics)