# data_augmentation.py

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

# Statistics of CIFAR-100 dataset
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

class RuntimeAugmenter:
    """
    Runtime data augmentation pipeline for CIFAR-100.
    Combines Mixup/CutMix and normalization.
    Basic augmentations (Crop, Flip, RandAugment) are handled by DataLoader.
    """
    def __init__(self, mix_prob=0.95, total_epochs=200):
        """
        Args:
            mix_prob: Probability of applying Mixup/CutMix (default: 0.95)
            total_epochs: Total number of training epochs
        """
        self.mix_prob = mix_prob
        self.total_epochs = total_epochs
        
        # Normalization transform
        self.normalize = T.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD)

    def get_curriculum_strength(self, epoch):
        """
        Progressive curriculum: ramp up from 0 to 1.0 over first 50% of training.
        This makes early training easier and more stable.
        """
        warmup_epochs = 40
        if epoch < warmup_epochs:
            # Smooth ramp using cosine
            return 0.5 * (1 - np.cos(np.pi * epoch / warmup_epochs))
        return 1.0

    def mixup(self, images, labels, alpha=1.0):
        """
        Apply Mixup augmentation.
        Args:
            images: Batch of images [B, C, H, W]
            labels: Batch of labels [B]
            alpha: Beta distribution parameter
        Returns:
            Mixed images, (labels_a, labels_b), lambda
        """
        batch_size = images.size(0)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(alpha, alpha)
        
        # Random permutation
        indices = torch.randperm(batch_size, device=images.device)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # Return mixed labels
        labels_a = labels
        labels_b = labels[indices]
        
        return mixed_images, (labels_a, labels_b), lam

    def cutmix(self, images, labels, alpha=1.0):
        """
        Apply CutMix augmentation.
        Args:
            images: Batch of images [B, C, H, W]
            labels: Batch of labels [B]
            alpha: Beta distribution parameter
        Returns:
            Mixed images, (labels_a, labels_b), lambda
        """
        batch_size = images.size(0)
        _, _, H, W = images.shape
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(alpha, alpha)
        
        # Random permutation
        indices = torch.randperm(batch_size, device=images.device)
        
        # Generate random bounding box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_ratio)
        cut_h = int(H * cut_ratio)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        
        # Return mixed labels
        labels_a = labels
        labels_b = labels[indices]
        
        return mixed_images, (labels_a, labels_b), lam

    def __call__(self, images, labels, epoch):
        """
        Apply augmentation pipeline.
        
        Args:
            images: Batch of images [B, C, H, W], float32 in [0, 1], NOT normalized
            labels: Batch of labels [B]
            epoch: Current training epoch
            
        Returns:
            Augmented images (normalized), labels (tuple if mixed), lambda value
        """
        # Get curriculum strength (progressive difficulty)
        strength = self.get_curriculum_strength(epoch)
        
        # Apply Mixup or CutMix with probability
        lam = 1.0
        if torch.rand(1).item() < self.mix_prob * strength:
            if torch.rand(1).item() < 0.5:
                # Apply Mixup
                images, labels, lam = self.mixup(images, labels, alpha=1.0)
            else:
                # Apply CutMix
                images, labels, lam = self.cutmix(images, labels, alpha=1.0)
        else:
            # No mixing - keep original labels
            labels = (labels, None)
        
        # Normalize (final step)
        images = self.normalize(images)
        
        return images, labels, lam