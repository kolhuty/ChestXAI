"""Data augmentation and preprocessing transforms for chest X-ray images."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform() -> A.Compose:
    """Get data augmentation pipeline for training data."""
    return A.Compose([
        # Resize image while maintaining aspect ratio
        A.LongestMaxSize(max_size=512),
        
        # Pad to square if needed (black padding)
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0),
        
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),  # 50% chance of horizontal flip
        A.ShiftScaleRotate(
            shift_limit=0.02,      # Small shifts (2% of image size)
            scale_limit=0.05,      # Small scaling (5% change)
            rotate_limit=5,        # Small rotations (5 degrees)
            border_mode=0,         # Black border for out-of-bounds
            p=0.3                  # 30% probability
        ),
        
        # Photometric augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.05,  # Small brightness changes (5%)
            contrast_limit=0.05,    # Small contrast changes (5%)
            p=0.3                   # 30% probability
        ),
        
        # CLAHE for better contrast in medical images
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        
        # Normalize to [-1, 1] range for better training stability
        A.Normalize(mean=(0.5,), std=(0.5,)),
        
        # Convert to PyTorch tensor
        ToTensorV2()
    ])


def get_val_transform() -> A.Compose:
    """Get preprocessing pipeline for validation/test data."""
    return A.Compose([
        # Resize image while maintaining aspect ratio
        A.LongestMaxSize(max_size=512),
        
        # Pad to square if needed (black padding)
        A.PadIfNeeded(min_height=512, min_width=512, border_mode=0),
        
        # Normalize to [-1, 1] range (same as training)
        A.Normalize(mean=(0.5,), std=(0.5,)),
        
        # Convert to PyTorch tensor
        ToTensorV2()
    ])
