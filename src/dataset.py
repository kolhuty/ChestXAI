"""PyTorch Dataset class for chest X-ray images."""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
import os


class ChestXRayDataset(Dataset):
    """PyTorch Dataset for chest X-ray images and disease labels."""
    
    def __init__(self, df: pd.DataFrame, image_dir: str, img_size: tuple[int, int],
                 is_test: bool = False, label_cols: list[str] | None = None, 
                 transform=None) -> None:
        """Initialize the dataset."""
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.img_size = img_size
        self.is_test = is_test
        self.label_cols = label_cols
        self.transform = transform

        # Validate that the image directory exists
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory {self.image_dir} not found.")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        # Get the row data for this index
        row = self.df.iloc[idx]
        img_name = row['Image_name']
        img_path = os.path.join(self.image_dir, img_name)

        # Load grayscale image using OpenCV
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Handle corrupted or missing images by creating a black image
        if img is None or img.size == 0:
            img = np.zeros(self.img_size, dtype=np.uint8)
        else:
            # Resize image to target size
            img = cv2.resize(img, self.img_size)

        # Add channel dimension for albumentations (H, W, 1)
        img = np.expand_dims(img, axis=-1)

        # Apply data augmentation if transform is provided
        if self.transform:
            # Albumentations expects a dictionary with 'image' key
            img = self.transform(image=img)["image"]
        else:
            # Fallback: convert to tensor using torchvision transforms
            img = transforms.ToTensor()(img)

        # Return only image for test datasets, image and labels for training
        if self.is_test:
            return img
        else:
            # Convert labels to float32 tensor
            labels = torch.tensor(row[self.label_cols].values.astype(np.float32))
            return img, labels