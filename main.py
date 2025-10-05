"""
Main training script for chest X-ray disease classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models

from config.config import load_config
from src.data_loader import load_train_data
from src.transforms import get_train_transform, get_val_transform
from src.dataset import ChestXRayDataset
from torch.utils.data import DataLoader
from src.model import CNNClassifier
from utils.helper import compute_pos_weight
from src.train import Trainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main() -> None:
    """Main training pipeline for chest X-ray classification."""
    # Load configuration parameters
    cfg = load_config()

    print(f"Using device: {DEVICE}")

    # Load and split training data
    train_data, val_data = load_train_data(cfg['csv_dir_train'], cfg['label_columns'], cfg['subset_frac'])
    
    # Get data augmentation transforms
    train_transform = get_train_transform()  # Includes augmentations
    val_transform = get_val_transform()      # No augmentations, only normalization

    # Create datasets
    train_dataset = ChestXRayDataset(train_data, cfg['image_dir_train'], img_size=cfg['img_size'],
                                     is_test=False, label_cols=cfg['label_columns'], transform=train_transform)

    val_dataset = ChestXRayDataset(val_data, cfg['image_dir_train'], img_size=cfg['img_size'],
                                   is_test=False, label_cols=cfg['label_columns'], transform=val_transform)

    # Create data loaders with appropriate settings
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=3, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=3, pin_memory=True)

    # Initialize pre-trained backbone model
    if hasattr(models, cfg['model_name']):
        backbone_fn = getattr(models, cfg['model_name'])
    else:
        raise ValueError(f"Model {cfg['model_name']} not found in torchvision.models")

    # Load pre-trained backbone with ImageNet weights
    backbone = backbone_fn(weights="IMAGENET1K_V1")
    
    # Modify first convolutional layer to accept grayscale input (1 channel instead of 3)
    orig_conv = backbone.features[0][0]
    backbone.features[0][0] = nn.Conv2d(
        1,  # Single channel for grayscale images
        orig_conv.out_channels,
        kernel_size=orig_conv.kernel_size,
        stride=orig_conv.stride,
        padding=orig_conv.padding,
        bias=False
    )
    
    # Extract feature dimension and remove original classifier
    feature_dim = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()

    # Create custom classifier for disease classification
    model = CNNClassifier(backbone, feature_dim=feature_dim, num_classes=cfg['num_classes'])
    model.to(DEVICE)

    # Set up differential learning rates for different model components
    first_layer_params = list(model.backbone.features[0][0].parameters())  # New grayscale layer
    other_backbone_params = [p for n, p in model.backbone.named_parameters()
                             if "features.0.0" not in n]  # Pre-trained layers
    fc_params = model.fc.parameters()  # Custom classifier

    # Optimizer with different learning rates for different components
    optimizer = torch.optim.Adam([
        {'params': first_layer_params, 'lr': 1e-3},      # Higher LR for new grayscale layer
        {'params': other_backbone_params, 'lr': cfg['lr']},  # Lower LR for pre-trained layers
        {'params': fc_params, 'lr': 1e-3}                # Higher LR for custom classifier
    ])

    # Compute class weights to handle dataset imbalance
    pos_weight = compute_pos_weight(train_data, label_columns=cfg['label_columns'], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Initialize trainer and start training
    trainer = Trainer(model, optimizer, criterion, train_loader, val_loader, device=DEVICE, config=cfg)
    trainer.fit()

if __name__ == "__main__":
    main()
