"""CNN classifier model for chest X-ray disease classification."""

import torch.nn as nn


class CNNClassifier(nn.Module):
    """CNN classifier for chest X-ray disease classification."""
    
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int) -> None:
        """Initialize the CNN classifier."""
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        # Extract features using the pre-trained backbone
        features = self.backbone(x)
        
        # Classify features using the custom classifier
        return self.fc(features)
