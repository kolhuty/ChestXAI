"""Inference utilities for chest X-ray disease classification."""

import tqdm
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


def run_inference(model: torch.nn.Module, test_dataset: torch.utils.data.Dataset, 
                 device: torch.device, batch_size: int = 8) -> np.ndarray:
    """Run inference on a test dataset using a trained model."""
    # Create data loader for test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
    
    # Set model to evaluation mode
    model.eval()
    all_preds = []
    
    # Run inference without gradient computation
    with torch.no_grad():
        for imgs in tqdm(test_loader, desc="Inference"):
            # Move images to device
            imgs = imgs.to(device)
            
            # Get model predictions
            logits = model(imgs)
            
            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
    
    # Concatenate all predictions
    return np.concatenate(all_preds, axis=0)


def create_submission(preds: np.ndarray, sample_submission_path: str, 
                     label_columns: list[str], output_path: str) -> None:
    """Create a submission file from model predictions."""
    # Load the sample submission file
    submission = pd.read_csv(sample_submission_path)
    
    # Replace placeholder values with actual predictions
    submission[label_columns] = preds
    
    # Save the submission file
    submission.to_csv(output_path, index=False)