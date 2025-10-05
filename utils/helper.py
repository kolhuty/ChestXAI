"""Helper utilities for training and data processing."""

import torch
import pandas as pd


def compute_pos_weight(train_df: pd.DataFrame, label_columns: list[str], device: torch.device) -> torch.Tensor:
    """Compute positive class weights for handling class imbalance."""
    # Count positive samples for each class
    pos_counts = train_df[label_columns].sum(axis=0).values
    
    # Count negative samples for each class
    neg_counts = len(train_df) - pos_counts
    
    # Compute positive weights (neg_count / pos_count)
    # Add small epsilon to prevent division by zero
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-5), dtype=torch.float32).to(device)
    
    return pos_weight
