"""Results saving utilities for experiment tracking."""

import os
import pandas as pd


def save_results_csv(model_name: str, history: dict, num_epochs: int, lr: float, 
                    filepath: str = "results_without_metadata_efficientnet2.csv") -> None:
    """Save training results to a CSV file for experiment tracking."""
    # Create a row with the experiment results
    row = {
        "Model": model_name,
        "AUC": max(history['val']['auc']),  # Best validation AUC
        "F1": max(history['val']['f1']),    # Best validation F1-score
        "epoch": num_epochs,
        "learning rate": lr
    }

    # Append to existing CSV or create new one
    if os.path.exists(filepath):
        # Load existing results and append new row
        df = pd.read_csv(filepath)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        # Create new DataFrame with the results
        df = pd.DataFrame([row])

    # Save to CSV file
    df.to_csv(filepath, index=False)