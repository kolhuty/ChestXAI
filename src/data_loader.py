"""Data loading utilities for chest X-ray dataset."""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_train_data(path: str | bytes, label_columns: list[str], subset_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and split training data into train/validation sets."""
    # Load the training data CSV file
    try:
        train_df = pd.read_csv(path)
        print(f"Loaded train2.csv with {len(train_df)} rows")
    except FileNotFoundError:
        print("Error: train2.csv not found. Ensure dataset is attached.")
        raise

    # Validate that all required label columns exist in the dataset
    missing_cols = [col for col in label_columns if col not in train_df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns: {missing_cols}")

    # Sample a subset of the data if subset_frac < 1.0
    # This is useful for faster experimentation and development
    df_small = train_df.sample(frac=subset_frac, random_state=42)

    # Split into train, validation, test sets (85/10/5 split)
    train_data, val_test_data = train_test_split(df_small, test_size=0.15, random_state=42)
    val_data, test_data = train_test_split(val_test_data, test_size=0.33, random_state=42)

    print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")

    return train_data, val_data, test_data
