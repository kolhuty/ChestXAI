"""Visualization utilities for training results and model evaluation."""

import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curves(model_name: str, history: dict, num_epochs: int) -> None:
    """Plot learning curves for loss, AUC, and F1-score."""
    epochs = np.arange(1, num_epochs+1)

    plt.figure(figsize=(15,4))
    
    # Loss subplot
    plt.subplot(1,3,1)
    plt.plot(epochs, history['train']['loss'], label='train loss', marker='o')
    plt.plot(epochs, history['val']['loss'], label='val loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss {model_name}')
    plt.grid(True, alpha=0.3)
    
    # AUC subplot
    plt.subplot(1,3,2)
    plt.plot(epochs, history['train']['auc'], label='train AUC', marker='o')
    plt.plot(epochs, history['val']['auc'], label='val AUC', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.title(f'AUC {model_name}')
    plt.grid(True, alpha=0.3)
    
    # F1-score subplot
    plt.subplot(1,3,3)
    plt.plot(epochs, history['train']['f1'], label='train F1', marker='o')
    plt.plot(epochs, history['val']['f1'], label='val F1', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.title(f'F1 {model_name}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def bar_aucs(aucs_per_classes: np.ndarray, label_columns: list[str]) -> None:
    """Create a bar chart showing AUC scores for each disease class."""
    plt.figure(figsize=(30,4))
    
    # Create bar chart
    bars = plt.bar(label_columns, aucs_per_classes, alpha=0.7)
    
    # Customize the plot
    plt.xlabel('Disease Class')
    plt.ylabel('AUC Score')
    plt.title('AUC Scores by Disease Class')
    plt.ylim(0, 1.0)
    
    # Add value labels on top of bars
    for bar, auc in zip(bars, aucs_per_classes):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.3f}', ha='center', va='bottom')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()