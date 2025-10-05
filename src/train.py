"""Training pipeline for chest X-ray disease classification."""

import numpy as np
from tqdm import tqdm
import torch
from utils.metrics import MetricCalculator
from utils.plot import plot_learning_curves, bar_aucs
from utils.save_metrics import save_results_csv


class Trainer:
    """Training pipeline for chest X-ray disease classification model."""
    
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                 criterion: torch.nn.Module, train_loader: torch.utils.data.DataLoader, 
                 val_loader: torch.utils.data.DataLoader, device: torch.device, 
                 config: dict, metric_calculator: MetricCalculator | None = None) -> None:
        """Initialize the Trainer."""
        # Move model to specified device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Extract training parameters from config
        self.num_epochs = config.get("num_epochs", 10)
        self.batch_size = config.get("batch_size", 8)
        self.lr = config.get("lr", 1e-4)
        self.model_name = config.get("model_name")
        self.label_columns = config.get("label_columns")

        # Initialize metric calculator for evaluation
        self.metric_calculator = metric_calculator or MetricCalculator(self.label_columns)

        # Initialize training history tracking
        self.history = {
            'train': {'loss': [], 'auc': [], 'f1': [], 'precision': [], 'recall': []},
            'val': {'loss': [], 'auc': [], 'f1': [], 'precision': [], 'recall': []}
        }

    def train_one_batch(self, imgs: torch.Tensor, labels: torch.Tensor) -> tuple[float, np.ndarray, np.ndarray]:
        """Train the model on a single batch of data."""
        # Move data to device and ensure correct data types
        imgs, labels = imgs.to(self.device, dtype=torch.float), labels.to(self.device, dtype=torch.float)
        
        # Zero out gradients from previous iteration
        self.optimizer.zero_grad()
        
        # Forward pass through the model
        logits = self.model(imgs)
        
        # Compute loss
        loss = self.criterion(logits, labels)
        
        # Backward pass and parameter update
        loss.backward()
        self.optimizer.step()
        
        # Convert predictions to probabilities and move to CPU for metric computation
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        
        return loss.item(), probs, labels.cpu().numpy()

    def train_one_epoch(self) -> tuple[float, dict]:
        """Train the model for one complete epoch."""
        # Set model to training mode
        self.model.train()
        running_loss = 0
        all_preds, all_labels = [], []

        # Iterate through training batches with progress bar
        pbar = tqdm(self.train_loader, desc="Train")
        for imgs, labels in pbar:
            # Train on current batch
            loss, probs, labels_np = self.train_one_batch(imgs, labels)
            
            # Accumulate loss (weighted by batch size)
            running_loss += loss * imgs.size(0)
            
            # Store predictions and labels for metric computation
            all_preds.append(probs)
            all_labels.append(labels_np)
            
            # Update progress bar with current loss
            pbar.set_postfix({'Loss': f"{running_loss / ((pbar.n+1)*self.batch_size):.4f}"})

        # Concatenate all predictions and labels
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Compute average loss for the epoch
        epoch_loss = running_loss / len(self.train_loader.dataset)
        
        # Compute all metrics for the epoch
        metrics = self.metric_calculator.compute_all_metrics(all_labels, all_preds)
        
        return epoch_loss, metrics

    def validate_one_epoch(self) -> tuple[float, dict]:
        """Validate the model for one complete epoch."""
        # Set model to evaluation mode
        self.model.eval()
        val_logits_all, val_probs_all, val_labels_all = [], [], []

        # Run validation without gradient computation
        with torch.no_grad():
            for imgs, labels in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                imgs, labels = imgs.to(self.device, dtype=torch.float), labels.to(self.device, dtype=torch.float)
                
                # Forward pass
                logits = self.model(imgs)
                probs = torch.sigmoid(logits)

                # Store logits, probabilities, and labels
                val_logits_all.append(logits.detach())
                val_probs_all.append(probs.detach().cpu().numpy())
                val_labels_all.append(labels.detach())

        # Concatenate all validation results
        val_logits = torch.cat(val_logits_all).to(self.device)
        val_labels = torch.cat(val_labels_all).to(self.device)
        val_preds = np.vstack(val_probs_all)

        # Compute validation loss
        val_loss = self.criterion(val_logits, val_labels).item()
        
        # Find optimal thresholds for each class based on F1 score
        self.metric_calculator.find_best_thresholds(val_labels.cpu().numpy(), val_preds)
        
        # Compute validation metrics
        val_metrics = self.metric_calculator.compute_all_metrics(val_labels.cpu().numpy(), val_preds)
        
        return val_loss, val_metrics

    def fit(self) -> None:
        """Run the complete training process."""
        for epoch in range(self.num_epochs):
            # Train for one epoch
            train_loss, train_metrics = self.train_one_epoch()
            
            # Validate for one epoch
            val_loss, val_metrics = self.validate_one_epoch()

            # Update training history with metrics from both phases
            for phase, loss, metrics in zip(['train', 'val'], [train_loss, val_loss], [train_metrics, val_metrics]):
                self.history[phase]['loss'].append(loss)
                self.history[phase]['auc'].append(metrics['auc'][0])
                self.history[phase]['f1'].append(metrics['f1'][0])
                self.history[phase]['precision'].append(metrics['precision'][0])
                self.history[phase]['recall'].append(metrics['recall'][0])

            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.num_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Print detailed metrics for both training and validation
            self.metric_calculator.print_detailed_metrics(train_metrics, phase="Train")
            self.metric_calculator.print_detailed_metrics(val_metrics, phase="Validation")

        # Find and report best validation AUC
        best_val_auc = max(self.history['val']['auc'])
        print(f"The best validation AUC: {best_val_auc}")

        # Generate visualizations and save results
        plot_learning_curves(self.model_name, self.history, self.num_epochs)
        bar_aucs(val_metrics['auc'][1], self.label_columns)
        save_results_csv(self.model_name, self.history, self.num_epochs, self.lr)
