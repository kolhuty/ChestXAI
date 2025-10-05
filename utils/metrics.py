"""Metrics computation utilities for chest X-ray disease classification."""

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve


class MetricCalculator:
    """Calculator for evaluation metrics in multi-label classification."""
    
    def __init__(self, class_names: list[str], threshold: float | np.ndarray = 0.5) -> None:
        """Initialize the MetricCalculator."""
        self.class_names = class_names
        self.threshold = threshold

    def _apply_thresholds(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply thresholds to convert probabilities to binary predictions."""
        if isinstance(self.threshold, (float, int)):
            # Single threshold for all classes
            return (y_pred >= self.threshold).astype(int)
        else:
            # Different threshold for each class
            return (y_pred >= self.threshold[None, :]).astype(int)

    def compute_auc(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute Area Under the ROC Curve (AUC) for each class."""
        aucs = []
        for i in range(y_true.shape[1]):
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            except ValueError:
                raise ValueError("Empty array for metric")
            aucs.append(auc)
        return np.nanmean(aucs), np.array(aucs, dtype=np.float32)

    def compute_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute F1-score for each class."""
        y_pred_binary = self._apply_thresholds(y_pred)
        f1_scores = []
        for i in range(y_true.shape[1]):
            try:
                f1 = f1_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
            except ValueError:
                raise ValueError("Empty array for metric")
            f1_scores.append(f1)
        return np.nanmean(f1_scores), np.array(f1_scores, dtype=np.float32)

    def compute_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute precision for each class."""
        y_pred_binary = self._apply_thresholds(y_pred)
        precision_scores = []
        for i in range(y_true.shape[1]):
            try:
                precision = precision_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
            except ValueError:
                raise ValueError("Empty array for metric")
            precision_scores.append(precision)
        return np.nanmean(precision_scores), np.array(precision_scores, dtype=np.float32)

    def compute_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
        """Compute recall for each class."""
        y_pred_binary = self._apply_thresholds(y_pred)
        recall_scores = []
        for i in range(y_true.shape[1]):
            try:
                recall = recall_score(y_true[:, i], y_pred_binary[:, i], zero_division=0)
            except ValueError:
                raise ValueError("Empty array for metric")
            recall_scores.append(recall)
        return np.nanmean(recall_scores), np.array(recall_scores, dtype=np.float32)

    def compute_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute all evaluation metrics for the given predictions."""
        auc_mean, auc_per_class = self.compute_auc(y_true, y_pred)
        f1_mean, f1_per_class = self.compute_f1(y_true, y_pred)
        precision_mean, precision_per_class = self.compute_precision(y_true, y_pred)
        recall_mean, recall_per_class = self.compute_recall(y_true, y_pred)

        return {
            'auc': (auc_mean, auc_per_class),
            'f1': (f1_mean, f1_per_class),
            'precision': (precision_mean, precision_per_class),
            'recall': (recall_mean, recall_per_class)
        }

    def find_best_thresholds(self, y_true: np.ndarray, y_probs: np.ndarray) -> None:
        """Find optimal thresholds for each class based on F1-score maximization."""
        C = y_true.shape[1]
        best_thresholds = np.zeros(C)
        best_f1s = np.zeros(C)
        
        for c in range(C):
            y_c = y_true[:, c]
            p_c = y_probs[:, c]

            # Handle classes with no positive examples
            if y_c.sum() == 0:
                best_thresholds[c] = 0.5
                best_f1s[c] = 0.0
                continue

            # Compute precision-recall curve
            prec, rec, thr = precision_recall_curve(y_c, p_c)
            
            # Calculate F1-score for each threshold
            f1 = (2 * prec * rec) / (prec + rec + 1e-12)
            f1 = f1[:-1]  # Remove last element to align with thresholds
            
            # Find threshold that maximizes F1-score
            idx = np.argmax(f1)
            best_thresholds[c] = thr[idx]
            best_f1s[c] = f1[idx]

        # Update the threshold for future metric computations
        self.threshold = best_thresholds

    def print_detailed_metrics(self, metrics: dict, phase: str = "Train") -> None:
        """Print detailed metrics in a formatted manner."""
        print(f"\n{phase} metrics")
        print(f"AUC: {metrics['auc'][0]:.3f}")
        print(f"F1: {metrics['f1'][0]:.3f}")
        print(f"Precision: {metrics['precision'][0]:.3f}")
        print(f"Recall: {metrics['recall'][0]:.3f}")
        print(f"Optimal thresholds per class: {self.threshold}")
        self._print_problematic_classes(metrics)

    def _print_problematic_classes(self, metrics: dict, threshold: float = 0.7) -> None:
        """Print classes with poor performance for debugging purposes."""
        auc_per_class = metrics['auc'][1]
        f1_per_class = metrics['f1'][1]
        problematic_classes = []
        
        for i, (auc, f1) in enumerate(zip(auc_per_class, f1_per_class)):
            if auc < threshold or f1 < threshold:
                class_name = self.class_names[i]
                problematic_classes.append((class_name, auc, f1))

        if problematic_classes:
            print("\nProblematic classes:")
            for class_name, auc, f1 in problematic_classes:
                print(f"  {class_name}: AUC={auc:.3f}, F1={f1:.3f}")
