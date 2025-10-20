"""Metrics computation utilities for chest X-ray disease classification."""

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(y_true, y_pred):
    """Compute mean AUC and AUC for each class."""
    aucs = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        aucs.append(auc)
    aucs = np.array(aucs, dtype=np.float32)
    return np.nanmean(aucs), aucs
