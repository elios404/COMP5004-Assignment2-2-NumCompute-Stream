"""
Classification and regression evaluation metrics.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # TODO: Compute fraction of correctly predicted samples
    raise NotImplementedError


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # TODO: Compute precision (TP / (TP + FP)) with zero-division safeguard
    raise NotImplementedError


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # TODO: Compute recall (TP / (TP + FN)) with zero-division safeguard
    raise NotImplementedError


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # TODO: Compute harmonic mean of precision and recall
    raise NotImplementedError


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return confusion matrix where row is true label, column is pred label."""
    # TODO: Compute confusion matrix
    raise NotImplementedError


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # TODO: Compute Mean Squared Error
    raise NotImplementedError


def roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve for binary classification.
    Returns: (fpr, tpr, thresholds)
    """
    # TODO: Sort scores, compute true/false positive rates across thresholds
    raise NotImplementedError


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Area Under the Curve using trapezoidal rule."""
    # TODO: Approximate integral via trapezoidal method
    raise NotImplementedError