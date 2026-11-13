"""Impurity and split selection for streaming classification trees."""

from __future__ import annotations

import numpy as np


def _impurity(y: np.ndarray, criterion: str) -> float:
    """Compute Gini impurity or entropy from a non-empty vector of labels."""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / y.size

    if criterion == "gini":
        return float(1.0 - np.sum(probabilities ** 2))
    if criterion == "entropy":
        return float(-np.sum(probabilities * np.log2(probabilities)))
    
    raise ValueError("criterion must be 'gini' or 'entropy'.")


def _best_split(X: np.ndarray, y: np.ndarray, criterion: str = "gini") -> tuple[int, float, float] | None:
    """Return (feature, threshold, gain) for a split of one node.

    X has shape (N, D) and contains finite numeric features. y has shape (N,).
    Values at or below the threshold go to the left child. Ties favour the
    lowest feature index, then the lowest threshold. Differences within eight
    floating-point epsilons are treated as ties. Return None when no split
    reduces impurity beyond that tolerance.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)

    if X.ndim != 2 or y.ndim != 1:
        raise ValueError("Expected a 2D feature array and a 1D label array.")
    if X.shape[0] != y.size or y.size == 0:
        raise ValueError("X and y must contain the same non-zero number of samples.")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite feature values.")

    parent_impurity = _impurity(y, criterion)
    gain_tolerance = 8 * np.finfo(float).eps
    best_split = None
    best_gain = 0.0

    for feature_index in range(X.shape[1]):
        column = X[:, feature_index]
        values = np.unique(column)

        for lower, upper in zip(values[:-1], values[1:]):
            threshold = float(lower / 2 + upper / 2)
            
            if not lower <= threshold < upper:
                threshold = float(lower)

            left = column <= threshold
            n_left = np.count_nonzero(left)
            weighted_impurity = (
                n_left * _impurity(y[left], criterion)
                + (y.size - n_left) * _impurity(y[~left], criterion)
            ) / y.size
            gain = parent_impurity - weighted_impurity

            if gain > best_gain + gain_tolerance:
                best_gain = gain
                best_split = (feature_index, threshold, gain)

    return best_split
