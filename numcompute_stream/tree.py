"""Classification trees for numeric features."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _impurity_from_counts(counts: np.ndarray, criterion: str) -> np.ndarray:
    probabilities = counts / counts.sum(axis=-1, keepdims=True)
    if criterion == "gini":
        return 1.0 - np.sum(probabilities ** 2, axis=-1)
    if criterion == "entropy":
        log_probabilities = np.zeros_like(probabilities)
        np.log2(probabilities, out=log_probabilities, where=probabilities > 0)
        return -np.sum(probabilities * log_probabilities, axis=-1)
    raise ValueError("criterion must be 'gini' or 'entropy'.")


def _impurity(y: np.ndarray, criterion: str) -> float:
    """Compute Gini impurity or entropy from a non-empty vector of labels."""
    _, counts = np.unique(y, return_counts=True)
    return float(_impurity_from_counts(counts, criterion))


def _best_split(
    X: np.ndarray, y: np.ndarray, criterion: str = "gini", features: np.ndarray | None = None
) -> tuple[int, float, float] | None:
    """Return (feature, threshold, gain) for a split of one node.

    X has shape (N, D) and contains finite numeric features. y has shape (N,).
    features optionally contains sorted indices of the columns to search.
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

    _, encoded, counts = np.unique(y, return_inverse=True, return_counts=True)
    parent_impurity = _impurity_from_counts(counts, criterion)
    gain_tolerance = 8 * np.finfo(float).eps
    indicators = encoded[:, None] == np.arange(counts.size)
    candidates = []
    best_gain = gain_tolerance

    for feature_index in range(X.shape[1]) if features is None else features:
        column = X[:, feature_index]
        order = np.argsort(column, kind="stable")
        values = column[order]
        boundaries = np.flatnonzero(values[:-1] < values[1:])
        if boundaries.size == 0:
            continue

        left_counts = np.cumsum(indicators[order], axis=0)[boundaries]
        right_counts = counts - left_counts
        n_left = boundaries + 1
        gains = parent_impurity - (
            n_left * _impurity_from_counts(left_counts, criterion)
            + (y.size - n_left) * _impurity_from_counts(right_counts, criterion)
        ) / y.size
        best_gain = max(best_gain, gains.max())
        tied = (gains > gain_tolerance) & (gains >= best_gain - gain_tolerance)
        if tied.any():
            candidates.append((feature_index, values[boundaries[tied]],
                               values[boundaries[tied] + 1], gains[tied]))

    for feature_index, lower_values, upper_values, gains in candidates:
        tied = np.flatnonzero(gains >= best_gain - gain_tolerance)
        if tied.size:
            index = tied[0]
            lower, upper = lower_values[index], upper_values[index]
            threshold = lower / 2 + upper / 2
            if not lower <= threshold < upper:
                threshold = lower
            return int(feature_index), float(threshold), float(gains[index])

    return None


@dataclass
class _Node:
    class_index: int
    feature: int | None = None
    threshold: float = 0.0
    left: _Node | None = None
    right: _Node | None = None


class DecisionTreeClassifier:
    """A batch classification tree using Gini impurity or entropy.

    The root has depth zero. max_depth limits the depth of the leaves, and
    min_samples_split is the minimum number of samples needed to try a split.
    max_features is the number of columns sampled without replacement per node.
    Use None to search every column. An integer random_state makes sampling
    reproducible across calls to fit.
    """

    def __init__(
        self, max_depth: int = 5, min_samples_split: int = 2, criterion: str = "gini",
        *, max_features: int | None = None, random_state: int | None = None,
    ):

        if isinstance(max_depth, bool) or not isinstance(max_depth, Integral) or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer.")
        if (
            isinstance(min_samples_split, bool)
            or not isinstance(min_samples_split, Integral)
            or min_samples_split < 2
        ):
            raise ValueError("min_samples_split must be an integer of at least 2.")
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if max_features is not None and (
            isinstance(max_features, bool) or not isinstance(max_features, Integral) or max_features < 1
        ):
            raise ValueError("max_features must be None or a positive integer.")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, Integral) or random_state < 0
        ):
            raise ValueError("random_state must be None or a non-negative integer.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self.root_: _Node | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    @staticmethod
    def _validate_features(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)

        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be a 2D array with at least one feature.")
        if X.dtype.kind not in "biuf":
            raise ValueError("X must contain real numeric feature values.")
        
        X = X.astype(float, copy=False)
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite feature values.")
        
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
        """Train from scratch on features of shape (N, D) and labels of shape (N,)."""
        X = self._validate_features(X)
        y = np.asarray(y)
        if self.max_features is not None and self.max_features > X.shape[1]:
            raise ValueError(f"max_features cannot exceed the number of input features ({X.shape[1]}).")

        if y.ndim != 1 or y.size == 0 or y.size != X.shape[0]:
            raise ValueError("y must be a non-empty 1D array with one label per sample.")
        if y.dtype.kind not in "biufUS" or (y.dtype.kind == "f" and not np.isfinite(y).all()):
            raise ValueError("y must contain finite numeric labels or strings.")

        classes, encoded = np.unique(y, return_inverse=True)
        rng = np.random.default_rng(self.random_state)
        self.root_ = self._build(X, encoded, depth=0, rng=rng)
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]

        return self

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator) -> _Node:
        classes, counts = np.unique(y, return_counts=True)
        node = _Node(class_index=int(classes[np.argmax(counts)]))

        if classes.size == 1 or depth >= self.max_depth or y.size < self.min_samples_split:
            return node

        features = None
        if self.max_features is not None and self.max_features < X.shape[1]:
            features = np.sort(rng.choice(X.shape[1], size=self.max_features, replace=False))
        split = _best_split(X, y, self.criterion, features)
        if split is None:
            return node

        node.feature, node.threshold, _ = split
        left = X[:, node.feature] <= node.threshold
        node.left = self._build(X[left], y[left], depth + 1, rng)
        node.right = self._build(X[~left], y[~left], depth + 1, rng)

        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return one class label per row of a feature array of shape (N, D)."""
        if self.root_ is None or self.classes_ is None:
            raise RuntimeError("DecisionTreeClassifier is not fitted yet.")
        
        X = self._validate_features(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}.")

        predictions = np.empty(X.shape[0], dtype=np.intp)
        pending = [(self.root_, np.arange(X.shape[0]))]
        
        while pending:
            node, rows = pending.pop()
            if rows.size == 0:
                continue
            if node.feature is None:
                predictions[rows] = node.class_index
            else:
                left = X[rows, node.feature] <= node.threshold
                pending.append((node.left, rows[left]))
                pending.append((node.right, rows[~left]))

        return self.classes_[predictions]
