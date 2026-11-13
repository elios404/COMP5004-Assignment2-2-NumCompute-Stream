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
) -> tuple[int, int | float | np.floating, float] | None:
    """Return (feature, threshold, gain) for a split of one node.

    X has shape (N, D) and contains finite numeric features. y has shape (N,).
    features optionally contains sorted indices of the columns to search.
    Values at or below the threshold go to the left child. Ties favour the
    lowest feature index, then the lowest threshold. Differences within eight
    floating-point epsilons are treated as ties. Return None when no split
    reduces impurity beyond that tolerance.
    """
    X = np.asarray(X)
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
            if X.dtype.kind in "iu" and max(abs(int(lower)), abs(int(upper))) >= 2**53:
                return int(feature_index), int(lower), float(gains[index])
            if X.dtype.kind in "biu":
                lower, upper = float(lower), float(upper)
            threshold = lower / 2 + upper / 2
            if not lower <= threshold < upper:
                threshold = lower
            if isinstance(threshold, np.generic):
                threshold = threshold.item()
            return int(feature_index), threshold, float(gains[index])

    return None


@dataclass
class _Node:
    class_index: int
    feature: int | None = None
    threshold: int | float | np.floating = 0.0
    left: _Node | None = None
    right: _Node | None = None


class DecisionTreeClassifier:
    """A classification tree with batch fitting and buffered retraining.

    The root has depth zero. max_depth limits the depth of the leaves, and
    min_samples_split is the minimum number of samples needed to try a split.
    max_features is the number of columns sampled without replacement per node.
    Use None to search every column. An integer random_state makes sampling
    reproducible across calls to fit.
    buffer_size is the minimum number of rows needed to retrain from the
    pending chunks. It defaults to 500.

    Leaves predict the most frequent class. Ties favour the first label in
    sorted order. Features must be finite real numbers.
    """

    def __init__(
        self, max_depth: int = 5, min_samples_split: int = 2, criterion: str = "gini",
        *, max_features: int | None = None, random_state: int | None = None,
        buffer_size: int = 500,
    ):

        if isinstance(max_depth, bool) or not isinstance(max_depth, Integral) or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer.")
        if (
            isinstance(min_samples_split, bool)
            or not isinstance(min_samples_split, Integral)
            or min_samples_split < 2
        ):
            raise ValueError("min_samples_split must be an integer of at least 2.")
        if not isinstance(criterion, str) or criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if max_features is not None and (
            isinstance(max_features, bool) or not isinstance(max_features, Integral) or max_features < 1
        ):
            raise ValueError("max_features must be None or a positive integer.")
        if random_state is not None and (
            isinstance(random_state, bool) or not isinstance(random_state, Integral) or random_state < 0
        ):
            raise ValueError("random_state must be None or a non-negative integer.")
        if isinstance(buffer_size, bool) or not isinstance(buffer_size, Integral) or buffer_size < 1:
            raise ValueError("buffer_size must be a positive integer.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        self.buffer_size = int(buffer_size)
        self.root_: _Node | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self._buffer: list[tuple[np.ndarray, np.ndarray]] = []
        self._buffered_rows = 0

    @staticmethod
    def _validate_features(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)

        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be a 2D array with at least one feature.")
        if X.dtype.kind not in "biuf":
            raise ValueError("X must contain real numeric feature values.")
        
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite feature values.")
        
        return X

    def _validate_training_data(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = self._validate_features(X)
        y = np.asarray(y)
        if self.max_features is not None and self.max_features > X.shape[1]:
            raise ValueError(f"max_features cannot exceed the number of input features ({X.shape[1]}).")

        if y.ndim != 1 or y.size == 0 or y.size != X.shape[0]:
            raise ValueError("y must be a non-empty 1D array with one label per sample.")
        if y.dtype.kind not in "biufUS" or (y.dtype.kind == "f" and not np.isfinite(y).all()):
            raise ValueError("y must contain finite numeric labels or strings.")
        
        return X, y

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeClassifier:
        """Train from scratch on X (N, D) and y (N,), then clear pending chunks."""
        X, y = self._validate_training_data(X, y)

        classes, encoded = np.unique(y, return_inverse=True)
        rng = np.random.default_rng(self.random_state)
        self.root_ = self._build(X, encoded, depth=0, rng=rng)
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self._buffer.clear()
        self._buffered_rows = 0

        return self

    def partial_fit(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> DecisionTreeClassifier:
        """Add a non-empty chunk of features (N, D) and labels (N,).

        Once at least buffer_size rows are available, replace the tree using
        all pending rows, including the whole new chunk. Otherwise, retain the
        current tree for prediction and keep a copy of the new chunk.

        The feature count must match the fitted tree or the first pending
        chunk. Pending chunks must use the same label family (numeric or
        string) and combine without losing integer precision.
        """
        X, y = self._validate_training_data(X_chunk, y_chunk)
        expected_features = self.n_features_in_

        if expected_features is None and self._buffer:
            expected_features = self._buffer[0][0].shape[1]
        if expected_features is not None and X.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {X.shape[1]}.")

        if self._buffer:
            buffered_labels = self._buffer[0][1]
            if (buffered_labels.dtype.kind in "US") != (y.dtype.kind in "US"):
                raise ValueError("Buffered chunks must use either numeric labels or string labels.")

        total_rows = self._buffered_rows + X.shape[0]
        if total_rows < self.buffer_size:
            self._buffer.append((X.copy(), y.copy()))
            self._buffered_rows = total_rows

            return self

        if not self._buffer:
            return self.fit(X, y)

        features, labels = zip(*self._buffer, (X, y))
        
        return self.fit(self._join_chunks(features), self._join_chunks(labels))

    @staticmethod
    def _join_chunks(chunks: tuple[np.ndarray, ...]) -> np.ndarray:
        joined = np.concatenate(chunks, axis=0)
        if joined.dtype.kind == "f":
            # Mixed numeric dtypes can round large integer values during concatenation.
            for chunk in chunks:
                if chunk.dtype.kind in "iu":
                    with np.errstate(invalid="ignore", over="ignore"):
                        restored = chunk.astype(joined.dtype).astype(chunk.dtype)
                    if not np.array_equal(restored, chunk):
                        raise ValueError("Chunk dtypes cannot be combined without losing integer precision.")
        return joined

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
                values = X[rows, node.feature]
                threshold = node.threshold
                if isinstance(threshold, int) and values.dtype.kind == "f":
                    # A rounded-up threshold would include values on the right.
                    values = values.astype(np.result_type(values.dtype, np.float64), copy=False)
                    rounded = values.dtype.type(threshold)
                    left = values < rounded if int(rounded) > threshold else values <= rounded
                elif values.dtype.kind in "iu":
                    cutoff = threshold if isinstance(threshold, int) else int(np.floor(threshold))
                    limits = np.iinfo(values.dtype)
                    if cutoff < limits.min or cutoff > limits.max:
                        left = np.full(values.shape, cutoff > limits.max)
                    else:
                        left = values <= cutoff
                else:
                    comparison_dtype = np.result_type(values.dtype, type(threshold))
                    left = values.astype(comparison_dtype, copy=False) <= threshold
                pending.append((node.left, rows[left]))
                pending.append((node.right, rows[~left]))

        return self.classes_[predictions]
