"""Cumulative and rolling classification metrics."""

from __future__ import annotations
from numbers import Integral
import numpy as np
from numcompute.metrics import roc_curve


def _label_array(values, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array.")
    if values.dtype.kind not in "biufUS" or (
        values.dtype.kind == "f" and not np.isfinite(values).all()
    ):
        raise ValueError(f"{name} must contain finite numeric or string labels.")

    return values


class StreamingMetrics:
    """Accumulate predictions for a fixed set of class labels.

    Labels are kept in the supplied order. Confusion matrix rows are true
    classes and columns are predicted classes. For binary classification,
    the second label is positive. For more classes, precision, recall and
    F1 are unweighted macro averages. Zero denominators yield zero.

    If window_size is set, rolling_result() uses the last window_size rows.
    Binary ROC AUC uses the rows supplied with scores and is NaN until
    scores are available for both classes.
    """

    def __init__(self, labels=(0, 1), *, window_size: int | None = None):
        labels = _label_array(labels, "labels")
        if labels.size < 2 or np.unique(labels).size != labels.size:
            raise ValueError("labels must contain at least two distinct classes.")
        if window_size is not None and (
            isinstance(window_size, bool) or not isinstance(window_size, Integral) or window_size < 1
        ):
            raise ValueError("window_size must be a positive integer.")

        self._labels = labels.copy()
        self._label_order = np.argsort(labels)
        self._sorted_labels = labels[self._label_order].astype(object)
        self._window_size = None if window_size is None else int(window_size)
        self.reset()

    def reset(self) -> None:
        """Clear cumulative and rolling data while keeping the configuration."""
        size = self._labels.size
        self._counts = np.zeros((size, size), dtype=np.int64)
        self._scored_chunks: list[tuple[np.ndarray, np.ndarray]] = []
        self._window_pairs = np.empty(0, dtype=np.intp)
        self._window_scores = np.empty(0, dtype=float)

    def _encode(self, values, name: str) -> np.ndarray:
        values = _label_array(values, name)
        kind = self._labels.dtype.kind
        if values.dtype.kind != kind and (
            values.dtype.kind not in "biuf" or kind not in "biuf"
        ):
            raise ValueError(f"{name} must use the configured label type.")

        # Object comparisons preserve distinct large integers across numeric dtypes.
        values = values.astype(object)
        indices = np.searchsorted(self._sorted_labels, values)
        indices = np.minimum(indices, self._labels.size - 1)
        if np.any(self._sorted_labels[indices] != values):
            raise ValueError(f"{name} contains an unknown class label.")

        return self._label_order[indices]

    def _confusion(self, pairs: np.ndarray) -> np.ndarray:
        size = self._labels.size
        return np.bincount(pairs, minlength=size * size).reshape(size, size)

    def update(self, y_true_chunk, y_pred_chunk, y_score_chunk=None) -> StreamingMetrics:
        """Add matching (N,) label arrays and optional finite binary scores.

        Higher scores must indicate the positive class. Updates without
        scores still contribute to the confusion matrix and rolling window.
        """
        actual = self._encode(y_true_chunk, "y_true_chunk")
        predicted = self._encode(y_pred_chunk, "y_pred_chunk")
        if actual.shape != predicted.shape:
            raise ValueError("True and predicted labels must have the same shape.")

        scores = None
        if y_score_chunk is not None:
            if self._labels.size != 2:
                raise ValueError("ROC AUC scores require binary classification.")
            scores = np.asarray(y_score_chunk)
            if scores.shape != actual.shape or scores.dtype.kind not in "biuf":
                raise ValueError("Scores must be a real 1D array matching the labels.")
            with np.errstate(over="ignore"):
                scores = scores.astype(float)
            if not np.isfinite(scores).all():
                raise ValueError("Scores must be finite.")

        pairs = actual * self._labels.size + predicted
        self._counts += self._confusion(pairs)

        if scores is not None:
            self._scored_chunks.append((actual, scores))

        if self._window_size is not None:
            window = self._window_size
            recent_scores = np.full(min(actual.size, window), np.nan) if scores is None else scores[-window:]
            self._window_pairs = np.concatenate((self._window_pairs, pairs[-window:]))[-window:].copy()
            self._window_scores = np.concatenate((self._window_scores, recent_scores))[-window:].copy()

        return self

    @staticmethod
    def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
        if labels.size == 0 or labels.min() == labels.max():
            return float("nan")

        fpr, tpr, _ = roc_curve(labels, scores)
        return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))

    def _result(self, counts: np.ndarray, auc: float) -> dict:
        true_positive = counts.diagonal()
        predicted = counts.sum(axis=0)
        actual = counts.sum(axis=1)
        total = int(actual.sum())

        precision = true_positive / np.maximum(predicted, 1)
        recall = true_positive / np.maximum(actual, 1)
        f1 = 2.0 * true_positive / np.maximum(predicted + actual, 1)
        selected = slice(1, 2) if self._labels.size == 2 else slice(None)

        return {
            "count": total,
            "labels": self._labels.copy(),
            "accuracy": float(true_positive.sum() / total) if total else float("nan"),
            "precision": float(precision[selected].mean()),
            "recall": float(recall[selected].mean()),
            "f1": float(f1[selected].mean()),
            "confusion_matrix": counts.copy(),
            "auc": auc,
        }

    def result(self) -> dict:
        """Return cumulative metrics, including a copy of the confusion matrix."""
        auc = float("nan")
        if self._scored_chunks:
            labels, scores = zip(*self._scored_chunks)
            auc = self._roc_auc(np.concatenate(labels), np.concatenate(scores))

        return self._result(self._counts, auc)

    @property
    def accuracy(self) -> float:
        """Return cumulative accuracy without calculating ROC AUC."""
        total = self._counts.sum()
        return float(self._counts.trace() / total) if total else float("nan")

    def rolling_result(self) -> dict:
        """Return metrics over the most recent window_size observations."""
        if self._window_size is None:
            raise RuntimeError("Rolling metrics require window_size.")

        scored = ~np.isnan(self._window_scores)
        labels = self._window_pairs[scored] // self._labels.size
        auc = self._roc_auc(labels, self._window_scores[scored])
        return self._result(self._confusion(self._window_pairs), auc)
