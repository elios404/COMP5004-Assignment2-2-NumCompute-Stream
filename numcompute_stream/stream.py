"""Evaluate and train a classification pipeline one chunk at a time."""

from __future__ import annotations

from time import perf_counter
import tracemalloc

import numpy as np

from .metrics import StreamingMetrics
from .pipeline import Pipeline


class StreamTrainer:
    """Run test-then-train evaluation and keep a log of successful chunks.

    Provide the full class list in labels. For binary evaluation the second
    label is positive, matching StreamingMetrics. Warm-up chunks train the
    pipeline but do not enter the evaluation counts.
    """

    def __init__(self, pipeline: Pipeline, labels=(0, 1), *, window_size: int | None = None):
        if not isinstance(pipeline, Pipeline):
            raise TypeError("pipeline must be a streaming Pipeline.")
        if not callable(getattr(pipeline.steps[-1][1], "predict", None)):
            raise TypeError("The final pipeline step must implement predict.")

        self.pipeline = pipeline
        self.metrics = StreamingMetrics(labels, window_size=window_size)
        self._labels = np.asarray(labels).copy()
        self.logs: list[dict] = []
        self.last_evaluation: dict | None = None

    def score_chunk(self, X, y) -> dict:
        """Evaluate the current model without training or updating the log.

        X is shaped (N, D) and y is shaped (N,). The result includes the
        predictions, the actual labels and optional binary scores.
        """
        actual = np.asarray(y)
        predictions = self.pipeline.predict(X)
        model = self.pipeline.steps[-1][1]
        scores = None

        if self._labels.size == 2 and callable(getattr(model, "predict_proba", None)):
            probabilities = np.asarray(self.pipeline.predict_proba(X))
            model_labels = np.asarray(getattr(model, "classes_", None), dtype=object)

            if model_labels.ndim != 1 or probabilities.shape != (actual.size, model_labels.size):
                raise ValueError("predict_proba must return one column per fitted class in classes_ order.")
            
            positive = np.flatnonzero(model_labels == self._labels.astype(object)[1])
            scores = probabilities[:, positive[0]] if positive.size else np.zeros(actual.size)

        chunk_metrics = StreamingMetrics(self._labels).update(actual, predictions, scores)
        return {
            **chunk_metrics.result(),
            "y_true": actual.copy(),
            "y_pred": np.asarray(predictions).copy(),
            "y_score": scores,
        }

    def fit_chunk(self, X, y) -> dict:
        """Score with the previous model, then train and return a log entry.

        peak_traced_bytes is the peak of allocations tracked by tracemalloc
        during training when this call starts the tracer. If the caller already
        has tracing enabled, it is the peak of that existing tracing session.
        fit_seconds measures partial_fit, including preprocessing and buffering.
        """
        actual = np.asarray(y)
        evaluation = None
        if self.pipeline.is_fitted:
            evaluation = self.score_chunk(X, actual)
        else:
            self.metrics._encode(actual, "y")
        previous_model = self.pipeline.steps[-1][1]

        owns_tracer = not tracemalloc.is_tracing()
        if owns_tracer:
            tracemalloc.start()

        try:
            start = perf_counter()
            self.pipeline.partial_fit(X, actual)
            elapsed = perf_counter() - start
            _, peak = tracemalloc.get_traced_memory()
        finally:
            if owns_tracer:
                tracemalloc.stop()

        if evaluation is not None:
            self.metrics.update(evaluation["y_true"], evaluation["y_pred"], evaluation["y_score"])
            self.last_evaluation = evaluation

        entry = {
            "chunk": len(self.logs) + 1,
            "n_samples": actual.size,
            "evaluated": evaluation is not None,
            "accuracy": None if evaluation is None else evaluation["accuracy"],
            "precision": None if evaluation is None else evaluation["precision"],
            "recall": None if evaluation is None else evaluation["recall"],
            "f1": None if evaluation is None else evaluation["f1"],
            "auc": None if evaluation is None else evaluation["auc"],
            "cumulative_accuracy": self.metrics.accuracy,
            "retrained": self.pipeline.steps[-1][1] is not previous_model,
            "fit_seconds": elapsed,
            "peak_traced_bytes": peak,
        }
        self.logs.append(entry)
        return entry.copy()
