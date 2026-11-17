"""Batch and streaming pipelines for NumCompute."""

from __future__ import annotations
from copy import copy, deepcopy
from numbers import Integral
from numcompute.pipeline import Pipeline as BatchPipeline
from .preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from .tree import DecisionTreeClassifier
import numpy as np


class Pipeline(BatchPipeline):
    """Chain transformers with an optional final estimator.

    Steps must support deepcopy. After training, named_steps exposes the
    pipeline's copies. Buffered estimators expose a positive buffer_size and
    are fitted on a full window transformed with one preprocessing state.
    Other incremental steps are updated immediately via partial_fit.
    A scaler directly after OneHotEncoder treats newly added columns as
    zero for earlier rows.
    """

    def __init__(self, steps):
        steps = list(steps)
        super().__init__(steps)

        names = [name for name, _ in steps]
        if len(set(names)) != len(names):
            raise ValueError("Pipeline step names must be unique.")
        for name, step in steps[:-1]:
            if not all(callable(getattr(step, method, None)) for method in ("fit", "transform")):
                raise TypeError(f"Intermediate step '{name}' must implement fit and transform.")
        
        name, final_step = steps[-1]
        if not callable(getattr(final_step, "fit", None)) or not any(
            callable(getattr(final_step, method, None)) for method in ("predict", "transform")
        ):
            raise TypeError(f"Final step '{name}' must implement fit and either predict or transform.")

        self._training_steps = self.steps
        self._pending: list[tuple[np.ndarray, np.ndarray]] = [] # buffer
        self._pending_rows = 0
        self.n_features_in_: int | None = None

    def _input(self, X, *, training=False, reset=False):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be a 2D array with at least one feature.")
        if training and X.shape[0] == 0:
            raise ValueError("Cannot train on an empty chunk.")
        if not reset and self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} input features, got {X.shape[1]}.")

        return X

    def _commit_fit(self, steps, n_features):
        self.steps = steps
        self._training_steps = steps
        self.n_features_in_ = n_features
        self._is_fitted = True
        self._pending.clear()
        self._pending_rows = 0

    def fit(self, X, y=None):
        """Train from scratch and clear pending chunks after a successful fit."""
        X = self._input(X, training=True, reset=True)

        # to keep the original pipeline in case of failure
        candidate = copy(self)
        candidate.steps = deepcopy(self.steps)
        super(Pipeline, candidate).fit(X, y)

        self._commit_fit(candidate.steps, X.shape[1])
        return self

    def fit_transform(self, X, y=None):
        """Fit a transformer-only pipeline and return its transformed data."""
        X = self._input(X, training=True, reset=True)

        candidate = copy(self)
        candidate.steps = deepcopy(self.steps)
        result = super(Pipeline, candidate).fit_transform(X, y)

        self._commit_fit(candidate.steps, X.shape[1])
        return result

    def predict(self, X):
        """Predict using the preprocessing state paired with the fitted model."""
        return super().predict(self._input(X))

    @property
    def is_fitted(self) -> bool:
        """Whether a complete set of fitted steps is available."""
        return self._is_fitted

    def predict_proba(self, X):
        """Apply the fitted preprocessing and return the final model's probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline is not fitted yet.")
        
        name, model = self.steps[-1]
        if not callable(getattr(model, "predict_proba", None)):
            raise AttributeError(f"Final step '{name}' does not implement predict_proba.")

        X = self._input(X)
        for _, step in self.steps[:-1]:
            X = step.transform(X)
        return model.predict_proba(X)

    def transform(self, X):
        """Apply the fitted steps of a transformer-only pipeline."""
        return super().transform(self._input(X))

    def partial_fit(self, X_chunk, y_chunk=None):
        """Update a pipeline from a non-empty feature chunk of shape (N, D).

        Buffered classification requires one label per row. Preprocessors are
        updated in order on the arriving chunk. At retraining, the raw window
        is transformed with the resulting state and then passed to model.fit.
        """
        X = self._input(X_chunk, training=True)
        n_features = X.shape[1]
        name, model = self._training_steps[-1]
        buffered = callable(getattr(model, "predict", None)) and hasattr(model, "buffer_size")
        y = None if y_chunk is None else np.asarray(y_chunk)
        if y is not None and (y.ndim == 0 or y.shape[0] != X.shape[0]):
            raise ValueError("y must contain one entry per sample.")

        if buffered:
            buffer_size = model.buffer_size
            if isinstance(buffer_size, bool) or not isinstance(buffer_size, Integral) or buffer_size < 1:
                raise ValueError("buffer_size must be a positive integer.")
            if y is None or y.ndim != 1:
                raise ValueError("Buffered classification requires one label per sample.")
            if y.dtype.kind not in "biufUS" or (y.dtype.kind == "f" and not np.isfinite(y).all()):
                raise ValueError("y must contain finite numeric labels or strings.")
            if self._pending:
                first_X, first_y = self._pending[0]
                for previous, current in ((first_X, X), (first_y, y)):
                    if previous.dtype.kind != current.dtype.kind and (
                        previous.dtype.kind not in "biuf" or current.dtype.kind not in "biuf"
                    ):
                        raise ValueError("Pending chunks must use compatible feature and label types.")

        update_steps = self._training_steps[:-1] if buffered else self._training_steps
        for step_name, step in update_steps:
            if not callable(getattr(step, "partial_fit", None)):
                raise TypeError(f"Step '{step_name}' must implement partial_fit.")

        candidate_steps = deepcopy(update_steps)
        transformed = X.copy()
        previous_step = None
        for index, (_, step) in enumerate(candidate_steps):
            if isinstance(previous_step, OneHotEncoder) and isinstance(step, (StandardScaler, MinMaxScaler)):
                width = transformed.shape[1]
                if step.n_features_in_ is None or width > step.n_features_in_:
                    previous_rows = previous_step._n_rows_seen - transformed.shape[0]
                    step._extend_one_hot(width, previous_rows)
            step.partial_fit(transformed, y)
            if buffered or index < len(candidate_steps) - 1:
                transformed = step.transform(transformed)
            previous_step = step

        if not buffered:
            self._commit_fit(candidate_steps, X.shape[1])
            return self

        total_rows = self._pending_rows + X.shape[0]
        if isinstance(model, DecisionTreeClassifier) and (
            np.asarray(transformed).dtype.kind not in "biuf" or not np.isfinite(transformed).all()
        ):
            raise ValueError("Tree input must contain only finite feature values after preprocessing.")
        if total_rows < buffer_size:
            self._pending.append((X.copy(), y.copy()))
            self._pending_rows = total_rows
            self._training_steps = [*candidate_steps, (name, model)]
            self.n_features_in_ = X.shape[1]
            return self

        if self._pending:
            features, labels = zip(*self._pending, (X, y))
            X = DecisionTreeClassifier._join_chunks(features)
            y = DecisionTreeClassifier._join_chunks(labels)

        for _, step in candidate_steps:
            X = step.transform(X)

        candidate_model = deepcopy(model)
        candidate_model.fit(X, y)
        self._commit_fit([*candidate_steps, (name, candidate_model)], n_features)
        return self
