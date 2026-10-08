"""
Pipeline abstractions and protocol interfaces for NumCompute.
"""

from __future__ import annotations
from typing import Any, List, Optional, Tuple
import numpy as np


class BaseTransformer:
    """Base class for all transformers."""

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> BaseTransformer:
        # TODO: Learn parameters from X (and optional y)
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        # TODO: Transform X using learned parameters
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class BaseEstimator:
    """Base class for all predictive models."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> BaseEstimator:
        # TODO: Fit model parameters from training data
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: Return predictions for input samples
        raise NotImplementedError


class Pipeline:
    """Chains multiple transformers with an optional final estimator."""

    def __init__(self, steps: List[Tuple[str, Any]]) -> None:
        self.steps = steps

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Pipeline:
        # TODO: Sequentially fit all transformers and final step
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        # TODO: Pass X through all transformers
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        # TODO: Fit and transform data through all pipeline steps
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: Transform X through preprocessors and call predict on final estimator
        raise NotImplementedError