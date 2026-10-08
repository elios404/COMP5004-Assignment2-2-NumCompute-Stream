"""
Preprocessing transformers for NumCompute.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from numcompute.pipeline import BaseTransformer


class StandardScaler(BaseTransformer):
    """Standardize features by removing the mean and scaling to unit variance."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> StandardScaler:
        # TODO: Compute statistics on X (shape: (n_samples, n_features))
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        # TODO: Scale X using computed statistics and return transformed array
        raise NotImplementedError


class MinMaxScaler(BaseTransformer):
    """Scale features to a given range (default 0 to 1)."""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.feature_range = feature_range

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> MinMaxScaler:
        # TODO: Compute min and max of X
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        # TODO: Scale X to feature_range
        raise NotImplementedError


class OneHotEncoder(BaseTransformer):
    """One-hot encode categorical features."""

    def __init__(self, drop_first: bool = False):
        self.drop_first = drop_first

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> OneHotEncoder:
        # TODO: Identify unique categories per column
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        # TODO: Transform X into one-hot numeric representation
        raise NotImplementedError