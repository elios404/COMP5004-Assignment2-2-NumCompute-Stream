"""
Descriptive statistics, Welford streaming stats, histogram, and quantiles.
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union
import numpy as np


def mean(arr: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    # TODO: Compute mean along specified axis
    raise NotImplementedError


def std(arr: np.ndarray, axis: Optional[int] = None, ddof: int = 0, keepdims: bool = False) -> np.ndarray:
    # TODO: Compute standard deviation along specified axis
    raise NotImplementedError


def median(arr: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    # TODO: Compute median along specified axis
    raise NotImplementedError


def minimum(arr: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    """Compute the minimum of an array along a specified axis."""
    # TODO: Calculate minimum along axis with shape consistency
    raise NotImplementedError


def maximum(arr: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    """Compute the maximum of an array along a specified axis."""
    # TODO: Calculate maximum along axis with shape consistency
    raise NotImplementedError


def histogram(
    data: np.ndarray,
    bins: int = 10,
    bin_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns: (counts, bin_edges)"""
    # TODO: Compute histogram counts and bin edges
    raise NotImplementedError


def quantiles(
    data: np.ndarray,
    q: Union[float, Sequence[float]],
    axis: Optional[int] = None,
    handle_nan: bool = True,
) -> np.ndarray:
    """Compute quantiles with optional NaN handling."""
    # TODO: Handle NaNs and calculate quantiles
    raise NotImplementedError


class WelfordStats:
    """Online streaming calculation of mean and variance via Welford's algorithm."""

    def __init__(self) -> None:
        self.count = 0
        self.mean_ = 0.0
        self.M2 = 0.0

    def update(self, x: Union[float, np.ndarray]) -> None:
        # TODO: Update running mean and M2 using Welford's recurrence
        raise NotImplementedError

    @property
    def variance(self) -> float:
        # TODO: Return sample variance
        raise NotImplementedError