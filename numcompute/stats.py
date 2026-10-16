"""
Descriptive statistics, Welford streaming stats, histogram, and quantiles.
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union
import numpy as np


def mean(arr: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    return np.mean(arr, axis=axis, keepdims=keepdims)
    

def std(arr: np.ndarray, axis: Optional[int] = None, ddof: int = 0, keepdims: bool = False) -> np.ndarray:
    return np.std(arr, axis=axis, keepdims=keepdims, ddof=ddof)


def median(arr: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    return np.median(arr, axis=axis, keepdims=keepdims)


def minimum(arr: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    return np.min(arr, axis=axis, keepdims=keepdims)


def maximum(arr: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    return np.max(arr, axis=axis, keepdims=keepdims)


def histogram(
    data: np.ndarray,
    bins: int = 10,
    bin_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    counts, bin_edges = np.histogram(
        data, bins=bins, range=bin_range
    )

    return counts, bin_edges

def quantiles(
    data: np.ndarray,
    q: Union[float, Sequence[float]],
    axis: Optional[int] = None,
    handle_nan: bool = True,
) -> np.ndarray:
    """Compute quantiles with optional NaN handling."""
    if handle_nan:
        return np.nanquantile(data, q, axis=axis)
    else:
        return np.quantile(data, q, axis=axis)

class WelfordStats:
    """Online streaming calculation of mean and variance via Welford's algorithm."""

    def __init__(self) -> None:
        self.count = 0
        self.mean_ = 0.0
        self.M2 = 0.0

    def update(self, x: Union[float, np.ndarray]) -> None:
        values = np.asarray(x).ravel()

        for value in values:
            self.count += 1

            diff1 = value - self.mean_
            self.mean_ += diff1 / self.count
            diff2 = value - self.mean_

            self.M2 += diff1 * diff2


    @property
    def variance(self) -> float:
        if self.count < 2:
            raise ValueError("Error: Atleast two observations needed")

        return self.M2 / (self.count-1)