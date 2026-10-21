"""
Descriptive statistics, Welford streaming stats, histogram, and quantiles.
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union
import numpy as np


def mean(arr: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    
    """
    Compute the arithmetic mean of an array.

    Parameters
    ----------
    arr : array_like
        Input array containing numerical values. Can have any number
        of dimensions.
    axis : int or tuple of int, optional
        Axis or axes along which to calculate the mean. If None,
        calculate the mean over all elements. Default is None.
    keepdims : bool, optional
        If True, retain reduced dimensions with size one.
        Default is False.

    Returns
    -------
    ndarray or scalar
        Mean values. If axis is specified, the corresponding
        dimensions are reduced, unless keepdims=True.

    Raises
    ------
    ValueError
        If an axis is out of bounds or the input cannot be reduced
        along the specified axis.

    Complexity
    ----------
    Time: O(N), where N is the number of input elements.
    Auxiliary space: Implementation-dependent, depending on the
    NumPy operation and output size.
    """
    return np.mean(arr, axis=axis, keepdims=keepdims)
    

def std(arr: np.ndarray, axis: Optional[int] = None, ddof: int = 0, keepdims: bool = False) -> np.ndarray:
    """
    Compute the standard deviation of array elements.

    Parameters
    ----------
    arr : array_like
        Input array containing numerical values.
    axis : int or tuple of int, optional
        Axis or axes along which to calculate standard deviation.
        If None, use all elements. Default is None.
    keepdims : bool, optional
        If True, retain reduced dimensions with size one.
        Default is False.
    ddof : int, optional
        Delta degrees of freedom. The divisor is N - ddof,
        where N is the number of elements used in the calculation.
        Default is 0.

    Returns
    -------
    ndarray or scalar
        Standard deviation values along the specified axis or axes.

    Raises
    ------
    ValueError
        If an axis is invalid.

    Notes
    -----
    NumPy may return NaN or emit a warning when there are
    insufficient observations for the chosen ddof.

    Complexity
    ----------
    Time: O(N), where N is the number of input elements.
    Auxiliary space: Implementation-dependent, depending on the
    NumPy operation and output size.
    """
    return np.std(arr, axis=axis, keepdims=keepdims, ddof=ddof)


def median(arr: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    """
    Compute the median of an array.

    Parameters
    ----------
    arr : array_like
        Input array containing numerical values.
    axis : int or tuple of int, optional
        Axis or axes along which to calculate the median.
        If None, use all elements. Default is None.
    keepdims : bool, optional
        If True, retain reduced dimensions with size one.
        Default is False.

    Returns
    -------
    ndarray or scalar
        Median values along the selected axis or axes.

    Raises
    ------
    ValueError
        If an axis is invalid or the input is empty.

    Complexity
    ----------
    Time: Typically O(N log N) as a conservative bound for
    selection or ordering operations.
    Auxiliary space: Implementation-dependent.
    """

    return np.median(arr, axis=axis, keepdims=keepdims)


def minimum(arr: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    """
    Find the minimum value of an array.

    Parameters
    ----------
    arr : array_like
        Input array containing numerical values.
    axis : int or tuple of int, optional
        Axis or axes along which to find the minimum.
        If None, use all elements. Default is None.
    keepdims : bool, optional
        If True, retain reduced dimensions with size one.
        Default is False.

    Returns
    -------
    ndarray or scalar
        Minimum values along the selected axis or axes.

    Raises
    ------
    ValueError
        If an axis is invalid or the input is empty.

    Complexity
    ----------
    Time: O(N), where N is the number of input elements.
    Auxiliary space: Implementation-dependent, depending on
    the NumPy operation and output size.
    """
    return np.min(arr, axis=axis, keepdims=keepdims)


def maximum(arr: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    """
    Find the maximum value of an array.

    Parameters
    ----------
    arr : array_like
        Input array containing numerical values.
    axis : int or tuple of int, optional
        Axis or axes along which to find the maximum.
        If None, use all elements. Default is None.
    keepdims : bool, optional
        If True, retain reduced dimensions with size one.
        Default is False.

    Returns
    -------
    ndarray or scalar
        Maximum values along the selected axis or axes.

    Raises
    ------
    ValueError
        If an axis is invalid or the input is empty.

    Complexity
    ----------
    Time: O(N), where N is the number of input elements.
    Auxiliary space: Implementation-dependent, depending on
    the NumPy operation and output size.
    """
    return np.max(arr, axis=axis, keepdims=keepdims)


def histogram(
    data: np.ndarray,
    bins: int = 10,
    bin_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    """
    Compute a histogram of the input data.

    Parameters
    ----------
    data : array_like
        Input data to distribute into histogram bins.
    bins : int or array_like, optional
        Number of equally spaced bins or explicit bin edges.
        Default is 10.
    bin_range : tuple of float, optional
        Lower and upper limits of the histogram range.
        Values outside the specified range are excluded.
        Default is None.

    Returns
    -------
    counts : ndarray
        Histogram counts, with shape (B,), where B is the number
        of bins.
    bin_edges : ndarray
        Bin boundaries, with shape (B + 1,).

    Raises
    ------
    ValueError
        If bins or the specified range are invalid.

    Complexity
    ----------
    Time: O(N + B), where N is the number of input values and
    B is the number of bins, under typical conditions.
    Auxiliary space: O(B) for the returned counts and bin edges,
    excluding implementation-dependent temporary storage.
    """
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

    """
    Compute quantiles of the input data.

    Parameters
    ----------
    data : array_like
        Input array containing numerical values.
    q : float or array_like of float
        Quantile or quantiles to calculate, in the interval [0, 1].
    axis : int or tuple of int, optional
        Axis or axes along which to calculate quantiles.
        If None, use all elements. Default is None.
    handle_nan : bool, optional
        If True, ignore NaN values using numpy.nanquantile.
        If False, use numpy.quantile. Default is True.

    Returns
    -------
    ndarray or scalar
        Computed quantile values. When q contains multiple
        quantiles, its dimensions precede the remaining data
        dimensions.

    Raises
    ------
    ValueError
        If q contains values outside [0, 1], an axis is invalid,
        or the selected data cannot produce a quantile.

    Complexity
    ----------
    Time: Typically O(N log N) as a conservative bound for
    ordering or selection operations.
    Auxiliary space: Implementation-dependent.
    """
    if handle_nan:
        return np.nanquantile(data, q, axis=axis)
    else:
        return np.quantile(data, q, axis=axis)

class WelfordStats:
    """Online streaming calculation of mean and variance via Welford's algorithm."""
    """
    Calculate streaming statistics using Welford's algorithm.

    Attributes
    ----------
    count : int
        Number of observations processed.
    mean_ : float
        Running arithmetic mean.
    M2 : float
        Running sum of squared deviations from the current mean.

    Notes
    -----
    The class maintains statistics incrementally without storing
    the entire history of observations.
    """

    def __init__(self) -> None:
        self.count = 0
        self.mean_ = 0.0
        self.M2 = 0.0

    def update(self, x: Union[float, np.ndarray]) -> None:
        """
        Incorporate new observations into the running statistics.

        Parameters
        ----------
        x : array_like
            One or more numerical observations. The input is
            flattened before its values are processed.

        Returns
        -------
        None
            Updates the object's internal statistics in place.

        Raises
        ------
        ValueError
            If the supplied values cannot be processed by the
            underlying numerical operations.

        Complexity
        ----------
        Time: O(K), where K is the number of new observations.
        Auxiliary space: O(K) if flattening creates a copy;
        otherwise, the additional storage depends on the input.
        """
        values = np.asarray(x).ravel()

        for value in values:
            self.count += 1

            diff1 = value - self.mean_
            self.mean_ += diff1 / self.count
            diff2 = value - self.mean_

            self.M2 += diff1 * diff2


    @property
    def variance(self) -> float:
        """
        Return the sample variance of the observations.

        Returns
        -------
        float
            Sample variance, calculated using M2 / (count - 1).

        Raises
        ------
        ValueError
            If fewer than two observations have been processed.

        Complexity
        ----------
        Time: O(1).
        Auxiliary space: O(1).
        """
        if self.count < 2:
            raise ValueError("Error: Atleast two observations needed")

        return self.M2 / (self.count-1)