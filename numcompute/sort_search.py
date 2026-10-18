"""
Sorting, partial sorting (top-k), quickselect, and binary search.
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np


def stable_sort(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable (order-preserving) ascending sort.

    Parameters
    arr : np.ndarray
        Input array of any shape; sorted along `axis`
    axis : int, default -1
        Axis along which to sort

    Returns
    np.ndarray: same shape and dtype as `arr`. Equal elements keep their relative order

    Raises
    numpy.AxisError
        If `axis` is out of bounds for `arr.ndim`

    Complexity
    Time O(nlogn), space O(n).
    """
    arr = np.asarray(arr)
    return np.sort(arr, axis=axis, kind="stable")


def multi_key_sort(arr: np.ndarray, columns: Sequence[int], ascending: bool = True) -> np.ndarray:
    """Sort the rows of a 2D array by multiple column priorities.

    Parameters
    arr : np.ndarray, shape (m, n)
        2D array to sort by row
    columns : Sequence[int]
        Column indices in priority order
    ascending : bool, default True
        Sort direction, applied to every key column

    Returns
    np.ndarray, shape (m, n)
        A new array with rows reordered; `arr` is not modified

    Raises
    IndexError
        If any entry of `columns` is out of bounds for `arr`'s column axis

    Complexity
    Time O(mlogm), space O(m) for the order plus
    O(m * n) for the returned copy
    """
    arr = np.asarray(arr)
    keys = []
    for col in reversed(columns):
        key = arr[:, col]
        if not ascending:
            key = -key
        keys.append(key)
    order = np.lexsort(keys)
    return arr[order]


def binary_search(sorted_array: np.ndarray, x: float) -> Tuple[int, bool]:
    """Binary search for `x` in an ascending-sorted 1D array.

    Parameters
    sorted_array : np.ndarray, shape (n,)
        1D array assumed sorted ascending
    x : float
        target value

    Returns
    Tuple[int, bool]
        `(insertion_index, exists)`: leftmost index to insert 'x', and whether `x` is at that index.

    Complexity
    Time O(log n), space O(1).
    """
    sorted_array = np.asarray(sorted_array)
    index = int(np.searchsorted(sorted_array, x, side="left"))
    exist = index < sorted_array.size and sorted_array[index] == x
    return index, bool(exist)


def topk(
    values: np.ndarray,
    k: int,
    largest: bool = True,
    return_indices: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Select the k largest/smallest values via partial sort.

    Parameters
    values : np.ndarray
        Input array of any shape
    k : int
        Number of elements to select
    largest : bool, default True
        If True, select the k largest values in descending order, else k smallest in ascending order. NaNs sort as the
        largest value, per NumPy's total ordering.
    return_indices : bool, default True
        If True, also return the selected indices

    Returns
    Tuple[np.ndarray, Optional[np.ndarray]]
        `(topk_values, topk_indices)`, each of shape `(min(k, values.size),)`.
        `topk_indices` is `None` when `return_indices` is False.

    Raises
    ValueError
        If `k` is negative.

    Complexity
    Time O(n) + O(klogk) for `np.argpartition` and k selected elements. 
    Space O(n) + O(k) for the flattened copy and the output.
    """
    arr = np.asarray(values).ravel()
    n = arr.size
    if k < 0:
        raise ValueError("k must be non-negative")
    k = min(k, n)

    if k == 0:
        idx = np.array([], dtype=int)
    elif largest:
        kth = n - k
        idx = np.argpartition(arr, kth)[kth:]
    else:
        idx = np.argpartition(arr, k - 1)[:k]

    selected_values = arr[idx]
    order = np.argsort(selected_values, kind="stable")
    if largest:
        order = order[::-1]

    topk_values = selected_values[order]
    topk_indices = None
    if return_indices:
        topk_indices = idx[order]
    return topk_values, topk_indices


def quickselect(arr: np.ndarray, k: int) -> float:
    """Find the k-th smallest element (0-indexed) of `arr`.

    Parameters
    arr : np.ndarray
        Input array of any shape
    k : int
        0-indexed rank to select (0 = smallest, `arr.size - 1` = largest).

    Returns
    float
        The k-th smallest element

    Raises
    ValueError
        If `k` is not in `[0, arr.size)`

    Complexity
    Time O(n) average case, space O(n)
    """
    values = np.asarray(arr).ravel()
    n = values.size
    if not (0 <= k < n):
        raise ValueError(f"{k} is out of bounds for array")

    return float(np.partition(values, k)[k])
