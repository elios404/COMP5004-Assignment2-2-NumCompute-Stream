"""
Sorting, partial sorting (top-k), quickselect, and binary search.
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np


def stable_sort(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    """Wrapper for stable sort preserving relative order of equal keys."""
    arr = np.asarray(arr)
    return np.sort(arr, axis=axis, kind="stable")


def multi_key_sort(arr: np.ndarray, columns: Sequence[int], ascending: bool = True) -> np.ndarray:
    """Sort a 2D array by multiple columns in sequence."""
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
    """
    Search for x in a sorted 1D array.
    Returns: (insertion_index, exists_boolean)
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
    """
    Find top-k values and optional indices using np.argpartition.
    Returns: (topk_values, topk_indices)
    """
    arr = np.asarray(values)
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
    topk_values = stable_sort(selected_values)
    if largest:
        topk_values = topk_values[::-1]

    topk_indices = None
    if return_indices:
        order = np.argsort(selected_values, kind="stable")
        if largest:
            order = order[::-1]
        topk_indices = idx[order]

    return topk_values, topk_indices


def quickselect(arr: np.ndarray, k: int) -> float:
    """
    Find the k-th smallest element (0-indexed) in an array using the
    Quickselect algorithm.
    """
    values = np.array(arr, dtype=float).ravel()
    n = values.size
    if not (0 <= k < n):
        raise ValueError(f"{k} is out of bounds for array")

    low, high = 0, n - 1
    while True:
        if low == high:
            return float(values[low])

        mid = (low + high) // 2
        values[mid], values[high] = values[high], values[mid]
        pivot = values[high]

        store = low
        for i in range(low, high):
            if values[i] < pivot:
                values[i], values[store] = values[store], values[i]
                store += 1
        values[store], values[high] = values[high], values[store]
        pivot_index = store

        if k == pivot_index:
            return float(values[k])
        if k < pivot_index:
            high = pivot_index - 1
        else:
            low = pivot_index + 1