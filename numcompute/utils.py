"""
Mathematical and data utility helpers.
"""

from __future__ import annotations
from typing import Generator, Optional, Tuple
import numpy as np



def euclidean_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between rows of X and Y.

    Parameters
    ----------
    X : np.ndarray
        First input array, shape (n_samples_X, n_features).
    Y : np.ndarray
        Second input array, shape (n_samples_Y, n_features).

    Returns
    -------
    np.ndarray
        Distance matrix of shape (n_samples_X, n_samples_Y).
        Each element contains the distance between two rows.

    Raises
    ------
    ValueError
        If inputs are not 2D, feature counts differ, there
        are no features, or inputs contain NaN or infinity.
    TypeError
        If inputs do not contain real numeric values.

    Complexity
    ----------
    Time: O(n * m * d)
    Space: O(n * m * d)
    where n and m are the numbers of rows, and d is
    the number of features.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Both inputs must be 2D arrays.")

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Inputs must have the same number of features.")

    if X.shape[1] == 0:
        raise ValueError("Inputs must contain at least one feature.")

    for arr in (X, Y):
        if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
            raise TypeError("Inputs must contain real numeric values.")

    X = X.astype(float)
    Y = Y.astype(float)

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError("Inputs must contain finite values.")

    differences = X[:, np.newaxis, :] - Y[np.newaxis, :, :]

    distances = np.hypot.reduce(differences, axis=2)

    return distances




def cosine_similarity(X: np.ndarray, Y: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """
    Compute pairwise cosine similarity between rows of X and Y.

    Parameters
    ----------
    X : np.ndarray
        First array, shape (n_samples_X, n_features).
    Y : np.ndarray
        Second array, shape (n_samples_Y, n_features).
    eps : float
        Positive tolerance for near-zero vector norms.
        Such vectors are treated as zero vectors.

    Returns
    -------
    np.ndarray
        Similarity matrix of shape (n_samples_X, n_samples_Y).
        Values lie between -1 and 1.
        Similarity involving a zero vector is defined as 0.

    Raises
    ------
    ValueError
        If inputs are not 2D, feature counts differ,
        contain no features, contain NaN or infinity,
        or eps is not positive and finite.
    TypeError
        If inputs are not real numeric arrays.

    Complexity
    ----------
    Time: O(n * m * d)
    Space: O(n * d + m * d + n * m)
    where n and m are row counts, and d is feature count.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Both inputs must be 2D arrays.")

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Inputs must have the same number of features.")

    if X.shape[1] == 0:
        raise ValueError("Inputs must contain at least one feature.")

    for arr in (X, Y):
        if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
            raise TypeError("Inputs must contain real numeric values.")

    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite.")

    X = X.astype(float)
    Y = Y.astype(float)

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
        raise ValueError("Inputs must contain finite values.")

    # Calculate vector magnitudes
    X_norm = np.hypot.reduce(X, axis=1)
    Y_norm = np.hypot.reduce(Y, axis=1)

    # Normalize each row, safely handling zero vectors
    X_unit = np.divide(
        X, X_norm[:, None],
        out=np.zeros_like(X),
        where=X_norm[:, None] > eps
    )

    Y_unit = np.divide(
        Y, Y_norm[:, None],
        out=np.zeros_like(Y),
        where=Y_norm[:, None] > eps
    )

    # Calculate all pairwise similarities
    similarities = X_unit @ Y_unit.T

    return np.clip(similarities, -1.0, 1.0)




def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute numerically stable softmax along a given axis.

    Parameters
    ----------
    x : np.ndarray
        Input numerical array with one or more dimensions.
    axis : int
        Axis along which softmax is calculated.
        Default is -1 (last axis).

    Returns
    -------
    np.ndarray
        Probabilities with the same shape as x.
        Values are between 0 and 1 and sum to 1
        along the chosen axis.

    Raises
    ------
    ValueError
        If x is empty, has no dimensions, contains
        NaN or infinity, or axis is out of range.
    TypeError
        If x is not a real numeric array or axis
        is not an integer.

    Complexity
    ----------
    Time: O(N)
    Space: O(N)
    where N is the total number of elements.
    """
    x = np.asarray(x)

    if x.ndim == 0 or x.size == 0:
        raise ValueError("Input must be a non-empty array.")

    if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise TypeError("Input must contain real numeric values.")

    if not isinstance(axis, (int, np.integer)):
        raise TypeError("Axis must be an integer.")

    if axis < -x.ndim or axis >= x.ndim:
        raise ValueError("Axis is out of range.")

    x = x.astype(np.float64)

    if not np.all(np.isfinite(x)):
        raise ValueError("Input must contain finite values.")

    # Subtract the maximum for numerical stability
    shifted = x - np.max(x, axis=axis, keepdims=True)

    # Calculate exponentials
    exp_values = np.exp(shifted)

    # Calculate sum along selected axis
    sum_exp = np.sum(exp_values, axis=axis, keepdims=True)

    # Convert to probabilities
    return exp_values / sum_exp




def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute numerically stable logsumexp along a given axis.

    Parameters
    ----------
    x : np.ndarray
        Input numerical array with one or more dimensions.
    axis : int
        Axis along which logsumexp is calculated.
        Default is -1 (last axis).

    Returns
    -------
    np.ndarray
        Logsumexp values with the selected axis removed.
        For a 1D input, returns a 0D NumPy array.

    Raises
    ------
    ValueError
        If input is empty, has no dimensions, contains
        NaN or infinity, or axis is out of range.
    TypeError
        If input is not real numeric or axis is not an integer.

    Complexity
    ----------
    Time: O(N)
    Space: O(N)
    where N is the total number of input elements.
    """
    x = np.asarray(x)

    if x.ndim == 0 or x.size == 0:
        raise ValueError("Input must be a non-empty array.")

    if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise TypeError("Input must contain real numeric values.")

    if not isinstance(axis, (int, np.integer)):
        raise TypeError("Axis must be an integer.")

    if axis < -x.ndim or axis >= x.ndim:
        raise ValueError("Axis is out of range.")

    x = x.astype(np.float64)

    if not np.all(np.isfinite(x)):
        raise ValueError("Input must contain finite values.")

    # Find maximum along the selected axis
    max_values = np.max(x, axis=axis, keepdims=True)

    # Subtract maximum to prevent overflow
    shifted = x - max_values

    # Calculate sum of exponentials
    sum_exp = np.sum(np.exp(shifted), axis=axis, keepdims=True)

    # Apply logarithm and add back maximum
    result = max_values + np.log(sum_exp)

    # Remove the reduced dimension
    return np.squeeze(result, axis=axis)




def batch_generator(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    batch_size: int = 32,
    shuffle: bool = True,
) -> Generator[Tuple[np.ndarray, Optional[np.ndarray]], None, None]:
    """
    Generate mini-batches of data with optional shuffling.

    Parameters
    ----------
    X : np.ndarray
        Input samples, shape (n_samples, ...).
    y : np.ndarray or None
        Optional target labels, shape (n_samples, ...).
    batch_size : int
        Number of samples per batch. Must be positive.
    shuffle : bool
        Whether to randomly shuffle samples before batching.

    Yields
    ------
    X_batch : np.ndarray
        A batch of input samples.
    y_batch : np.ndarray or None
        Corresponding target labels, or None if not provided.

    Raises
    ------
    ValueError
        If X has no sample dimension, batch_size is not
        positive, or X and y have different sample counts.
    TypeError
        If batch_size is not an integer or shuffle
        is not a boolean.

    Complexity
    ----------
    Time: O(n * d) across all batches.
    Space: O(n + b * d) when shuffling,
    where n is the number of samples, b is the batch
    size, and d is the number of values per sample.
    """
    X = np.asarray(X)

    if X.ndim == 0:
        raise ValueError("X must have a sample dimension.")

    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(
        batch_size, (int, np.integer)
    ):
        raise TypeError("batch_size must be an integer.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if not isinstance(shuffle, (bool, np.bool_)):
        raise TypeError("shuffle must be a boolean.")

    n_samples = X.shape[0]

    if y is not None:
        y = np.asarray(y)

        if y.ndim == 0 or y.shape[0] != n_samples:
            raise ValueError("X and y must have the same sample count.")

    # Generate sample indices
    indices = np.arange(n_samples)

    # Randomly shuffle indices if requested
    if shuffle:
        indices = np.random.permutation(indices)

    # Generate batches
    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        batch_indices = indices[start:end]

        X_batch = X[batch_indices]

        if y is not None:
            y_batch = y[batch_indices]
        else:
            y_batch = None

        yield X_batch, y_batch
