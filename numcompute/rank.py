"""
Ranking with tie-breaking and percentile calculations.
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union
import numpy as np

METHODS = {"average", "dense", "ordinal"}
INTERPOLATIONS = {"linear", "lower", "higher", "midpoint"}

def rank(
    data: np.ndarray,
    method: str = "average",
    axis: int = -1,
) -> np.ndarray:
    """Compute ranks of data along an axis.

    Parameters
    data : np.ndarray
        Input array of any shape
    method : average, dense, ordinal, default 'average'
        Tie breaking rule:
        - 'average': tied elements share the mean of the ranks
        - 'dense': ties share a rank, next distinct value takes the next integer
        - 'ordinal': ties broken by index order
    axis : int, default -1
        Axis along which each 1D slice is ranked independently.

    Returns
    np.ndarray
        Same shape as data

    Raises
    ValueError
        If method is not in ('average', 'dense', 'ordinal')
    numpy.AxisError
        If axis is out of bounds for data.ndim

    Complexity
    Time O(n log n) per slice + O(n) in tie-breaking. Space O(n) per slice.
    """
    if method not in METHODS:
        raise ValueError(f"Unsupported method")

    data = np.asarray(data, dtype=float)
    moved = np.moveaxis(data, axis, -1)
    shape = moved.shape
    n = shape[-1]
    rows = int(np.prod(shape[:-1]))
    flat = moved.reshape(rows, n)

    if n == 0:
        ranks = flat.copy()
    else:
        nan_bool = np.isnan(flat)
        # double argsort to get index ranks
        order = np.argsort(flat, axis=-1, kind="stable")
        rank_index = np.argsort(order, axis=-1)

        if method == "ordinal":
            ranks = (rank_index + 1).astype(float)
        else:
            sorted_by_val = np.take_along_axis(flat, order, axis=-1)

            # mark start of each tie group
            tie_start = np.empty_like(sorted_by_val, dtype=bool)
            tie_start[:, 0] = True
            tie_start[:, 1:] = sorted_by_val[:, 1:] != sorted_by_val[:, :-1]
            tie_group_id = np.cumsum(tie_start, axis=-1)  # 1-indexed run id per sorted slot

            if method == "dense":
                ranks = np.take_along_axis(tie_group_id.astype(float), rank_index, axis=-1)
            else:
                tie_end = np.empty_like(sorted_by_val, dtype=bool)
                tie_end[:, -1] = True
                tie_end[:, :-1] = sorted_by_val[:, :-1] != sorted_by_val[:, 1:]

                positions = np.arange(1, n + 1)
                # each tie group's start position
                tie_group_start = np.maximum.accumulate(
                    np.where(tie_start, positions, 0), axis=-1
                )
                # get tie group's end position from right to left, then reverse
                tie_group_end = np.minimum.accumulate(
                    np.where(tie_end, positions, n + 1)[:, ::-1], axis=-1
                )[:, ::-1]

                avg_sorted = (tie_group_start + tie_group_end) / 2.0
                ranks = np.take_along_axis(avg_sorted, rank_index, axis=-1)

        ranks[nan_bool] = np.nan

    ranks = ranks.reshape(shape)
    return np.moveaxis(ranks, -1, axis)


def percentile(
    data: np.ndarray,
    q: Union[float, Sequence[float]],
    interpolation: str = "linear",
    axis: Optional[int] = None,
) -> np.ndarray:
    """Compute the q-th percentile of data along an axis.

    Parameters
    data : np.ndarray
        Input array of any shape.
    q : float or Sequence[float]
        Percentile(s) to compute, each in [0, 100]. NaN is rejected, not
        treated as a percentile to compute.
    interpolation : {'linear', 'lower', 'higher', 'midpoint'}, default 'linear'
        Interpolation rule between the two nearest ranks when the exact
        percentile falls between data points.
    axis : int or None, default None
        Axis to reduce over

    Returns
    - np.ndarray of dtype float64.
    - If q is a scalar and axis is None: 0-d array
    - If q is a scalar and axis is given: data.shape with axis removed.
    - If q is a sequence of length m and axis is None: shape (m,)
    - If q is a sequence of length m and axis is given: shape (m,) + data.shape with axis removed.

    Raises
    ValueError
        If interpolation is invalid, or q is outside [0, 100] or NaN.
        If data has zero elements along the reduction axis
    numpy.AxisError
        If axis is out of bounds for data.ndim.

    Complexity
    Time O(n log n) per slice (sort) + O(m) for the m requested
    quantiles. Space O(n) per slice for the sorted copy.
    """
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {INTERPOLATIONS}")

    data = np.asarray(data, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    q_scalar = q_arr.ndim == 0
    q_flat = np.atleast_1d(q_arr)
    if not np.all((q_flat >= 0) & (q_flat <= 100)):
        raise ValueError(f"q must be between 0 and 100 and not NaN")

    if axis is None:
        flat = data.reshape(1, -1)
        out_shape: Tuple[int, ...] = ()
    else:
        moved = np.moveaxis(data, axis, -1)
        out_shape = moved.shape[:-1]
        rows = int(np.prod(out_shape))
        flat = moved.reshape(rows, moved.shape[-1])

    rows, n = flat.shape

    if n == 0:
        result = np.full((rows, len(q_flat)), np.nan)
    else:
        sorted_by_val = np.sort(flat, axis=-1, kind="stable")
        valid_counts = n - np.isnan(sorted_by_val).sum(axis=-1)
        # (q / 100) * (valid_counts - 1)
        frac_index = (q_flat[None, :] / 100.0) * (valid_counts[:, None] - 1)
        lower_index = np.floor(frac_index).astype(int)
        upper_index = np.ceil(frac_index).astype(int)
        frac = frac_index - lower_index

        lower_vals = np.take_along_axis(sorted_by_val, lower_index, axis=-1)
        upper_vals = np.take_along_axis(sorted_by_val, upper_index, axis=-1)

        if interpolation == "linear":
            result = lower_vals + (upper_vals - lower_vals) * frac
        elif interpolation == "lower":
            result = lower_vals
        elif interpolation == "higher":
            result = upper_vals
        else:
            result = (lower_vals + upper_vals) / 2.0

        result[valid_counts == 0] = np.nan

    result = result.T.reshape((len(q_flat),) + out_shape)
    if q_scalar:
        result = result[0]
    return result
