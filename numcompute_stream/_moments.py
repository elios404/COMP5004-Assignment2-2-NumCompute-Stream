"""Shared calculations for column-wise moments."""

import numpy as np


def _chunk_mean(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = ~np.isnan(X)
    count = observed.sum(axis=0)
    mean = (np.where(observed, X, 0.0) / np.maximum(count, 1)).sum(axis=0)
    return count, mean


def _merge_means(
    count: np.ndarray, mean: np.ndarray, chunk_count: np.ndarray, chunk_mean: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    total = count + chunk_count
    chunk_weight = chunk_count / np.maximum(total, 1)
    merged = mean * (count / np.maximum(total, 1)) + chunk_mean * chunk_weight

    return total, merged


def updated_moments(
    X: np.ndarray,
    previous: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None = None,
    with_variance: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Merge per-column counts, means and population variances, ignoring NaNs."""
    with np.errstate(over="ignore", invalid="ignore"):
        count, mean = _chunk_mean(X)
        variance = None
        if with_variance:
            centred = np.where(np.isnan(X), 0.0, X - mean)
            scaled = centred / np.sqrt(np.maximum(count, 1))
            variance = np.square(scaled).sum(axis=0)

        if previous is not None:
            old_count, old_mean, old_variance = previous
            total, merged_mean = _merge_means(old_count, old_mean, count, mean)

            if with_variance:
                old_weight = old_count / np.maximum(total, 1)
                new_weight = count / np.maximum(total, 1)
                weight = np.sqrt(old_weight * new_weight)
                distance = mean * weight - old_mean * weight
                variance = old_weight * old_variance + new_weight * variance + distance ** 2

            count, mean = total, merged_mean

    if not np.isfinite(mean).all() or (variance is not None and not np.isfinite(variance).all()):
        raise ValueError("Statistics exceed the floating-point range.")

    return count, mean, variance
