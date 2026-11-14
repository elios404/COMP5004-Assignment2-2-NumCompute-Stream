"""Column-wise statistics for streams of numeric data."""

from __future__ import annotations

import numpy as np

from ._moments import updated_moments
from .preprocessing import _numeric_input


class StreamingStats:
    """Track statistics independently for each column of an (N, D) stream.

    NaNs are skipped. Keep history for exact quantiles, or set
    keep_history=False to retain only the running statistics.
    Supply fixed bin_edges to accumulate a histogram for each column.
    """

    def __init__(self, bin_edges: np.ndarray | None = None, *, keep_history: bool = True):
        self.keep_history = keep_history
        self._bin_edges = None
        if bin_edges is not None:
            edges = np.asarray(bin_edges)
            if edges.ndim != 1 or edges.size < 2 or edges.dtype.kind not in "iuf":
                raise ValueError("bin_edges must contain at least two real edges.")
            with np.errstate(over="ignore"):
                edges = edges.astype(np.float64)
            if not np.isfinite(edges).all() or not np.all(edges[:-1] < edges[1:]):
                raise ValueError("bin_edges must be finite and strictly increasing.")
            self._bin_edges = edges

        self._moments = None
        self._histogram = None
        self._chunks: list[np.ndarray] = []

    def update_stats(self, X_chunk: np.ndarray) -> StreamingStats:
        """Add a non-empty (N, D) chunk and return this accumulator."""
        n_features = None if self._moments is None else self._moments[0].size
        X = _numeric_input(X_chunk, n_features, training=True)

        moments = updated_moments(X, self._moments)
        if self._bin_edges is not None:
            counts = np.stack([np.histogram(column, self._bin_edges)[0] for column in X.T])
            if self._histogram is not None:
                counts += self._histogram
            self._histogram = counts

        if self.keep_history:
            self._chunks.append(X.copy())
        self._moments = moments

        return self

    def _require_data(self) -> None:
        if self._moments is None:
            raise RuntimeError("Call update_stats before reading statistics.")

    @property
    def count(self) -> np.ndarray:
        """Return the number of observed values per column, shape (D,)."""
        self._require_data()
        return self._moments[0].copy()

    @property
    def mean(self) -> np.ndarray:
        """Return the mean per column, using NaN for unobserved columns."""
        self._require_data()
        count, mean, _ = self._moments
        return np.where(count > 0, mean, np.nan)

    @property
    def variance(self) -> np.ndarray:
        """Return the population variance per column, shape (D,)."""
        self._require_data()
        count, _, variance = self._moments
        return np.where(count > 0, variance, np.nan)

    def quantiles(self, q: float | np.ndarray) -> np.ndarray:
        """Return exact linear quantiles with shape q.shape + (D,)."""
        self._require_data()
        if not self.keep_history:
            raise RuntimeError("Quantiles require keep_history=True.")

        q = np.asarray(q)
        if q.dtype.kind not in "iuf" or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise ValueError("q must contain finite numbers in [0, 1].")

        history = np.concatenate(self._chunks, axis=0)
        observed = self._moments[0] > 0
        result = np.full(q.shape + (history.shape[1],), np.nan)
        if observed.any():
            result[..., observed] = np.nanquantile(history[:, observed], q, axis=0)

        return result

    def histogram(self) -> tuple[np.ndarray, np.ndarray]:
        """Return counts (D, B) and fixed edges (B + 1,) as copies.

        Values outside the edge range are excluded. The final bin includes
        its right edge.
        """
        self._require_data()
        if self._bin_edges is None:
            raise RuntimeError("Histogram counts require bin_edges.")

        return self._histogram.copy(), self._bin_edges.copy()
