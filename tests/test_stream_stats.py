"""Tests for column-wise stream statistics."""

import numpy as np
import pytest

from numcompute_stream.stats import StreamingStats


def test_streaming_moments_match_batch_statistics():
    X = np.array([[0, np.nan], [2, 8], [10, np.nan], [20, 4]], dtype=float)
    stats = StreamingStats(keep_history=False)

    assert stats.update_stats(X[:3]) is stats
    stats.update_stats(X[3:])

    np.testing.assert_array_equal(stats.count, [4, 2])
    np.testing.assert_allclose(stats.mean, np.nanmean(X, axis=0))
    np.testing.assert_allclose(stats.variance, np.nanvar(X, axis=0))


def test_missing_columns_recover_when_observations_arrive():
    stats = StreamingStats().update_stats([[4, np.nan], [4, np.nan]])
    np.testing.assert_allclose(stats.mean, [4, np.nan])
    np.testing.assert_allclose(stats.variance, [0, np.nan])
    np.testing.assert_allclose(stats.quantiles(0.5), [4, np.nan])

    stats.update_stats([[np.nan, 10]])
    np.testing.assert_array_equal(stats.count, [2, 1])
    np.testing.assert_allclose(stats.mean, [4, 10])
    np.testing.assert_allclose(stats.variance, [0, 0])


def test_streaming_variance_handles_a_large_offset():
    X = (1e12 + np.arange(12.0)).reshape(6, 2)
    stats = StreamingStats().update_stats(X[:2]).update_stats(X[2:])

    np.testing.assert_allclose(stats.variance, np.var(X, axis=0), atol=1e-3)


def test_quantiles_use_exact_history_and_keep_input_shape():
    X = np.array([[0, np.nan], [10, 8], [100, 4], [110, 2]])
    chunk = X[:1].copy()
    stats = StreamingStats().update_stats(chunk)
    chunk[:] = 999
    stats.update_stats(X[1:])

    q = [0.25, 0.5, 0.75]
    np.testing.assert_allclose(stats.quantiles(q), np.nanquantile(X, q, axis=0))
    assert stats.quantiles(0.5).shape == (2,)
    assert stats.quantiles([]).shape == (0, 2)
    assert stats.quantiles(np.empty((2, 0))).shape == (2, 0, 2)
    assert stats.quantiles(np.empty((0, 2))).shape == (0, 2, 2)


def test_histogram_accumulates_counts_without_history():
    edges = np.array([0.0, 1.0, 3.0])
    stats = StreamingStats(edges, keep_history=False)
    edges[:] = 100

    stats.update_stats([[-1, np.nan], [0, 1]])
    stats.update_stats([[1, 0.5], [3, 3], [4, 4]])
    counts, bins = stats.histogram()

    np.testing.assert_array_equal(counts, [[1, 2], [1, 2]])
    np.testing.assert_array_equal(bins, [0, 1, 3])
    counts[:] = -1
    bins[:] = -1
    np.testing.assert_array_equal(stats.histogram()[0], [[1, 2], [1, 2]])
    np.testing.assert_array_equal(stats.histogram()[1], [0, 1, 3])


def test_invalid_chunks_preserve_accumulated_statistics():
    stats = StreamingStats([0, 10]).update_stats([[2, 4]])
    for chunk in ([[1]], np.empty((0, 2)), [[np.inf, 1]], [["x", "y"]]):
        with pytest.raises(ValueError):
            stats.update_stats(chunk)

    with pytest.raises(ValueError, match="floating-point range"):
        stats.update_stats([[1e308, 0]])

    np.testing.assert_array_equal(stats.count, [1, 1])
    np.testing.assert_allclose(stats.quantiles(0.5), [2, 4])
    np.testing.assert_array_equal(stats.histogram()[0], [[1], [1]])


def test_streaming_stats_validates_queries_and_bin_edges():
    stats = StreamingStats()
    with pytest.raises(RuntimeError, match="update_stats"):
        stats.mean

    stats.update_stats([[1, 2]])
    with pytest.raises(RuntimeError, match="bin_edges"):
        stats.histogram()

    for q in (-0.1, 1.1, np.nan, "0.5"):
        with pytest.raises(ValueError, match="q must"):
            stats.quantiles(q)

    for edges in ([1], [1, 1], [1, 0], [0, np.inf], ["zero", "one"]):
        with pytest.raises(ValueError, match="bin_edges"):
            StreamingStats(edges)

    with pytest.raises(RuntimeError, match="keep_history"):
        StreamingStats(keep_history=False).update_stats([[1]]).quantiles(0.5)
