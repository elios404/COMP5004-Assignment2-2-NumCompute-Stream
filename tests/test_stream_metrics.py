"""Tests for cumulative and rolling classification metrics."""

import numpy as np
import pytest

from numcompute_stream.metrics import StreamingMetrics


def test_cumulative_results_weight_chunks_by_sample_count():
    metrics = StreamingMetrics()
    assert metrics.update([0], [0]) is metrics
    metrics.update([1, 1, 0], [1, 0, 1])

    result = metrics.result()
    assert result["count"] == 4
    np.testing.assert_array_equal(result["confusion_matrix"], [[1, 1], [1, 1]])
    for name in ("accuracy", "precision", "recall", "f1"):
        assert result[name] == pytest.approx(0.5)


def test_multiclass_results_use_macro_averages_and_configured_order():
    metrics = StreamingMetrics(labels=["cat", "dog", "bird"])
    result = metrics.update(["cat", "dog", "cat", "bird"], ["cat", "dog", "dog", "cat"]).result()

    np.testing.assert_array_equal(result["labels"], ["cat", "dog", "bird"])
    np.testing.assert_array_equal(result["confusion_matrix"], [[1, 1, 0], [0, 1, 0], [1, 0, 0]])
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(1 / 3)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(7 / 18)
    assert np.isnan(result["auc"])


def test_auc_combines_scores_across_chunks_and_handles_ties():
    metrics = StreamingMetrics(labels=["yes", "no"])
    scores = np.array([0.2, 0.7])
    metrics.update(["yes", "yes"], ["no", "no"], scores)
    assert np.isnan(metrics.result()["auc"])

    scores[:] = 100
    metrics.update(["no", "no"], ["yes", "yes"], [0.7, 0.9])
    assert metrics.result()["auc"] == pytest.approx(0.875)
    assert metrics.result()["accuracy"] == 0.0


def test_rolling_window_counts_rows_and_evicts_scores():
    metrics = StreamingMetrics(window_size=3)
    metrics.update([0, 1], [0, 1], [0.1, 0.9])
    metrics.update([0, 1], [1, 0], [0.4, 0.2])

    rolling = metrics.rolling_result()
    assert rolling["count"] == 3
    assert rolling["accuracy"] == pytest.approx(1 / 3)
    assert rolling["auc"] == pytest.approx(0.5)

    metrics.update([0, 0, 0, 0], [1, 0, 0, 1])
    np.testing.assert_array_equal(metrics.rolling_result()["confusion_matrix"], [[2, 1], [0, 0]])
    assert np.isnan(metrics.rolling_result()["auc"])
    assert metrics.result()["auc"] == pytest.approx(0.75)
    assert metrics.result()["count"] == 8


def test_metrics_reject_invalid_updates_without_losing_results():
    metrics = StreamingMetrics(window_size=2).update([0, 1], [0, 1], [0.1, 0.9])
    for args in (
        ([[0]], [0]), ([], []), ([0], [0, 1]), ([0], [2]),
        ([np.nan], [0]), ([0], ["0"]), ([0], [0], [np.inf]),
        ([0], [0], [0.1, 0.2]), ([0], [0], ["0.9"]),
    ):
        with pytest.raises(ValueError):
            metrics.update(*args)

    for result in (metrics.result(), metrics.rolling_result()):
        assert result["count"] == 2
        assert result["auc"] == 1.0
        np.testing.assert_array_equal(result["confusion_matrix"], np.eye(2))


def test_metrics_reset_and_result_copies():
    labels = np.array([0, 1])
    metrics = StreamingMetrics(labels, window_size=2).update([1], [1])
    result = metrics.result()
    result["confusion_matrix"][:] = 99
    result["labels"][:] = 99
    labels[:] = 99
    assert metrics.result()["count"] == 1
    np.testing.assert_array_equal(metrics.result()["labels"], [0, 1])

    metrics.reset()
    for result in (metrics.result(), metrics.rolling_result()):
        assert result["count"] == 0
        assert result["precision"] == result["recall"] == result["f1"] == 0
        assert np.isnan(result["accuracy"])
        assert np.isnan(result["auc"])
        np.testing.assert_array_equal(result["confusion_matrix"], np.zeros((2, 2)))


def test_metrics_validate_configuration_and_auc_scope():
    for labels in ([0], [0, 0], [[0, 1]], [0, np.inf]):
        with pytest.raises(ValueError, match="labels"):
            StreamingMetrics(labels)
    for window_size in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="window_size"):
            StreamingMetrics(window_size=window_size)

    metrics = StreamingMetrics(labels=[0, 1, 2])
    with pytest.raises(ValueError, match="binary"):
        metrics.update([0], [0], [0.5])
    with pytest.raises(RuntimeError, match="window_size"):
        metrics.rolling_result()


def test_metrics_keep_large_integer_class_labels_distinct():
    labels = np.array([2**53, 2**53 + 1], dtype=np.uint64)
    metrics = StreamingMetrics(labels)
    metrics.update([float(2**53)], labels[1:])
    np.testing.assert_array_equal(metrics.result()["confusion_matrix"], [[0, 1], [0, 0]])
