"""Tests for the streaming plots using a non-interactive backend."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from numcompute_stream.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


def test_metric_plot_preserves_chunk_positions_and_saves(tmp_path):
    path = tmp_path / "accuracy.png"
    fig = plot_metric_over_time([None, np.nan, 0.75], "Accuracy over time", "Accuracy", save_path=path)
    try:
        ax = fig.axes[0]
        assert ax.get_title() == "Accuracy over time"
        assert ax.get_ylabel() == "Accuracy"
        assert ax.get_xlim() == (0.5, 3.5)
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2, 3])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [np.nan, np.nan, 0.75])
        assert path.read_bytes().startswith(b"\x89PNG")
    finally:
        plt.close(fig)


def test_model_comparison_uses_the_same_chunks_and_labels():
    fig = compare_models([None, 0.7, None], [None, 0.8, None], ["Buffer 100", "Buffer 500"])
    try:
        ax = fig.axes[0]
        assert ax.get_legend_handles_labels()[1] == ["Buffer 100", "Buffer 500"]
        assert ax.get_xlim() == (0.5, 3.5)
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), ax.lines[1].get_xdata())
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [np.nan, 0.8, np.nan])
    finally:
        plt.close(fig)


def test_predictions_use_a_shared_axis_for_string_classes():
    fig = plot_predictions_vs_ground_truth(["cat", "dog"], ["cat", "bird"])
    try:
        ax = fig.axes[0]
        assert [tick.get_text() for tick in ax.get_yticklabels()] == ["cat", "dog", "bird"]
        np.testing.assert_array_equal(ax.collections[0].get_offsets(), [[1, 0], [2, 1]])
        np.testing.assert_array_equal(ax.collections[1].get_offsets(), [[1, 0], [2, 2]])
    finally:
        plt.close(fig)


def test_show_is_opt_in(monkeypatch):
    calls = []
    monkeypatch.setattr(plt, "show", lambda: calls.append(True))
    hidden = plot_metric_over_time([1], "Accuracy", "Accuracy")
    shown = plot_metric_over_time([1], "Accuracy", "Accuracy", show=True)
    try:
        assert calls == [True]
    finally:
        plt.close(hidden)
        plt.close(shown)


def test_plots_reject_invalid_inputs_before_creating_figures():
    before = plt.get_fignums()
    for values in ([], [[1]], [np.inf], ["0.5"], [1 + 2j]):
        with pytest.raises(ValueError):
            plot_metric_over_time(values, "Accuracy", "Accuracy")

    with pytest.raises(ValueError, match="same chunks"):
        compare_models([0.1], [0.1, 0.2], ["A", "B"])
    with pytest.raises(ValueError, match="model names"):
        compare_models([0.1], [0.1], ["A"])
    with pytest.raises(ValueError, match="matching"):
        plot_predictions_vs_ground_truth([0, 1], [0])
    with pytest.raises(ValueError, match="finite"):
        plot_predictions_vs_ground_truth([np.nan], [0])
    assert plt.get_fignums() == before
