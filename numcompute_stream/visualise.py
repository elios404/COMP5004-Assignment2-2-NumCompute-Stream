"""Matplotlib plots for streaming metrics and predictions."""

from __future__ import annotations

from os import PathLike

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np


def _metric_series(values) -> np.ndarray:
    # None in a warm-up log becomes NaN and leaves a gap in the line.
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Expected a non-empty 1D metric series.")
    if values.dtype.kind not in "biufO":
        raise ValueError("Metric values must be real numbers or None.")
    if values.dtype.kind == "O" and any(
        value is not None and not isinstance(value, (int, float, np.integer, np.floating, np.bool_))
        for value in values
    ):
        raise ValueError("Metric values must be real numbers or None.")

    with np.errstate(over="ignore"):
        values = values.astype(float)
    if np.isinf(values).any():
        raise ValueError("Metric values must be finite or missing.")
    return values


def _finish(fig: Figure, save_path: str | PathLike[str] | None, show: bool) -> Figure:
    try:
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
    except Exception:
        plt.close(fig)
        raise

    return fig


def plot_metric_over_time(
    metric_values, title: str, ylabel: str, *, save_path=None, show: bool = False
) -> Figure:
    """Plot one value per chunk, numbered from one.

    None and NaN leave gaps. Return a figure for inline display or further
    editing. Pass save_path to save a copy, or show=True to display it.
    """
    values = _metric_series(metric_values)
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, values.size + 1), values, marker="o")
    ax.set_xlim(0.5, values.size + 0.5)
    ax.set(title=title, xlabel="Chunk", ylabel=ylabel)
    ax.xaxis.get_major_locator().set_params(integer=True)
    return _finish(fig, save_path, show)


def compare_models(
    metric1, metric2, labels, *, title: str = "Model comparison",
    ylabel: str = "Metric", save_path=None, show: bool = False
) -> Figure:
    """Plot two series aligned to the same chunks, with two legend labels."""
    first = _metric_series(metric1)
    second = _metric_series(metric2)
    if first.shape != second.shape:
        raise ValueError("Model metrics must have one value for each of the same chunks.")
    if not isinstance(labels, (list, tuple, np.ndarray)) or np.ndim(labels) != 1:
        raise ValueError("labels must contain two model names.")
    if len(labels) != 2 or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("labels must contain two non-empty model names.")

    chunks = np.arange(1, first.size + 1)
    fig, ax = plt.subplots()
    ax.plot(chunks, first, marker="o", label=labels[0])
    ax.plot(chunks, second, marker="s", linestyle="--", label=labels[1])
    ax.set_xlim(0.5, chunks.size + 0.5)
    ax.set(title=title, xlabel="Chunk", ylabel=ylabel)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend()

    return _finish(fig, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, *, save_path=None, show: bool = False) -> Figure:
    """Compare aligned (N,) class labels for a scored chunk.

    The class axis is shared by the two sets of labels. Hollow markers make
    overlapping actual and predicted values visible.
    """
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if actual.ndim != 1 or actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("Expected matching non-empty 1D label arrays.")
    for labels in (actual, predicted):
        if labels.dtype.kind not in "biufUS" or (
            labels.dtype.kind == "f" and not np.isfinite(labels).all()
        ):
            raise ValueError("Labels must contain finite numeric or string classes.")

    actual = actual.astype(object)
    predicted = predicted.astype(object)
    classes = list(dict.fromkeys([*actual, *predicted]))
    indices = {label: index for index, label in enumerate(classes)}
    rows = np.arange(1, actual.size + 1)

    fig, ax = plt.subplots()
    ax.scatter(rows, [indices[label] for label in actual], s=55, facecolors="none",
               edgecolors="tab:blue", label="Actual")
    ax.scatter(rows, [indices[label] for label in predicted], s=24, marker="x",
               color="tab:orange", label="Predicted")
    ax.set_yticks(range(len(classes)), [str(label) for label in classes])
    ax.set(title="Predictions vs ground truth", xlabel="Sample in chunk", ylabel="Class")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend()
    
    return _finish(fig, save_path, show)
