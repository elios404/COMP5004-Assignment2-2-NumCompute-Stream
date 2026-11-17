"""Measure Gini split search and buffered retraining on a fixed stream."""

from pathlib import Path
from statistics import mean, median
import csv
import platform

import numpy as np

from numcompute.benchmarking import compare_implementations
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier, _best_split


SEED = 42
N_ROWS = 1280
N_FEATURES = 6
CHUNK_SIZE = 40
BUFFER_SIZES = (80, 160, 320)
REPEATS = 5
MAX_DEPTH = 4
GAIN_TOLERANCE = 8 * np.finfo(float).eps


def loop_best_split(X, y):
    """Reference Gini search for this benchmark's finite, binary-labelled data."""
    n_rows = len(y)
    total_positive = int(y.sum())
    positive_rate = total_positive / n_rows
    parent_gini = 2 * positive_rate * (1 - positive_rate)
    candidates = []
    best_gain = GAIN_TOLERANCE

    for feature, column in enumerate(X.T):
        order = np.argsort(column, kind="stable")
        values = column[order]
        labels = y[order].tolist()
        left_positive = 0
        for index, (lower, upper) in enumerate(zip(values[:-1], values[1:])):
            left_positive += labels[index]
            if lower == upper:
                continue

            n_left = index + 1
            n_right = n_rows - n_left
            p_left = left_positive / n_left
            p_right = (total_positive - left_positive) / n_right
            left_gini = 2 * p_left * (1 - p_left)
            right_gini = 2 * p_right * (1 - p_right)
            gain = parent_gini - (n_left * left_gini + n_right * right_gini) / n_rows

            if gain > GAIN_TOLERANCE:
                candidates.append((feature, lower / 2 + upper / 2, gain))
                best_gain = max(best_gain, gain)

    return next((split for split in candidates if split[2] >= best_gain - GAIN_TOLERANCE), None)


def split_timings(X, y):
    results = []
    for size in BUFFER_SIZES:
        windows = [(X[i:i + size], y[i:i + size]) for i in range(0, len(y), size)]
        timings = []
        for window_X, window_y in windows:
            reference = loop_best_split(window_X, window_y)
            vectorised = _best_split(window_X, window_y)
            if reference is None or vectorised is None:
                if reference is not vectorised:
                    raise AssertionError("Only one implementation found a split.")
            else:
                if reference[:2] != vectorised[:2]:
                    raise AssertionError("Split search implementations chose different splits.")
                np.testing.assert_allclose(
                    reference[2], vectorised[2], rtol=0, atol=GAIN_TOLERANCE,
                )

            timings.append(compare_implementations(
                loop_best_split, _best_split, window_X, window_y, repeat=REPEATS,
            ))

        loop_ms = 1000 * mean(run["loop_time"] for run in timings)
        vectorised_ms = 1000 * mean(run["vec_time"] for run in timings)
        results.append({
            "window_rows": size, "windows": len(windows),
            "loop_ms": round(loop_ms, 6), "vectorised_ms": round(vectorised_ms, 6),
            "speedup": round(loop_ms / vectorised_ms, 3),
        })
    return results


def run_stream(X, y, buffer_size):
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("tree", DecisionTreeClassifier(max_depth=MAX_DEPTH, buffer_size=buffer_size)),
    ])
    trainer = StreamTrainer(pipeline, labels=(0, 1))
    for start in range(0, len(y), CHUNK_SIZE):
        stop = start + CHUNK_SIZE
        trainer.fit_chunk(X[start:stop], y[start:stop])
    return trainer.logs


def buffer_timings(X, y):
    results = []
    # Interleave buffer sizes so they run under similar timing conditions.
    trials = [{size: run_stream(X, y, size) for size in BUFFER_SIZES} for _ in range(REPEATS)]
    first_trial = trials[0]
    shared_chunks = [
        index for index, entries in enumerate(zip(*first_trial.values()))
        if all(entry["evaluated"] for entry in entries)
    ]
    for size in BUFFER_SIZES:
        logs = first_trial[size]
        scored_rows = sum(logs[i]["n_samples"] for i in shared_chunks)
        correct = sum(round(logs[i]["accuracy"] * logs[i]["n_samples"]) for i in shared_chunks)
        refit_ms = [
            1000 * entry["fit_seconds"] for trial in trials for entry in trial[size]
            if entry["retrained"]
        ]
        results.append({
            "buffer_rows": size,
            "refits_per_run": sum(entry["retrained"] for entry in logs),
            "first_scored_chunk": next(entry["chunk"] for entry in logs if entry["evaluated"]),
            "shared_scored_rows": scored_rows,
            "shared_accuracy": round(correct / scored_rows, 6),
            "median_refit_update_ms": round(median(refit_ms), 6),
        })
    return results


def write_results(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(path.name)
    for row in rows:
        print(row)


def main():
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(N_ROWS, N_FEATURES))
    boundary = np.where(np.arange(N_ROWS) < N_ROWS // 2, -0.6, 0.6)
    y = (X[:, 0] + 0.6 * X[:, 1] > boundary).astype(int)

    output = Path(__file__).resolve().parent
    write_results(output / "stream_split_results.csv", split_timings(X, y))
    write_results(output / "stream_buffer_results.csv", buffer_timings(X, y))
    (output / "stream_environment.txt").write_text(
        f"Platform: {platform.platform()}\nPython: {platform.python_version()}\n"
        f"NumPy: {np.__version__}\nSeed: {SEED}\nRows: {N_ROWS}\nFeatures: {N_FEATURES}\n"
        f"Chunk rows: {CHUNK_SIZE}\nBuffer rows: {BUFFER_SIZES}\nTree max depth: {MAX_DEPTH}\n"
        f"Timing repetitions: {REPEATS}\nSplit calls per repetition per window: 10\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
