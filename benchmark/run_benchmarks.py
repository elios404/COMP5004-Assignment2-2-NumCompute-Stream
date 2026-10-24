
"""
Reproducible benchmark: Python loops versus NumPy MSE.
"""

import csv
import platform
from pathlib import Path

import numpy as np
import math
from numcompute.utils import euclidean_distance, softmax
from numcompute.metrics import mse
from numcompute.benchmarking import compare_implementations


def mse_loop(y_true, y_pred):
    """Calculate MSE using an ordinary Python loop."""
    total = 0.0

    for actual, predicted in zip(y_true, y_pred):
        difference = float(actual) - float(predicted)
        total += difference ** 2

    return total / len(y_true)


def euclidean_distance_loop(X, Y):
    """
    Calculate pairwise Euclidean distances using Python loops.

    Parameters
    ----------
    X : np.ndarray
        First array, shape (n_samples_X, n_features).
    Y : np.ndarray
        Second array, shape (n_samples_Y, n_features).

    Returns
    -------
    np.ndarray
        Pairwise distance matrix of shape
        (n_samples_X, n_samples_Y).

    Complexity
    ----------
    Time: O(n * m * d)
    Space: O(n * m)
    """
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]
    n_features = X.shape[1]

    distances = np.zeros((n_samples_X, n_samples_Y))

    for i in range(n_samples_X):
        for j in range(n_samples_Y):
            distance = 0.0

            for k in range(n_features):
                difference = float(X[i, k]) - float(Y[j, k])
                distance = math.hypot(distance, difference)

            distances[i, j] = distance

    return distances


def softmax_loop(x):
    """
    Calculate numerically stable softmax using Python loops.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional array of numerical scores.

    Returns
    -------
    np.ndarray
        Softmax probabilities with the same shape as x.

    Complexity
    ----------
    Time: O(n)
    Space: O(n)
    """
    # Find maximum value for numerical stability
    max_value = float(max(x))

    exp_values = []
    total = 0.0

    # Calculate exponentials using a Python loop
    for value in x:
        exp_value = math.exp(float(value) - max_value)
        exp_values.append(exp_value)
        total += exp_value

    # Normalize the values to obtain probabilities
    probabilities = []

    for value in exp_values:
        probabilities.append(value / total)

    return np.array(probabilities)

def main():
    # Fixed seed makes input data reproducible
    rng = np.random.default_rng(42)

    environment = (
        f"Python: {platform.python_version()}\n"
        f"NumPy: {np.__version__}\n"
        f"Platform: {platform.platform()}\n"
        f"Machine: {platform.machine()}\n"
    )

    print(environment)
    print("Benchmark: Python loop vs NumPy MSE")
    print("Each implementation: repeat=5, number=10")
    print("-" * 65)
    print(f"{'Samples':>10} {'Loop (ms)':>12} {'NumPy (ms)':>12} {'Speedup':>12}")

    results = []

    for n in [1000, 10000, 100000]:
        # Create identical inputs for both implementations
        y_true = rng.normal(size=n)
        y_pred = y_true + rng.normal(scale=0.5, size=n)

        # Verify numerical correctness before timing
        loop_result = mse_loop(y_true, y_pred)
        numpy_result = mse(y_true, y_pred)

        if not np.isclose(loop_result, numpy_result, rtol=1e-10):
            raise ValueError("Implementations produced different results.")

        timings = compare_implementations(
            mse_loop, mse, y_true, y_pred, repeat=5
        )

        
        row = {
            "operation": "MSE",
            "samples": n,
            "loop_ms": timings["loop_time"] * 1000,
            "numpy_ms": timings["vec_time"] * 1000,
            "speedup": timings["speedup"]
        }


        results.append(row)

        print(
            f"{n:>10} "
            f"{row['loop_ms']:>12.4f} "
            f"{row['numpy_ms']:>12.4f} "
            f"{row['speedup']:>11.2f}x"
        )
    
    # Benchmark 2: Euclidean Distance
    print("\nBenchmark: Euclidean Distance (Loop vs NumPy)")
    print("-" * 65)
    print(
        f"{'Rows in X/Y':>12} {'Loop (ms)':>12} "
        f"{'NumPy (ms)':>12} {'Speedup':>12}"
    )

    for n in [20, 50, 100]:
        # Generate reproducible 2D arrays with 8 features
        X = rng.normal(size=(n, 8))
        Y = rng.normal(size=(n, 8))

        # Verify both implementations give the same results
        loop_result = euclidean_distance_loop(X, Y)
        numpy_result = euclidean_distance(X, Y)

        np.testing.assert_allclose(
            loop_result, numpy_result,
            rtol=1e-10, atol=1e-12
        )

        # Measure execution times
        timings = compare_implementations(
            euclidean_distance_loop,
            euclidean_distance,
            X, Y,
            repeat=5
        )

        row = {
            "operation": "Euclidean Distance",
            "samples": n,
            "loop_ms": timings["loop_time"] * 1000,
            "numpy_ms": timings["vec_time"] * 1000,
            "speedup": timings["speedup"]
        }

        results.append(row)

        print(
            f"{n:>12} "
            f"{row['loop_ms']:>12.4f} "
            f"{row['numpy_ms']:>12.4f} "
            f"{row['speedup']:>11.2f}x"
        )
    
    # Benchmark 3: Softmax
    print("\nBenchmark: Softmax (Loop vs NumPy)")
    print("-" * 65)
    print(
        f"{'Samples':>10} {'Loop (ms)':>12} "
        f"{'NumPy (ms)':>12} {'Speedup':>12}"
    )

    for n in [1000, 10000, 100000]:
        # Generate reproducible scores, including large values
        x = rng.normal(loc=1000.0, scale=2.0, size=n)

        # Verify that both implementations produce the same result
        loop_result = softmax_loop(x)
        numpy_result = softmax(x)

        np.testing.assert_allclose(
            loop_result, numpy_result,
            rtol=1e-10, atol=1e-12
        )

        # Benchmark loop-based vs vectorized implementations
        timings = compare_implementations(
            softmax_loop,
            softmax,
            x,
            repeat=5
        )

        row = {
            "operation": "Softmax",
            "samples": n,
            "loop_ms": timings["loop_time"] * 1000,
            "numpy_ms": timings["vec_time"] * 1000,
            "speedup": timings["speedup"]
        }

        results.append(row)

        print(
            f"{n:>10} "
            f"{row['loop_ms']:>12.4f} "
            f"{row['numpy_ms']:>12.4f} "
            f"{row['speedup']:>11.2f}x"
        )
    


    # Save results for the assignment report
    output_dir = Path(__file__).resolve().parent

    with (output_dir / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    (output_dir / "environment.txt").write_text(
        environment, encoding="utf-8"
    )

    print("\nResults saved in benchmark/results.csv")
    print("Environment saved in benchmark/environment.txt")


if __name__ == "__main__":
    main()
