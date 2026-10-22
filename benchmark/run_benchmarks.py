
"""
Reproducible benchmark: Python loops versus NumPy MSE.
"""

import csv
import platform
from pathlib import Path

import numpy as np

from numcompute.metrics import mse
from numcompute.benchmarking import compare_implementations


def mse_loop(y_true, y_pred):
    """Calculate MSE using an ordinary Python loop."""
    total = 0.0

    for actual, predicted in zip(y_true, y_pred):
        difference = float(actual) - float(predicted)
        total += difference ** 2

    return total / len(y_true)


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
