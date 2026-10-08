"""
Micro-benchmarking harness comparing NumPy vectorisation vs pure Python loops.
"""

from __future__ import annotations
from typing import Any, Callable, Dict


def benchmark(
    func: Callable[..., Any],
    *args: Any,
    repeat: int = 5,
    number: int = 10,
    **kwargs: Any,
) -> float:
    """Measure average execution time of func in seconds."""
    # TODO: Measure runtime using time.perf_counter
    raise NotImplementedError


def compare_implementations(
    loop_fn: Callable[..., Any],
    vec_fn: Callable[..., Any],
    *args: Any,
    repeat: int = 5,
) -> Dict[str, float]:
    """Compare execution times between loop and vectorized implementations."""
    # TODO: Run benchmarks and return dictionary with runtimes and speedup factor
    raise NotImplementedError