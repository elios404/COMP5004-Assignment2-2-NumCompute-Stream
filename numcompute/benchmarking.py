"""
Micro-benchmarking harness comparing NumPy vectorisation vs pure Python loops.
"""

from __future__ import annotations
from typing import Any, Callable, Dict
import time



def benchmark(
    func: Callable[..., Any],
    *args: Any,
    repeat: int = 5,
    number: int = 10,
    **kwargs: Any,
) -> float:
    """
    Measure the average execution time of a function.

    Parameters
    ----------
    func : callable
        Function whose execution time is measured.
    *args : Any
        Positional arguments passed to the function.
    repeat : int
        Number of timing measurements. Default is 5.
    number : int
        Number of executions per measurement. Default is 10.
    **kwargs : Any
        Keyword arguments passed to the function.

    Returns
    -------
    float
        Average execution time per function call in seconds.

    Raises
    ------
    TypeError
        If func is not callable or repeat/number
        are not integers.
    ValueError
        If repeat or number is not positive.

    Complexity
    ----------
    Time: O(repeat * number * T)
        where T is the runtime of func.
    Extra space: O(1), excluding memory used by func.
    """
    if not callable(func):
        raise TypeError("func must be callable.")

    for name, value in (("repeat", repeat), ("number", number)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")

        if value <= 0:
            raise ValueError(f"{name} must be positive.")

    total_time = 0.0

    for _ in range(repeat):
        start = time.perf_counter()

        for _ in range(number):
            func(*args, **kwargs)

        end = time.perf_counter()

        total_time += end - start

    average_time = total_time / (repeat * number)

    return float(average_time)




def compare_implementations(
    loop_fn: Callable[..., Any],
    vec_fn: Callable[..., Any],
    *args: Any,
    repeat: int = 5,
) -> Dict[str, float]:
    """
    Compare Python loop and NumPy vectorized implementations.

    Parameters
    ----------
    loop_fn : callable
        Function using a Python loop.
    vec_fn : callable
        Function using vectorized operations.
    *args : Any
        Positional arguments passed to both functions.
    repeat : int
        Number of timing repetitions. Default is 5.

    Returns
    -------
    dict
        Dictionary containing:
        - loop_time: Average loop execution time in seconds.
        - vec_time: Average vectorized execution time in seconds.
        - speedup: loop_time divided by vec_time.

    Raises
    ------
    TypeError
        If either function is not callable or repeat
        is not an integer.
    ValueError
        If repeat is not positive.

    Complexity
    ----------
    Time: O(repeat * 10 * (T_loop + T_vec))
    where T_loop and T_vec are the execution times
    of the respective functions.
    Extra space: O(1), excluding function execution.
    """
    if not callable(loop_fn) or not callable(vec_fn):
        raise TypeError("Both implementations must be callable.")

    if isinstance(repeat, bool) or not isinstance(repeat, int):
        raise TypeError("repeat must be an integer.")

    if repeat <= 0:
        raise ValueError("repeat must be positive.")

    # Measure the average time for the Python loop
    loop_time = benchmark(
        loop_fn, *args, repeat=repeat, number=10
    )

    # Measure the average time for the vectorized function
    vec_time = benchmark(
        vec_fn, *args, repeat=repeat, number=10
    )

    # Calculate speedup, safely handling zero timing
    if vec_time == 0:
        speedup = float("inf") if loop_time > 0 else 1.0
    else:
        speedup = loop_time / vec_time

    return {
        "loop_time": float(loop_time),
        "vec_time": float(vec_time),
        "speedup": float(speedup)
    }
