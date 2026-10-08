"""
Numerical gradient and Jacobian estimation using finite differences.
"""

from __future__ import annotations
from typing import Callable
import numpy as np


def grad(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float = 1e-5,
    method: str = "central",
) -> np.ndarray:
    """
    Estimate gradient of a scalar function f at vector point x.
    Supported methods: 'central', 'forward'.
    """
    # TODO: Finite-difference approximation of gradient vector
    raise NotImplementedError


def jacobian(
    F: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    h: float = 1e-5,
    method: str = "central",
) -> np.ndarray:
    """
    Estimate Jacobian matrix of vector-valued function F at point x.
    Output shape: (output_dim, input_dim)
    """
    # TODO: Finite-difference approximation of Jacobian matrix
    raise NotImplementedError