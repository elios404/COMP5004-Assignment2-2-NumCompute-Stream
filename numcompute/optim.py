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
    arr = np.asarray(x, dtype=float)
    gradient = np.zeros(arr.size)

    if method not in ("central", "forward"):
        raise ValueError("the method has to be either Central or Forward")

    for i in range(arr.size):
        x_plus = arr.copy()
        x_plus[i] += h

        if method == 'central':
            x_minus = arr.copy()
            x_minus[i] -= h
            gradient[i] = (f(x_plus) - f(x_minus)) / (2 * h)
        else:
            gradient[i] = (f(x_plus) - f(arr)) / h

    return gradient
    
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
    output = F(x)
    arr = np.asarray(x, dtype=float)
    jacobian_matrx = np.zeros((output.size, arr.size))
    
    if method not in ("central", "forward"):
        raise ValueError("the method has to be either Central or Forward")
    
    for i in range(arr.size):
        x_plus = arr.copy()
        x_plus[i] += h
    
        if method == 'central':
            x_minus = arr.copy()
            x_minus[i] -= h
            jacobian_matrx[:, i] = (F(x_plus) - F(x_minus)) / (2 * h)
        else:
            jacobian_matrx[:, i] = (F(x_plus) - output) / h
    
    return jacobian_matrx
    