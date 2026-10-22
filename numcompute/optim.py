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
    """
    Estimate the gradient of a scalar-valued function using
    finite differences.

    Parameters
    ----------
    f : callable
        Function that accepts an input array of shape (D,)
        and returns a scalar.
    x : array_like
        Input point with shape (D,), where D is the number
        of input variables.
    h : float, optional
        Positive perturbation step size. Default is 1e-5.
    method : {"central", "forward"}, optional
        Finite-difference method. Default is "central".

    Returns
    -------
    ndarray
        Estimated gradient with shape (D,).

    Raises
    ------
    ValueError
        If method is not "central" or "forward".

    Complexity
    ----------
    Time: O(D * C_f), where C_f is the cost of one evaluation
    of f. Central differences require approximately 2D function
    evaluations; forward differences require D + 1.
    Auxiliary space: O(D), excluding memory used by f.

    Notes
    -----
    Central differences estimate derivatives using points on both
    sides of each coordinate. Forward differences use the current
    point and a positively perturbed point.
    """
    arr = np.asarray(x, dtype=float)
    gradient = np.zeros(arr.size)

    if method not in ("central", "forward"):
        raise ValueError("the method has to be either Central or Forward")

    calced_value = f(arr)

    for i in range(arr.size):
        x_plus = arr.copy()
        x_plus[i] += h

        if method == 'central':
            x_minus = arr.copy()
            x_minus[i] -= h
            gradient[i] = (f(x_plus) - f(x_minus)) / (2 * h)
        else:
            gradient[i] = (f(x_plus) - calced_value) / h

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

    """
    Estimate the Jacobian matrix of a vector-valued function
    using finite differences.

    Parameters
    ----------
    F : callable
        Function that accepts an input array of shape (D,)
        and returns an array containing M output values.
    x : array_like
        Input point with shape (D,), where D is the number
        of input variables.
    h : float, optional
        Positive perturbation step size. Default is 1e-5.
    method : {"central", "forward"}, optional
        Finite-difference method. Default is "central".

    Returns
    -------
    ndarray
        Jacobian matrix with shape (M, D), where M is the
        number of outputs and D is the number of inputs.

    Raises
    ------
    ValueError
        If method is not "central" or "forward".

    Complexity
    ----------
    Time: O(D * C_F + M * D), where C_F is the cost of one
    evaluation of F. Central differences require approximately
    2D function evaluations; forward differences require
    D + 1 evaluations.
    Auxiliary space: O(M * D + D + M), including the output
    matrix and working arrays, excluding memory used by F.
    """
    output = F(x)
    arr = np.asarray(x, dtype=float)
    jacobian_matrix = np.zeros((output.size, arr.size))
    
    if method not in ("central", "forward"):
        raise ValueError("the method has to be either Central or Forward")
    
    for i in range(arr.size):
        x_plus = arr.copy()
        x_plus[i] += h
    
        if method == 'central':
            x_minus = arr.copy()
            x_minus[i] -= h
            jacobian_matrix[:, i] = (F(x_plus) - F(x_minus)) / (2 * h)
        else:
            jacobian_matrix[:, i] = (F(x_plus) - output) / h
    
    return jacobian_matrix
    