import numpy as np
import pytest

from numcompute.optim import grad, jacobian

def test_central_diff():
    def f(x):
        return x[0] ** 2 + x[1]

    x = np.array([2.0, 4.0])
    result = grad(f, x, method='central')

    assert np.isfinite(result).all() 

def test_forward_diff():
    def f(x):
        return x[0] ** 2 + x[1]

    x = np.array([2.0, 4.0])
    result = grad(f, x, method='forward')

    assert np.isfinite(result).all() 


def test_jacobian_central():
    def F(x):
        return np.array([
            x[0]**2 + x[1],
            x[0] + x[1]**2
        ])

    x = np.array([4.0, 6,0])
    result = jacobian(F, x, method='central')

    assert np.isfinite(result).all() 

def test_jacobian_forward():
    def F(x):
        return np.array([
            x[0]**2 + x[1],
            x[0] + x[1]**2
        ])

    x = np.array([4.0, 6,0])
    result = jacobian(F, x, method='forward')

    assert np.isfinite(result).all() 