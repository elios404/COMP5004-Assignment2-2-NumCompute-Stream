import numpy as np
import pytest

from numcompute.optim import grad, jacobian

def test_central_diff():
    def f(x):
        return x[0] ** 2 + x[1]

    x = np.array([2.0, 4.0])
    result = grad(f, x, method='central')

    np.testing.assert_allclose(result, [2.0, 4.0], atol=1e-4) 

def test_forward_diff():
    def f(x):
        return x[0] ** 2 + x[1]

    x = np.array([2.0, 4.0])
    result = grad(f, x, method='forward')

    np.testing.assert_allclose(result, [2.0, 4.0], atol=1e-4) 


def test_jacobian_central():
    def F(x):
        return np.array([
            x[0]**2 + x[1],
            x[0] + x[1]**2
        ])

    x = np.array([4.0, 6,0])
    result = jacobian(F, x, method='central')

    np.testing.assert_allclose(result, [4.0, 6.0], atol=1e-4) 

def test_jacobian_forward():
    def F(x):
        return np.array([
            x[0]**2 + x[1],
            x[0] + x[1]**2
        ])

    x = np.array([4.0, 6.0])
    result = jacobian(F, x, method='forward')

    np.testing.assert_allclose(result, [4.0, 6.0], atol=1e-4)


import numpy as np
import pytest

from numcompute.optim import grad, jacobian




def test_central_diff_linear_function():
    def f(x):
        return 2 * x[0] + 3 * x[1]

    result = grad(f, np.array([2.0, 4.0]), method="central")
    np.testing.assert_allclose(result, [2.0, 3.0], atol=1e-4)


def test_forward_diff_linear_function():
    def f(x):
        return 2 * x[0] + 3 * x[1]

    result = grad(f, np.array([2.0, 4.0]), method="forward")
    np.testing.assert_allclose(result, [2.0, 3.0], atol=1e-4)


def test_central_diff_quadratic_function():
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    result = grad(f, np.array([3.0, 4.0]), method="central")
    np.testing.assert_allclose(result, [6.0, 8.0], atol=1e-4)


def test_forward_diff_quadratic_function():
    def f(x):
        return x[0] ** 2 + x[1] ** 2

    result = grad(f, np.array([3.0, 4.0]), method="forward")
    np.testing.assert_allclose(result, [6.0, 8.0], atol=1e-3)


def test_central_diff_constant_function():
    def f(x):
        return 7.0

    result = grad(f, np.array([1.0, 2.0]), method="central")
    np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-8)


def test_forward_diff_constant_function():
    def f(x):
        return 7.0

    result = grad(f, np.array([1.0, 2.0]), method="forward")
    np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-8)


def test_central_diff_single_variable():
    def f(x):
        return x[0] ** 2

    result = grad(f, np.array([5.0]), method="central")
    np.testing.assert_allclose(result, [10.0], atol=1e-4)


def test_forward_diff_single_variable():
    def f(x):
        return x[0] ** 2

    result = grad(f, np.array([5.0]), method="forward")
    np.testing.assert_allclose(result, [10.0], atol=1e-3)


def test_central_diff_three_variables():
    def f(x):
        return x[0] ** 2 + 2 * x[1] ** 2 + 3 * x[2] ** 2

    result = grad(f, np.array([1.0, 2.0, 3.0]), method="central")
    np.testing.assert_allclose(result, [2.0, 8.0, 18.0], atol=1e-4)


def test_forward_diff_three_variables():
    def f(x):
        return x[0] ** 2 + 2 * x[1] ** 2 + 3 * x[2] ** 2

    result = grad(f, np.array([1.0, 2.0, 3.0]), method="forward")
    np.testing.assert_allclose(result, [2.0, 8.0, 18.0], atol=1e-3)


def test_central_diff_result_is_finite():
    def f(x):
        return x[0] ** 2 + x[1]

    result = grad(f, np.array([2.0, 4.0]), method="central")
    assert np.isfinite(result).all()


def test_forward_diff_result_is_finite():
    def f(x):
        return x[0] ** 2 + x[1]

    result = grad(f, np.array([2.0, 4.0]), method="forward")
    assert np.isfinite(result).all()


def test_grad_invalid_method():
    def f(x):
        return x[0] ** 2

    with pytest.raises(ValueError):
        grad(f, np.array([2.0]), method="invalid")


# =========================
# Jacobian tests (11 cases)
# =========================

def test_jacobian_central():
    def F(x):
        return np.array([
            x[0] ** 2 + x[1],
            x[0] + x[1] ** 2
        ])

    result = jacobian(F, np.array([3.0, 4.0]), method="central")
    expected = np.array([[6.0, 1.0], [1.0, 8.0]])

    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_jacobian_forward():
    def F(x):
        return np.array([
            x[0] ** 2 + x[1],
            x[0] + x[1] ** 2
        ])

    result = jacobian(F, np.array([3.0, 4.0]), method="forward")
    expected = np.array([[6.0, 1.0], [1.0, 8.0]])

    np.testing.assert_allclose(result, expected, atol=1e-3)


def test_jacobian_linear_function():
    def F(x):
        return np.array([
            2 * x[0] + 3 * x[1],
            4 * x[0] - x[1]
        ])

    result = jacobian(F, np.array([1.0, 2.0]), method="central")
    expected = np.array([[2.0, 3.0], [4.0, -1.0]])

    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_jacobian_constant_function():
    def F(x):
        return np.array([5.0, -2.0])

    result = jacobian(F, np.array([1.0, 2.0]), method="central")
    np.testing.assert_allclose(result, np.zeros((2, 2)), atol=1e-8)


def test_jacobian_single_input():
    def F(x):
        return np.array([x[0] ** 2, 3 * x[0]])

    result = jacobian(F, np.array([2.0]), method="central")
    np.testing.assert_allclose(result, [[4.0], [3.0]], atol=1e-4)


def test_jacobian_single_output():
    def F(x):
        return np.array([x[0] ** 2 + x[1] ** 2])

    result = jacobian(F, np.array([3.0, 4.0]), method="central")
    np.testing.assert_allclose(result, [[6.0, 8.0]], atol=1e-4)


def test_jacobian_output_shape():
    def F(x):
        return np.array([
            x[0] + x[1],
            x[0] - x[1],
            x[0] * x[1]
        ])

    result = jacobian(F, np.array([2.0, 3.0]))
    assert result.shape == (3, 2)


def test_jacobian_result_is_finite():
    def F(x):
        return np.array([x[0] ** 2, x[1] ** 2])

    result = jacobian(F, np.array([2.0, 3.0]))
    assert np.isfinite(result).all()


def test_jacobian_invalid_method():
    def F(x):
        return np.array([x[0] ** 2])

    with pytest.raises(ValueError):
        jacobian(F, np.array([2.0]), method="invalid")


def test_jacobian_linear_function_forward():
    def F(x):
        return np.array([
            2 * x[0] + 3 * x[1],
            4 * x[0] - x[1]
        ])

    result = jacobian(F, np.array([1.0, 2.0]), method="forward")
    expected = np.array([[2.0, 3.0], [4.0, -1.0]])

    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_jacobian_three_inputs():
    def F(x):
        return np.array([
            x[0] + 2 * x[1] + 3 * x[2],
            x[0] * x[1] + x[2]
        ])

    result = jacobian(F, np.array([1.0, 2.0, 3.0]))
    expected = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 1.0]
    ])

    np.testing.assert_allclose(result, expected, atol=1e-4)


def test_jacobian_two_outputs_two_inputs():
    def F(x):
        return np.array([x[0] + x[1], x[0] - x[1]])

    result = jacobian(F, np.array([5.0, 2.0]))
    expected = np.array([[1.0, 1.0], [1.0, -1.0]])

    np.testing.assert_allclose(result, expected, atol=1e-4)
