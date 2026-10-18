"""
Unit tests for preprocessing transformers (numcompute/preprocessing.py).
"""

import numpy as np
import pytest

from numcompute.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    SimpleImputer,
    OneHotEncoder,
)


# ==============================================================================
# 1. StandardScaler Tests
# ==============================================================================

def test_standard_scaler_basic():
    """Test standard scaling to zero mean and unit variance."""
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Check fitted parameters
    assert np.allclose(scaler.mean_, [2.0, 20.0])
    assert np.allclose(scaler.scale_, np.std(X, axis=0))

    # Check transformed properties: mean ~ 0, std ~ 1
    assert np.allclose(np.mean(X_scaled, axis=0), [0.0, 0.0])
    assert np.allclose(np.std(X_scaled, axis=0), [1.0, 1.0])


def test_standard_scaler_constant_feature():
    """Edge Case: Zero variance (constant column) defense to avoid 0-division."""
    # Col 0 is standard, Col 1 has all equal values (5.0)
    X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Constant column should not produce NaNs or Infs
    assert not np.isnan(X_scaled).any()
    assert not np.isinf(X_scaled).any()
    assert np.allclose(X_scaled[:, 1], 0.0)


def test_standard_scaler_non_contiguous_strides():
    """Edge Case: Ensure scaler handles non-contiguous memory slices (e.g. arr[::-1])."""
    base = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    reversed_arr = base[::-1]  # non-contiguous view

    scaler = StandardScaler()
    out = scaler.fit_transform(reversed_arr)
    assert out.shape == (3, 2)
    assert np.allclose(np.mean(out, axis=0), 0.0)


def test_standard_scaler_errors():
    """Test exceptions for unfitted transform, non-2D arrays, and empty arrays."""
    scaler = StandardScaler()

    # Calling transform before fit
    with pytest.raises(RuntimeError, match="not fitted yet"):
        scaler.transform(np.array([[1.0, 2.0]]))

    # Non-2D input
    with pytest.raises(ValueError, match="Expected 2D array"):
        scaler.fit(np.array([1.0, 2.0, 3.0]))

    # Empty array
    with pytest.raises(ValueError, match="Cannot fit on empty array"):
        scaler.fit(np.empty((0, 2)))


# ==============================================================================
# 2. MinMaxScaler Tests
# ==============================================================================

def test_min_max_scaler_basic():
    """Test scaling features to default [0, 1] and custom [-1, 1] range."""
    X = np.array([[10.0, -10.0], [20.0, 0.0], [30.0, 10.0]])

    # 1. Default range [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(np.min(X_scaled, axis=0), [0.0, 0.0])
    assert np.allclose(np.max(X_scaled, axis=0), [1.0, 1.0])

    # 2. Custom range [-1, 1]
    custom_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_custom = custom_scaler.fit_transform(X)
    assert np.allclose(np.min(X_custom, axis=0), [-1.0, -1.0])
    assert np.allclose(np.max(X_custom, axis=0), [1.0, 1.0])


def test_min_max_scaler_constant_feature():
    """Edge Case: Feature with max == min (range 0) should not divide by zero."""
    X = np.array([[1.0, 7.0], [5.0, 7.0]])
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Constant column should be mapped to the min range value (0.0) without NaN
    assert not np.isnan(X_scaled).any()
    assert np.allclose(X_scaled[:, 1], 0.0)


def test_min_max_scaler_invalid_range():
    """Test ValueError when feature_range min >= max."""
    with pytest.raises(ValueError, match="strictly smaller"):
        MinMaxScaler(feature_range=(1.0, 0.0))


# ==============================================================================
# 3. SimpleImputer Tests
# ==============================================================================

def test_simple_imputer_strategies():
    """Test mean, median, and constant imputation strategies."""
    # Data with NaNs: Col 0 mean is 2.0, median is 2.0. Col 1 mean is 20.0, median is 20.0
    X = np.array([[1.0, 10.0], [np.nan, 20.0], [3.0, np.nan]])

    # Mean strategy
    imp_mean = SimpleImputer(strategy="mean")
    X_mean = imp_mean.fit_transform(X)
    assert not np.isnan(X_mean).any()
    assert np.allclose(X_mean[1, 0], 2.0)
    assert np.allclose(X_mean[2, 1], 15.0)

    # Median strategy
    X_med_in = np.array([[1.0], [100.0], [np.nan], [2.0]])  # values: 1, 2, 100 -> median is 2.0
    imp_median = SimpleImputer(strategy="median")
    X_med_out = imp_median.fit_transform(X_med_in)
    assert np.allclose(X_med_out[2, 0], 2.0)

    # Constant strategy
    imp_const = SimpleImputer(strategy="constant", fill_value=-99.0)
    X_const = imp_const.fit_transform(X)
    assert X_const[1, 0] == -99.0
    assert X_const[2, 1] == -99.0


def test_simple_imputer_all_nan_column():
    """Edge Case: An entire column of NaNs should be filled with 0.0 without breaking."""
    X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    assert not np.isnan(X_imputed).any()
    assert np.allclose(X_imputed[:, 1], 0.0)


def test_simple_imputer_invalid_args():
    """Test invalid strategy and missing fill_value."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        SimpleImputer(strategy="invalid_mode")

    with pytest.raises(ValueError, match="fill_value must be provided"):
        SimpleImputer(strategy="constant", fill_value=None)


# ==============================================================================
# 4. OneHotEncoder Tests
# ==============================================================================

def test_one_hot_encoder_basic():
    """Test one-hot encoding on integers and string categories."""
    # Column 0: 2 classes (0, 1); Column 1: 3 classes (10, 20, 30) -> Total 5 columns
    X = np.array([[0, 10], [1, 20], [0, 30], [1, 10]])
    enc = OneHotEncoder(drop_first=False)
    X_enc = enc.fit_transform(X)

    assert X_enc.shape == (4, 5)
    # Row 0: class 0 (col 0: 1, col 1: 0) and class 10 (col 2: 1, col 3: 0, col 4: 0)
    assert np.allclose(X_enc[0], [1.0, 0.0, 1.0, 0.0, 0.0])


def test_one_hot_encoder_drop_first():
    """Test drop_first=True to avoid dummy variable trap."""
    # Col 0: 2 classes -> drops 1, keeps 1. Col 1: 3 classes -> drops 1, keeps 2 -> Total 3 columns
    X = np.array([[0, 10], [1, 20], [0, 30]])
    enc = OneHotEncoder(drop_first=True)
    X_enc = enc.fit_transform(X)

    assert X_enc.shape == (3, 3)


def test_one_hot_encoder_mismatched_features():
    """Test ValueError when transform has different number of features than fit."""
    X_train = np.array([[1, 2], [3, 4]])
    X_test = np.array([[1], [2]])  # Only 1 feature

    enc = OneHotEncoder()
    enc.fit(X_train)
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        enc.transform(X_test)