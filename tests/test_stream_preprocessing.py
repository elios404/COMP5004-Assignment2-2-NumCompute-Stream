"""Tests for chunk-wise preprocessing."""

import numpy as np
import pytest

from numcompute_stream.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler


def test_standard_scaler_merges_uneven_chunks_with_missing_values():
    X = np.array([[0, np.nan], [2, 8], [10, np.nan], [20, 4]], dtype=float)
    scaler = StandardScaler().partial_fit(X[:3]).partial_fit(X[3:])

    np.testing.assert_array_equal(scaler.n_samples_seen_, [4, 2])
    np.testing.assert_allclose(scaler.mean_, np.nanmean(X, axis=0))
    np.testing.assert_allclose(scaler.var_, np.nanvar(X, axis=0))
    np.testing.assert_allclose(scaler.transform(X), (X - scaler.mean_) / scaler.scale_)


def test_standard_scaler_keeps_statistics_for_an_all_missing_chunk_column():
    scaler = StandardScaler(with_mean=False).partial_fit([[2, np.nan], [4, np.nan]])
    np.testing.assert_allclose(scaler.scale_, [1, 1])
    np.testing.assert_allclose(scaler.transform([[3, 12]]), [[3, 12]])

    scaler.partial_fit([[np.nan, 12], [np.nan, 16]])
    assert scaler.mean_ is None
    np.testing.assert_allclose(scaler.var_, [1, 4])
    np.testing.assert_allclose(scaler.transform([[3, 12]]), [[3, 6]])


def test_min_max_scaler_extends_extrema_and_handles_missing_and_constant_columns():
    scaler = MinMaxScaler(feature_range=(-1, 1)).partial_fit([[2, np.nan, 5], [4, np.nan, 5]])
    np.testing.assert_allclose(scaler.transform([[3, 20, 5]]), [[0, 20, -1]])

    scaler.partial_fit([[6, 20, np.nan], [np.nan, 30, np.nan]])
    np.testing.assert_allclose(scaler.data_min_, [2, 20, 5])
    np.testing.assert_allclose(scaler.data_max_, [6, 30, 5])
    np.testing.assert_allclose(scaler.transform([[8, 25, 5], [np.nan, 20, 5]]),[[2, 0, -1], [np.nan, -1, -1]])


def test_numeric_transformers_reject_bad_chunks_and_allow_refit():
    for transformer in (StandardScaler(), MinMaxScaler(), SimpleImputer()):
        with pytest.raises(RuntimeError, match="not fitted"):
            transformer.transform([[1]])
        transformer.partial_fit([[2, 4], [4, 6]])
        expected = transformer.transform([[3, 5]])
        with pytest.raises(ValueError, match="expected 2 features"):
            transformer.partial_fit([[9]])
        with pytest.raises(ValueError, match="expected 2 features"):
            transformer.transform([[9]])
        with pytest.raises(ValueError, match="empty"):
            transformer.partial_fit(np.empty((0, 2)))
        with pytest.raises(ValueError, match="finite or NaN"):
            transformer.fit([[np.inf, 2]])
        np.testing.assert_allclose(transformer.transform([[3, 5]]), expected)
        assert transformer.fit_transform([[10], [20]]).shape == (2, 1)
        assert transformer.transform(np.empty((0, 1))).shape == (0, 1)


def test_standard_scaler_handles_a_large_offset():
    X = (1e12 + np.arange(12.0)).reshape(6, 2)
    scaler = StandardScaler().partial_fit(X[:2]).partial_fit(X[2:])
    np.testing.assert_allclose(scaler.var_, np.var(X, axis=0), atol=1e-3)
    scaler.fit([[4, 4]])
    np.testing.assert_allclose(scaler.transform([[4, 4]]), [[0, 0]])


def test_scalers_reject_unrepresentable_statistics_without_replacing_state():
    for scaler in (StandardScaler(), MinMaxScaler()):
        scaler.fit([[2], [4]])
        expected = scaler.transform([[3]])
        with pytest.raises(ValueError, match="floating-point range"):
            scaler.fit([[-1e308], [1e308]])
        np.testing.assert_allclose(scaler.transform([[3]]), expected)
    for bounds in ((0, np.inf), (0, 0), (0,), "bad"):
        with pytest.raises(ValueError, match="feature_range"):
            MinMaxScaler(bounds)


def test_scalers_reject_overflow_during_transform():
    for scaler in (StandardScaler(), MinMaxScaler()):
        scaler.fit([[-1e308]])
        with pytest.raises(ValueError, match="Scaled values exceed"):
            scaler.transform([[1e308]])


def test_imputer_updates_column_means_and_uses_constant_fill():
    imputer = SimpleImputer().partial_fit([[2, np.nan], [4, np.nan]])
    np.testing.assert_allclose(imputer.transform([[np.nan, np.nan]]), [[3, 0]])
    imputer.partial_fit([[np.nan, 5], [9, np.nan]])
    np.testing.assert_allclose(imputer.transform([[np.nan, np.nan]]), [[5, 5]])

    constant = SimpleImputer(strategy="constant", fill_value=-1)
    constant.partial_fit([[np.nan, 3]]).partial_fit([[8, np.nan]])
    np.testing.assert_allclose(constant.transform([[6, np.nan]]), [[6, -1]])
    with pytest.raises(ValueError, match="fill_value"):
        SimpleImputer(strategy="constant", fill_value=np.nan)


def test_median_imputer_retains_exact_history_and_resets_on_fit():
    chunk = np.array([[1, np.nan], [9, np.nan]])
    imputer = SimpleImputer(strategy="median").partial_fit(chunk)
    chunk[:, 0] = 99
    imputer.partial_fit([[2, 7]])
    np.testing.assert_allclose(imputer.statistics_, [2, 7])
    imputer.fit([[10, 12]]).partial_fit([[20, np.nan]])
    np.testing.assert_allclose(imputer.statistics_, [15, 12])


def test_encoder_appends_new_categories_without_moving_existing_columns():
    encoder = OneHotEncoder().partial_fit([["b", "y"], ["c", "z"]])
    old = encoder.transform([["b", "z"]])
    encoder.partial_fit([["a", "z"]])
    encoder.partial_fit([["c", "x"]])

    np.testing.assert_array_equal(encoder.transform([["b", "z"]])[:, :4], old)
    np.testing.assert_array_equal(encoder.transform([["a", "x"]]), [[0, 0, 0, 0, 1, 1]])
    np.testing.assert_array_equal(encoder.transform([["unknown", "unknown"]]), np.zeros((1, 6)))
    assert encoder.fit_transform([["new"]]).tolist() == [[1.0]]


def test_encoder_keeps_the_first_baseline_across_chunks():
    encoder = OneHotEncoder(drop_first=True).partial_fit([[np.nan]]).partial_fit([[5.0]])
    assert encoder.transform([[5.0]]).shape == (1, 0)
    encoder.partial_fit([[3.0], [5.0], [np.nan]])
    np.testing.assert_array_equal(encoder.transform([[5], [3], [np.nan]]), [[0], [1], [0]])


def test_encoder_preserves_large_numeric_categories_and_validates_input():
    encoder = OneHotEncoder().partial_fit([[2**53 + 1]])
    encoder.partial_fit([[float(2**53)]])
    np.testing.assert_array_equal(encoder.transform([[2**53 + 1], [2**53]]), [[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="expected 1 features"):
        encoder.partial_fit([[1, 2]])
    with pytest.raises(ValueError, match="Category type"):
        encoder.partial_fit([["new"]])
    np.testing.assert_array_equal(encoder.transform([[2**53 + 1]]), [[1, 0]])
