"""Preprocessing transformers with chunk-wise fitting."""

from __future__ import annotations

import numpy as np

from numcompute.pipeline import BaseTransformer


def _check_shape(X: np.ndarray, n_features: int | None, training: bool) -> None:
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("Expected a 2D array with at least one feature.")
    if training and X.shape[0] == 0:
        raise ValueError("Cannot fit on an empty array.")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"Feature dimension mismatch: expected {n_features} features, got {X.shape[1]}.")


def _numeric_input(X: np.ndarray, n_features: int | None, training: bool) -> np.ndarray:
    X = np.asarray(X)
    _check_shape(X, n_features, training)

    if X.dtype.kind not in "biuf":
        raise ValueError("Expected real numeric features.")
    with np.errstate(over="ignore"):
        X = X.astype(np.float64, copy=False)
    if np.isinf(X).any():
        raise ValueError("Features must be finite or NaN.")
    
    return X


def _chunk_mean(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = ~np.isnan(X)
    count = observed.sum(axis=0)
    mean = (np.where(observed, X, 0.0) / np.maximum(count, 1)).sum(axis=0)
    return count, mean


def _merge_means(
    count: np.ndarray, mean: np.ndarray, chunk_count: np.ndarray, chunk_mean: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    total = count + chunk_count
    chunk_weight = chunk_count / np.maximum(total, 1)
    merged = mean * (count / np.maximum(total, 1)) + chunk_mean * chunk_weight

    return total, merged


class StandardScaler(BaseTransformer):
    """Standardise numeric features using cumulative population variance.

    Fit and transform accept arrays of shape (N, D). Missing values are
    skipped during fitting and preserved during transformation. Columns
    with no observations use a centre of zero and a scale of one.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.n_features_in_ = None
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_samples_seen_ = None
        self._mean = None

    def fit(self, X: np.ndarray, y=None) -> StandardScaler:
        """Replace the fitted statistics using X."""
        return self._update(X, reset=True)

    def partial_fit(self, X: np.ndarray, y=None) -> StandardScaler:
        """Merge the statistics of a new chunk into the fitted state."""
        return self._update(X, reset=False)

    def _update(self, X: np.ndarray, reset: bool) -> StandardScaler:
        X = _numeric_input(X, None if reset else self.n_features_in_, training=True)
        with np.errstate(over="ignore", invalid="ignore"):
            count, mean = _chunk_mean(X)
            variance = None

            if self.with_std:
                centred = np.where(np.isnan(X), 0.0, X - mean)
                scaled = centred / np.sqrt(np.maximum(count, 1))
                variance = np.square(scaled).sum(axis=0)

            if not reset and self.n_samples_seen_ is not None:
                old_count = self.n_samples_seen_
                total, merged_mean = _merge_means(old_count, self._mean, count, mean)
                if self.with_std:
                    old_weight = old_count / np.maximum(total, 1)
                    new_weight = count / np.maximum(total, 1)
                    weight = np.sqrt(old_weight * new_weight)
                    distance = mean * weight - self._mean * weight
                    variance = old_weight * self.var_ + new_weight * variance + distance ** 2
                count, mean = total, merged_mean

        if not np.isfinite(mean).all() or (variance is not None and not np.isfinite(variance).all()):
            raise ValueError("StandardScaler statistics exceed the floating-point range.")
        
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = count
        self._mean = mean
        self.mean_ = mean if self.with_mean else None
        self.var_ = variance
        self.scale_ = None if variance is None else np.where(variance == 0, 1.0, np.sqrt(variance))

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the scaled (N, D) array using the fitted statistics."""
        if self.n_features_in_ is None:
            raise RuntimeError("StandardScaler is not fitted yet.")
        
        result = _numeric_input(X, self.n_features_in_, training=False).copy()
        with np.errstate(over="ignore"):
            if self.with_mean:
                result -= self.mean_
            if self.with_std:
                result /= self.scale_
        if np.isinf(result).any():
            raise ValueError("Scaled values exceed the floating-point range.")
        
        return result


class MinMaxScaler(BaseTransformer):
    """Scale numeric features using cumulative minima and maxima.

    NaNs are skipped during fitting and preserved during transformation.
    Columns with no observations pass through until a value is observed.
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)):
        bounds = np.asarray(feature_range)
        if bounds.shape != (2,) or bounds.dtype.kind not in "iuf":
            raise ValueError("feature_range must contain two real numbers.")
        
        lower, upper = map(float, bounds)
        if not np.isfinite([lower, upper, upper - lower]).all() or lower >= upper:
            raise ValueError("feature_range must be finite and strictly increasing.")
        
        self.feature_range = (lower, upper)
        self.n_features_in_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None

    def fit(self, X: np.ndarray, y=None) -> MinMaxScaler:
        """Replace the extrema using X of shape (N, D)."""
        return self._update(X, reset=True)

    def partial_fit(self, X: np.ndarray, y=None) -> MinMaxScaler:
        """Extend the per-feature extrema using a chunk of shape (N, D)."""
        return self._update(X, reset=False)

    def _update(self, X: np.ndarray, reset: bool) -> MinMaxScaler:
        X = _numeric_input(X, None if reset else self.n_features_in_, training=True)
        minimum = np.fmin.reduce(X, axis=0)
        maximum = np.fmax.reduce(X, axis=0)

        if not reset and self.data_min_ is not None:
            minimum = np.fmin(self.data_min_, minimum)
            maximum = np.fmax(self.data_max_, maximum)
        with np.errstate(over="ignore"):
            data_range = maximum - minimum
        if np.isinf(data_range).any():
            raise ValueError("MinMaxScaler data range exceeds the floating-point range.")
        
        self.n_features_in_ = X.shape[1]
        self.data_min_ = minimum
        self.data_max_ = maximum
        self.data_range_ = data_range

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the scaled (N, D) array without clipping unseen values."""
        if self.n_features_in_ is None:
            raise RuntimeError("MinMaxScaler is not fitted yet.")
        
        X = _numeric_input(X, self.n_features_in_, training=False)
        lower, upper = self.feature_range
        observed = ~np.isnan(self.data_min_)
        scale = np.where((self.data_range_ == 0) | ~observed, 1.0, self.data_range_)

        with np.errstate(over="ignore"):
            scaled = (X - self.data_min_) / scale * (upper - lower) + lower
        if np.isinf(scaled).any():
            raise ValueError("Scaled values exceed the floating-point range.")
        
        return np.where(observed, scaled, X)


class SimpleImputer(BaseTransformer):
    """Replace missing values with a cumulative mean, median or constant.

    Unobserved columns use zero for mean and median imputation. Exact median
    imputation retains the input chunks, so its memory grows with the stream.
    """

    def __init__(self, strategy: str = "mean", fill_value: float | None = None):
        if not isinstance(strategy, str) or strategy not in ("mean", "median", "constant"):
            raise ValueError("strategy must be 'mean', 'median' or 'constant'.")
        if strategy == "constant":
            value = np.asarray(fill_value)
            if value.ndim != 0 or value.dtype.kind not in "biuf" or not np.isfinite(value):
                raise ValueError("A finite numeric fill_value is required for the constant strategy.")
            
        self.strategy = strategy
        self.fill_value = fill_value
        self.n_features_in_ = None
        self.statistics_ = None
        self._count = None
        self._chunks: list[np.ndarray] = []

    def fit(self, X: np.ndarray, y=None) -> SimpleImputer:
        """Replace the imputation statistics using X of shape (N, D)."""
        return self._update(X, reset=True)

    def partial_fit(self, X: np.ndarray, y=None) -> SimpleImputer:
        """Update the imputation statistics from another chunk of shape (N, D)."""
        return self._update(X, reset=False)

    def _update(self, X: np.ndarray, reset: bool) -> SimpleImputer:
        X = _numeric_input(X, None if reset else self.n_features_in_, training=True)
        count = None
        chunks = []

        with np.errstate(over="ignore", invalid="ignore"):
            if self.strategy == "constant":
                statistics = np.full(X.shape[1], self.fill_value, dtype=np.float64)
            elif self.strategy == "mean":
                count, statistics = _chunk_mean(X)
                if not reset and self._count is not None:
                    count, statistics = _merge_means(self._count, self.statistics_, count, statistics)
            else:
                chunks = [X.copy()] if reset else [*self._chunks, X.copy()]
                history = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
                observed = (~np.isnan(history)).any(axis=0)
                statistics = np.zeros(X.shape[1])
                statistics[observed] = np.nanmedian(history[:, observed], axis=0)

        if not np.isfinite(statistics).all():
            raise ValueError("Imputation statistics exceed the floating-point range.")
        
        self.n_features_in_ = X.shape[1]
        self.statistics_ = statistics
        self._count = count
        self._chunks = chunks

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return an (N, D) array with NaNs replaced by fitted values."""
        if self.n_features_in_ is None:
            raise RuntimeError("SimpleImputer is not fitted yet.")
        
        X = _numeric_input(X, self.n_features_in_, training=False)
        return np.where(np.isnan(X), self.statistics_, X)


class OneHotEncoder(BaseTransformer):
    """Encode categorical columns and append new output columns as needed.

    Inputs of shape (N, D) use a numeric, Unicode or byte-string array dtype.
    Unknown categories and NaNs encode as zeros. With drop_first, the first
    category learnt in each feature remains the baseline, even for a singleton.
    """

    def __init__(self, drop_first: bool = False):
        self.drop_first = drop_first
        self.n_features_in_ = None
        self.categories_ = None
        self._family = None
        self._output_columns: list[tuple[int, np.ndarray]] = []

    def fit(self, X: np.ndarray, y=None) -> OneHotEncoder:
        """Replace learned categories using X."""
        return self._update(X, reset=True)

    def partial_fit(self, X: np.ndarray, y=None) -> OneHotEncoder:
        """Discover new categories without changing existing output positions."""
        return self._update(X, reset=False)

    def _categorical_input(self, X: np.ndarray, training: bool, reset: bool = False):
        X = np.asarray(X)
        _check_shape(X, None if reset else self.n_features_in_, training)
        family = X.dtype.kind

        if family in "biuf":
            if np.isinf(X).any():
                raise ValueError("Numeric categories must be finite or NaN.")
            family = "numeric"
        elif family not in "US":
            raise ValueError("Expected numeric, Unicode or byte-string categories.")
        
        if not reset and self._family is not None and family != self._family:
            raise ValueError("Category type does not match the fitted encoder.")
        
        return X, family

    def _update(self, X: np.ndarray, reset: bool) -> OneHotEncoder:
        X, family = self._categorical_input(X, training=True, reset=reset)
        categories = []
        output_columns = [] if reset else self._output_columns.copy()

        for feature, column in enumerate(X.T):
            if column.dtype.kind == "f":
                column = column[~np.isnan(column)]

            discovered = np.unique(column).astype(object)
            known = (
                np.empty(0, dtype=object)
                if reset or self.categories_ is None else self.categories_[feature]
            )
            new = discovered[~np.isin(discovered, known)]
            categories.append(np.concatenate((known, new)))
            emitted = new[1:] if self.drop_first and known.size == 0 else new
            if emitted.size:
                output_columns.append((feature, emitted))

        self.n_features_in_ = X.shape[1]
        self.categories_ = categories
        self._family = family
        self._output_columns = output_columns

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return the (N, C) one-hot array in the learned output order."""
        if self.n_features_in_ is None:
            raise RuntimeError("OneHotEncoder is not fitted yet.")
        
        X, _ = self._categorical_input(X, training=False)
        X = X.astype(object)
        blocks = [X[:, feature, None] == categories for feature, categories in self._output_columns]
        
        if not blocks:
            return np.empty((X.shape[0], 0), dtype=np.float64)
        return np.concatenate(blocks, axis=1).astype(np.float64)
