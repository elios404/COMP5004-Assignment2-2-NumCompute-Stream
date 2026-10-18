"""
Preprocessing transformers for NumCompute.
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from numcompute.pipeline import BaseTransformer


class StandardScaler(BaseTransformer):
    """
    Standardize features by removing the mean and scaling to unit variance.

    Parameters
    ----------
    with_mean : bool, default=True
        If True, center the data before scaling.
    with_std : bool, default=True
        If True, scale the data to unit variance.

    Attributes
    ----------
    mean_ : Optional[np.ndarray]
        Per-feature empirical mean. Shape: (D,)
    scale_ : Optional[np.ndarray]
        Per-feature relative scaling factor (standard deviation). Shape: (D,)
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> StandardScaler:
        """
        Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (N, D).
        y : None
            Ignored.For the pipeline API consistency.

        Returns
        -------
        StandardScaler
            Fitted scaler instance.

        Raises
        ------
        ValueError
            If X is not a 2D array or has 0 rows.

        Complexity
        ----------
        Time: O(N * D)
        Space: O(D)
        """
        X = np.asarray(X, dtype=np.float64) # in case of different input types (like lists)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on empty array.")

        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = None

        if self.with_std:
            std = np.std(X, axis=0)
            self.scale_ = std.copy()
            self.scale_[self.scale_ == 0.0] = 1.0
        else:
            self.scale_ = None

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling.

        Parameters
        ----------
        X : np.ndarray
            Data to transform of shape (N, D).

        Returns
        -------
        np.ndarray
            Transformed array of shape (N, D).

        Raises
        ------
        RuntimeError
            If transform is called before fit.
        ValueError
            If X is not a 2D array.

        Complexity
        ----------
        Time: O(N * D)
        Space: O(N * D)
        """
        if self.with_mean and self.mean_ is None:
            raise RuntimeError("StandardScaler is not fitted yet.")
        if self.with_std and self.scale_ is None:
            raise RuntimeError("StandardScaler is not fitted yet.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")

        X_out = X.copy()
        if self.with_mean:
            X_out -= self.mean_
        if self.with_std:
            X_out /= self.scale_

        return X_out

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)


class MinMaxScaler(BaseTransformer):
    """
    Transform features by scaling each feature to a given range.

    Parameters
    ----------
    feature_range : Tuple[float, float], default=(0.0, 1.0)
        Desired range of transformed data (min, max).

    Attributes
    ----------
    data_min_ : Optional[np.ndarray]
        Per-feature minimum seen in the data. Shape: (D,)
    data_max_ : Optional[np.ndarray]
        Per-feature maximum seen in the data. Shape: (D,)
    data_range_ : Optional[np.ndarray]
        Per-feature range (data_max_ - data_min_). Shape: (D,)
    """

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        if feature_range[0] >= feature_range[1]:
            raise ValueError("feature_range minimum must be strictly smaller than maximum.")
        self.feature_range = feature_range
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None
        self.data_range_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> MinMaxScaler:
        """
        Compute the minimum and maximum of X for later scaling.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (N, D).
        y : Optional[np.ndarray], default=None
            Ignored. Present for API consistency.

        Returns
        -------
        MinMaxScaler
            Fitted scaler instance.

        Raises
        ------
        ValueError
            If X is not a 2D array or has 0 rows.

        Complexity
        ----------
        Time: O(N * D) for finding minimum and maximum along axis 0.
        Space: O(D) to store per-feature min, max, and range statistics.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on empty array.")

        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        range_ = self.data_max_ - self.data_min_

        self.data_range_ = range_.copy()
        self.data_range_[self.data_range_ == 0.0] = 1.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features of X according to feature_range.

        Parameters
        ----------
        X : np.ndarray
            Data to transform of shape (N, D).

        Returns
        -------
        np.ndarray
            Transformed array of shape (N, D).

        Raises
        ------
        RuntimeError
            If transform is called before fit.
        ValueError
            If X is not a 2D array.

        Complexity
        ----------
        Time: O(N * D) for vectorised linear scaling.
        Space: O(N * D) to allocate memory for the returned array.
        """
        if self.data_min_ is None or self.data_range_ is None: # needed for calculating the scaling
            raise RuntimeError("MinMaxScaler is not fitted yet.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")

        min_val, max_val = self.feature_range
        norm = (X - self.data_min_) / self.data_range_

        return norm * (max_val - min_val) + min_val

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)


class SimpleImputer(BaseTransformer):
    """
    Univariate imputer for completing missing values with simple statistics.

    Parameters
    ----------
    strategy : str, default='mean'
        The imputation strategy. Supported: 'mean', 'median', 'constant'.
    fill_value : Optional[float], default=None
        Value used when strategy='constant'.

    Attributes
    ----------
    statistics_ : Optional[np.ndarray]
        Per-feature fill values calculated along axis 0. Shape: (D,)
    """

    def __init__(self, strategy: str = "mean", fill_value: Optional[float] = None):
        if strategy not in ("mean", "median", "constant"):
            raise ValueError(f"Unknown strategy '{strategy}'. Choose 'mean', 'median', or 'constant'.")
        if strategy == "constant" and fill_value is None:
            raise ValueError("fill_value must be provided when strategy='constant'.")

        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> SimpleImputer:
        """
        Compute the imputation statistics for each column.

        Parameters
        ----------
        X : np.ndarray
            Training data containing potential NaNs. Shape: (N, D).
        y : Optional[np.ndarray], default=None
            Ignored. Present for API consistency.

        Returns
        -------
        SimpleImputer
            Fitted imputer instance.

        Raises
        ------
        ValueError
            If X is not a 2D array or has 0 rows.

        Complexity
        ----------
        Time: O(N * D) for computing column-wise nan-aware statistics.
        Space: O(D) to store the statistics vector.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on empty array.")

        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == "constant":
            self.statistics_ = np.ones(X.shape[1], dtype=np.float64) * self.fill_value

        self.statistics_[np.isnan(self.statistics_)] = 0.0

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Impute all missing values in X with the fitted statistics.

        Parameters
        ----------
        X : np.ndarray
            Data to impute. Shape: (N, D).

        Returns
        -------
        np.ndarray
            Imputed array with NaNs replaced. Shape: (N, D).

        Raises
        ------
        RuntimeError
            If transform is called before fit.
        ValueError
            If X is not a 2D array.

        Complexity
        ----------
        Time: O(N * D) for finding NaNs and vectorized replacement.
        Space: O(N * D) to allocate memory for the imputed copy.
        """
        if self.statistics_ is None:
            raise RuntimeError("SimpleImputer is not fitted yet.")

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")

        X_out = X.copy()
        nan_mask = np.isnan(X_out)

        X_out = np.where(nan_mask, self.statistics_, X_out)

        return X_out

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)
    

class OneHotEncoder(BaseTransformer):
    """
    Encode categorical features as a one-hot numeric array using pure NumPy.
    Assumption: All features are categorical and represented as integers or strings.

    Parameters
    ----------
    drop_first : bool, default=False
        Whether to drop the first category per feature to avoid multicollinearity.

    Attributes
    ----------
    categories_ : Optional[list[np.ndarray]]
        List of unique categories discovered per feature column during fit.
    """

    def __init__(self, drop_first: bool = False):
        self.drop_first = drop_first
        self.categories_: Optional[list[np.ndarray]] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> OneHotEncoder:
        """
        Identify unique categories per column.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (N, D).
        y : Optional[np.ndarray], default=None
            Ignored. Present for API consistency.

        Returns
        -------
        OneHotEncoder
            Fitted encoder instance.

        Raises
        ------
        ValueError
            If X is not a 2D array or has 0 rows.

        Complexity
        ----------
        Time: O(D * N log N) due to sorting in np.unique for each column.
        Space: O(C) where C is the total number of unique categories across all features.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on empty array.")

        cats_list = []
        for col_idx in range(X.shape[1]):
            unique_cats = np.unique(X[:, col_idx])
            
            if self.drop_first and len(unique_cats) > 1:
                unique_cats = unique_cats[1:]
                
            cats_list.append(unique_cats)

        self.categories_ = cats_list
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X into one-hot numeric representation using broadcasting.

        Parameters
        ----------
        X : np.ndarray
            Data to encode of shape (N, D).

        Returns
        -------
        np.ndarray
            Binary one-hot encoded matrix of shape (N, total_unique_categories).

        Raises
        ------
        RuntimeError
            If transform is called before fit.
        ValueError
            If X is not a 2D array or feature count does not match fitted categories.

        Complexity
        ----------
        Time: O(N * C) where C is the total number of categories (vectorised comparison).
        Space: O(N * C) for the output encoded array.
        """
        if self.categories_ is None:
            raise RuntimeError("OneHotEncoder is not fitted yet.")

        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead.")
        if X.shape[1] != len(self.categories_):
            raise ValueError(f"Feature dimension mismatch: expected {len(self.categories_)} features, got {X.shape[1]}.")

        one_hot_blocks = []
        for col_idx, cats in enumerate(self.categories_):
            if len(cats) == 0:
                continue

            col_data = X[:, col_idx : col_idx + 1]

            encoded_block = (col_data == cats.reshape(1, -1)).astype(np.float64)
            one_hot_blocks.append(encoded_block)

        if not one_hot_blocks:
            return np.empty((X.shape[0], 0), dtype=np.float64)

        return np.hstack(one_hot_blocks)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X, y).transform(X)