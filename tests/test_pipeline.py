"""
Unit tests for pipeline abstractions (numcompute/pipeline.py).
"""

from __future__ import annotations
from typing import Optional
import numpy as np
import pytest

from numcompute.pipeline import (
    BaseEstimator,
    BaseTransformer,
    FeatureUnion,
    Pipeline,
)
from numcompute.preprocessing import SimpleImputer, StandardScaler


# ==============================================================================
# Test Doubles (Mocks for testing pipeline flow)
# ==============================================================================

class AddConstantTransformer(BaseTransformer):
    """Simple dummy transformer that adds a constant value."""

    def __init__(self, value: float = 1.0):
        self.value = value

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> AddConstantTransformer:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X + self.value


class MultiplyConstantTransformer(BaseTransformer):
    """Simple dummy transformer that multiplies by a constant value."""

    def __init__(self, factor: float = 2.0):
        self.factor = factor

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> MultiplyConstantTransformer:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.factor


class MockEstimator(BaseEstimator):
    """Simple dummy predictor returning row sums."""

    def __init__(self):
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> MockEstimator:
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Estimator is not fitted.")
        # Return row sums as dummy prediction
        return np.sum(X, axis=1)


# ==============================================================================
# 1. Sequential Pipeline Tests
# ==============================================================================

def test_pipeline_chaining_transformers():
    """Test sequential transformation: (X + 1) * 2."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    pipe = Pipeline([
        ("add", AddConstantTransformer(value=1.0)),
        ("mul", MultiplyConstantTransformer(factor=2.0)),
    ])

    # fit_transform
    X_out = pipe.fit_transform(X)
    expected = (X + 1.0) * 2.0
    assert np.allclose(X_out, expected)

    # transform on new data
    X_new = np.array([[0.0, 0.0]])
    assert np.allclose(pipe.transform(X_new), [[2.0, 2.0]])


def test_pipeline_with_real_preprocessors():
    """Test realistic end-to-end pipeline: SimpleImputer -> StandardScaler."""
    X = np.array([[1.0, 10.0], [np.nan, 20.0], [3.0, np.nan]])

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
    ])

    X_processed = pipe.fit_transform(X)

    # Ensure no NaNs remain and data is standardized
    assert not np.isnan(X_processed).any()
    assert np.allclose(np.mean(X_processed, axis=0), [0.0, 0.0])
    assert np.allclose(np.std(X_processed, axis=0), [1.0, 1.0])


def test_pipeline_with_final_estimator():
    """Test pipeline ending with an estimator: transform X then predict."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([1, 0])

    pipe = Pipeline([
        ("add", AddConstantTransformer(value=2.0)),
        ("model", MockEstimator()),
    ])

    pipe.fit(X, y)
    preds = pipe.predict(X)

    # After add(+2): [[3, 4], [5, 6]] -> row sums: [7, 11]
    assert np.allclose(preds, [7.0, 11.0])

    # Calling transform on an estimator pipeline must raise AttributeError
    with pytest.raises(AttributeError, match="does not implement 'transform'"):
        pipe.transform(X)


def test_pipeline_named_steps():
    """Test accessing individual components via named_steps property."""
    scaler = StandardScaler()
    pipe = Pipeline([("scaler", scaler)])

    assert "scaler" in pipe.named_steps
    assert pipe.named_steps["scaler"] is scaler


# ==============================================================================
# 2. Pipeline Error Handling & Edge Cases
# ==============================================================================

def test_pipeline_empty_error():
    """Edge Case: Empty pipeline steps must raise ValueError."""
    with pytest.raises(ValueError, match="steps cannot be empty"):
        Pipeline([])


def test_pipeline_invalid_intermediate_step():
    """Edge Case: Intermediate steps must implement both fit and transform."""
    with pytest.raises(TypeError, match="must implement both 'fit' and 'transform'"):
        Pipeline([
            ("invalid_step", "not_a_transformer"),
            ("final_step", AddConstantTransformer()),
        ])


def test_pipeline_unfitted_call():
    """Edge Case: Calling transform or predict before fit raises RuntimeError."""
    pipe = Pipeline([("step", AddConstantTransformer())])
    with pytest.raises(RuntimeError, match="not fitted yet"):
        pipe.transform(np.array([[1.0, 2.0]]))


# ==============================================================================
# 3. FeatureUnion Tests (Parallel Combiner)
# ==============================================================================

def test_feature_union_basic():
    """Test parallel feature concatenation: [X + 1, X * 10] -> horizontal stack."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])

    union = FeatureUnion([
        ("add", AddConstantTransformer(value=1.0)),
        ("mul", MultiplyConstantTransformer(factor=10.0)),
    ])

    X_out = union.fit(X).transform(X)

    # (2, 2) and (2, 2) stacked horizontally -> shape (2, 4)
    assert X_out.shape == (2, 4)
    expected = np.hstack([X + 1.0, X * 10.0])
    assert np.allclose(X_out, expected)


def test_feature_union_empty():
    """Edge Case: Empty FeatureUnion raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        FeatureUnion([])