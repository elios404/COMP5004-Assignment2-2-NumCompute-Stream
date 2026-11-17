"""Tests for batch and chunk-wise pipelines."""

import numpy as np
import pytest

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


def test_pipeline_keeps_the_fitted_scaler_until_retraining():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(buffer_size=3)),
    ])
    assert pipe.partial_fit([[0]], [0]) is pipe
    with pytest.raises(RuntimeError, match="not fitted"):
        pipe.predict([[0]])

    pipe.partial_fit([[2], [4]], [0, 1])
    np.testing.assert_array_equal(pipe.predict([[0], [4]]), [0, 1])

    pipe.partial_fit([[100], [102]], [1, 1])
    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, [2])
    np.testing.assert_array_equal(pipe.predict([[0], [4]]), [0, 1])

    pipe.partial_fit([[104], [106]], [2, 2])
    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, [418 / 7])
    np.testing.assert_array_equal(pipe.named_steps["model"].classes_, [1, 2])
    np.testing.assert_array_equal(pipe.predict([[102], [104], [106]]), [1, 2, 2])


def test_pipeline_pairs_encoder_columns_with_each_tree():
    pipe = Pipeline([
        ("encode", OneHotEncoder()),
        ("model", DecisionTreeClassifier(buffer_size=2)),
    ])
    pipe.partial_fit([["a"], ["b"]], [0, 1])
    pipe.partial_fit([["c"]], [2])

    assert pipe.named_steps["model"].n_features_in_ == 2
    np.testing.assert_array_equal(pipe.predict([["a"], ["b"]]), [0, 1])

    pipe.partial_fit([["b"]], [1])
    assert pipe.named_steps["model"].n_features_in_ == 3
    np.testing.assert_array_equal(pipe.predict([["c"], ["b"]]), [2, 1])


def test_pipeline_extends_scaler_after_encoder_discovers_a_category():
    initial = [["a"], ["b"]]
    all_rows = [*initial, ["c"], ["c"]]
    for scaler_type in (StandardScaler, MinMaxScaler):
        pipe = Pipeline([
            ("encode", OneHotEncoder()),
            ("scale", scaler_type()),
            ("model", DecisionTreeClassifier(buffer_size=2)),
        ]).fit(initial, [0, 1])

        fitted_scaler = pipe.named_steps["scale"]
        pipe.partial_fit([["c"]], [2])
        assert pipe.named_steps["scale"] is fitted_scaler
        assert fitted_scaler.n_features_in_ == 2
        np.testing.assert_array_equal(pipe.predict(initial), [0, 1])

        pipe.partial_fit([["c"]], [2])
        assert pipe.named_steps["model"].n_features_in_ == 3
        encoded = pipe.named_steps["encode"].transform(all_rows)
        expected = scaler_type().fit_transform(encoded)
        np.testing.assert_allclose(pipe.named_steps["scale"].transform(encoded), expected)
        np.testing.assert_array_equal(pipe.predict([["c"]]), [2])


def test_pipeline_accepts_zero_one_hot_columns_while_filling_the_buffer():
    for scaler_type in (StandardScaler, MinMaxScaler):
        pipe = Pipeline([
            ("encode", OneHotEncoder(drop_first=True)),
            ("scale", scaler_type()),
            ("model", DecisionTreeClassifier(buffer_size=3)),
        ])
        pipe.partial_fit([["a"]], [0])
        pipe.partial_fit([["a"]], [0])
        assert not pipe.is_fitted

        pipe.partial_fit([["b"]], [1])
        X = [["a"], ["b"]]
        expected = scaler_type().fit([[0.0], [0.0], [1.0]]).transform([[0.0], [1.0]])
        np.testing.assert_allclose(pipe.named_steps["scale"].transform([[0.0], [1.0]]), expected)
        np.testing.assert_array_equal(pipe.predict(X), [0, 1])


def test_pipeline_does_not_add_zero_observations_to_the_first_one_hot_chunk():
    pipe = Pipeline([("encode", OneHotEncoder()), ("scale", MinMaxScaler())])
    pipe.partial_fit([["a"], ["a"]])

    np.testing.assert_array_equal(pipe.named_steps["scale"].data_min_, [1.0])
    np.testing.assert_array_equal(pipe.transform([["a"]]), [[0.0]])


def test_pipeline_updates_transformers_in_order():
    pipe = Pipeline([
        ("impute", SimpleImputer()),
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(buffer_size=4)),
    ])
    pipe.partial_fit([[np.nan], [2]], [0, 0])
    pipe.partial_fit([[8], [10]], [1, 1])

    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, [5.5])
    np.testing.assert_array_equal(pipe.predict([[np.nan], [2], [10]]), [0, 0, 1])


def test_failed_update_preserves_buffer_and_fitted_steps():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(buffer_size=2, max_features=1)),
    ]).fit([[0], [2]], [0, 1])
    chunk = np.array([[4.0]])
    labels = np.array([2])
    pipe.partial_fit(chunk, labels)
    chunk[:] = 1000
    labels[:] = 9

    with pytest.raises(ValueError, match="input features"):
        pipe.partial_fit([[1, 2]], [0])
    with pytest.raises(ValueError, match="finite feature values"):
        pipe.partial_fit([[np.nan]], [3])
    np.testing.assert_array_equal(pipe.predict([[0], [2]]), [0, 1])
    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, [1])

    pipe.partial_fit([[6]], [3])
    np.testing.assert_array_equal(pipe.predict([[4], [6]]), [2, 3])
    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, [3])


def test_fit_clears_pending_stream_rows():
    pipe = Pipeline([("model", DecisionTreeClassifier(buffer_size=2))])
    pipe.partial_fit([[0]], [0])
    pipe.fit([[0, 1]], [5])
    pipe.partial_fit([[1, 2]], [7])

    np.testing.assert_array_equal(pipe.predict([[1, 2]]), [5])
    pipe.partial_fit([[2, 3]], [7])
    np.testing.assert_array_equal(pipe.predict([[1, 2]]), [7])


def test_transformer_only_pipeline_fits_and_updates_without_labels():
    pipe = Pipeline([("scale", StandardScaler())])
    np.testing.assert_allclose(pipe.fit_transform([[0], [2]]), [[-1], [1]])

    pipe.partial_fit([[4]])
    np.testing.assert_allclose(pipe.transform([[2]]), [[0]])


def test_batch_pipeline_accepts_transformer_without_fit_transform():
    class Offset:
        def fit(self, X, y=None):
            self.offset = X[0]

        def transform(self, X):
            return X - self.offset

    pipe = Pipeline([("offset", Offset())])
    np.testing.assert_array_equal(pipe.fit_transform([[3], [4]]), [[0], [1]])
    with pytest.raises(TypeError, match="must implement partial_fit"):
        pipe.partial_fit([[8]])
    np.testing.assert_array_equal(pipe.transform([[4]]), [[1]])


def test_pipeline_rejects_nonfinite_tree_input_before_buffering():
    pipe = Pipeline([("model", DecisionTreeClassifier(buffer_size=2))])
    with pytest.raises(ValueError, match="finite feature values"):
        pipe.partial_fit([[np.nan]], [0])

    pipe.partial_fit([[0]], [0])
    with pytest.raises(RuntimeError, match="not fitted"):
        pipe.predict([[0]])
    pipe.partial_fit([[1]], [1])
    np.testing.assert_array_equal(pipe.predict([[0], [1]]), [0, 1])


def test_pipeline_updates_an_unbuffered_estimator():
    class LastLabel:
        def fit(self, X, y):
            return self.partial_fit(X, y)

        def partial_fit(self, X, y):
            self.label = y[-1]

        def predict(self, X):
            return np.full(X.shape[0], self.label)

    pipe = Pipeline([("scale", StandardScaler()), ("model", LastLabel())])
    pipe.partial_fit([[0]], [1])
    np.testing.assert_array_equal(pipe.predict([[0]]), [1])

    pipe.partial_fit([[2]], [2])
    np.testing.assert_allclose(pipe.named_steps["scale"].mean_, [1])
    np.testing.assert_array_equal(pipe.predict([[0]]), [2])
