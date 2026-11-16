"""Unit tests for classification trees."""

import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier, _best_split, _impurity_from_counts


def test_impurity_from_counts_for_pure_and_balanced_groups():
    counts = np.array([[4, 0], [2, 2]])

    np.testing.assert_array_equal(
        _impurity_from_counts(counts, "gini"), [0.0, 0.5]
    )
    np.testing.assert_array_equal(
        _impurity_from_counts(counts, "entropy"), [0.0, 1.0]
    )


def test_best_split_separates_the_two_classes():
    X = [[4, 1], [4, 2], [4, 8], [4, 9]]
    y = [0, 0, 1, 1]

    assert _best_split(X, y) == (1, 5.0, 0.5)
    assert _best_split(X, y, "entropy") == (1, 5.0, 1.0)


def test_best_split_weights_children_by_sample_count():
    feature, threshold, gain = _best_split([[1], [2], [2], [2]], [0, 0, 1, 1])

    assert (feature, threshold) == (0, 1.5)
    assert gain == pytest.approx(1 / 6)


def test_best_split_prefers_first_feature_and_lowest_threshold_on_ties():
    X = [[1, 1], [2, 2], [3, 3]]

    assert _best_split(X, [0, 1, 0])[:2] == (0, 1.5)


def test_best_split_returns_none_without_an_improvement():
    assert _best_split([[2], [2]], [0, 1]) is None
    assert _best_split([[1], [1], [2], [2]], [0, 1, 0, 1]) is None


def test_best_split_handles_adjacent_float_values():
    lower = np.nextafter(1.0, 2.0)
    upper = np.nextafter(lower, 2.0)

    assert _best_split([[lower], [upper]], [0, 1]) == (0, lower, 0.5)


def test_best_split_rejects_nonfinite_features():
    with pytest.raises(ValueError, match="finite feature values"):
        _best_split([[1], [np.nan]], [0, 1])


def test_tree_learns_multiple_levels_and_preserves_labels():
    X = [[0], [1], [2], [3], [4], [5]]
    y = ["low", "low", "middle", "middle", "high", "high"]
    tree = DecisionTreeClassifier(max_depth=2, criterion="entropy")

    assert tree.fit(X, y) is tree
    np.testing.assert_array_equal(tree.predict([[0.5], [1.5], [2.5], [4.5]]),
                                  ["low", "low", "middle", "high"])


def test_tree_stops_at_depth_and_sample_limits():
    X = [[0], [1], [2], [3]]
    y = [1, 1, 0, 0]

    for tree in (DecisionTreeClassifier(max_depth=0),
                 DecisionTreeClassifier(min_samples_split=5)):
        tree.fit(X, y)
        np.testing.assert_array_equal(tree.predict(X), [0, 0, 0, 0])

    tree = DecisionTreeClassifier(max_depth=1, min_samples_split=4).fit(X, y)
    np.testing.assert_array_equal(tree.predict(X), y)


def test_tree_predicts_majority_when_no_split_is_possible():
    tree = DecisionTreeClassifier().fit([[2], [2], [2]], [7, 4, 7])
    np.testing.assert_array_equal(tree.predict([[2], [3]]), [7, 7])


def test_tree_refit_replaces_the_previous_tree():
    tree = DecisionTreeClassifier().fit([[1], [2]], [0, 1])
    tree.fit([[1, 2], [3, 4]], [7, 7])

    np.testing.assert_array_equal(tree.predict([[9, 10]]), [7])
    assert tree.predict(np.empty((0, 2))).shape == (0,)
    with pytest.raises(ValueError, match="Expected 2 features, got 1"):
        tree.predict([[1]])


def test_tree_rejects_invalid_inputs():
    tree = DecisionTreeClassifier()
    with pytest.raises(RuntimeError, match="not fitted"):
        tree.predict([[1]])
    with pytest.raises(ValueError, match="2D array"):
        tree.fit([1, 2], [0, 1])
    with pytest.raises(ValueError, match="one label per sample"):
        tree.fit([[1], [2]], [0])
    with pytest.raises(ValueError, match="finite feature values"):
        tree.fit([[1], [np.nan]], [0, 1])
    with pytest.raises(ValueError, match="finite numeric labels"):
        tree.fit([[1], [2]], [0, np.nan])


def test_tree_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="max_depth"):
        DecisionTreeClassifier(max_depth=-1)
    with pytest.raises(ValueError, match="min_samples_split"):
        DecisionTreeClassifier(min_samples_split=1)
    with pytest.raises(ValueError, match="criterion"):
        DecisionTreeClassifier(criterion="unknown")


def test_best_split_searches_only_the_requested_features():
    X = [[0, 0], [0, 1], [1, 2], [1, 2]]
    y = [0, 0, 1, 1]

    assert _best_split(X, y)[0] == 0
    assert _best_split(X, y, features=np.array([1])) == (1, 1.5, 0.5)


def test_tree_samples_features_reproducibly_at_each_node():
    values = np.arange(6)
    X = np.tile(values[:, None], (1, 4))
    y = [0, 0, 1, 1, 2, 2]
    tree = DecisionTreeClassifier(max_features=1, random_state=42).fit(X, y)
    rng = np.random.default_rng(42)
    root_feature = rng.choice(4, size=1, replace=False)[0]
    child_feature = rng.choice(4, size=1, replace=False)[0]

    assert tree.root_.feature == root_feature
    assert tree.root_.right.feature == child_feature
    np.testing.assert_array_equal(tree.predict(X), y)
    tree.fit(X, y)
    assert tree.root_.feature == root_feature
    assert tree.root_.right.feature == child_feature


def test_tree_validates_feature_sampling_settings():
    for max_features in (0, True, 1.5):
        with pytest.raises(ValueError, match="max_features"):
            DecisionTreeClassifier(max_features=max_features)
    with pytest.raises(ValueError, match="number of input features"):
        DecisionTreeClassifier(max_features=2).fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError, match="random_state"):
        DecisionTreeClassifier(random_state=-1)


def test_tree_preserves_adjacent_large_integer_features():
    for X in (
        np.array([[2**53], [2**53 + 1]], dtype=np.int64),
        np.array([[2**64 - 2], [2**64 - 1]], dtype=np.uint64),
    ):
        tree = DecisionTreeClassifier().fit(X, [0, 1])
        np.testing.assert_array_equal(tree.predict(X), [0, 1])


def test_tree_keeps_split_boundary_with_mixed_numeric_dtypes():
    boundary = 2**53
    tree = DecisionTreeClassifier().fit([[boundary + 3], [boundary + 4]], [0, 1])
    np.testing.assert_array_equal(tree.predict([[float(boundary + 4)]]), [1])

    tree.fit([[float(boundary)], [float(boundary + 2)]], [0, 1])
    np.testing.assert_array_equal(tree.predict([[boundary], [boundary + 1]]), [0, 1])

    tree.fit([[0.99999998], [1.0]], [0, 1])
    np.testing.assert_array_equal(tree.predict(np.array([[1.0]], dtype=np.float32)), [1])

    tree.fit(np.array([[2**64 - 2], [2**64 - 1]], dtype=np.uint64), [0, 1])
    np.testing.assert_array_equal(tree.predict(np.array([[127]], dtype=np.int8)), [0])


def test_tree_rejects_array_criterion():
    with pytest.raises(ValueError, match="criterion"):
        DecisionTreeClassifier(criterion=np.array(["gini"]))


def test_partial_fit_retrains_at_and_above_buffer_size():
    tree = DecisionTreeClassifier(buffer_size=3)
    assert tree.partial_fit([[0]], [0]) is tree
    with pytest.raises(RuntimeError, match="not fitted"):
        tree.predict([[0]])

    tree.partial_fit([[1], [2]], [0, 1])
    np.testing.assert_array_equal(tree.predict([[0], [2]]), [0, 1])

    tree.partial_fit([[0], [1]], [2, 2])
    np.testing.assert_array_equal(tree.predict([[0], [2]]), [0, 1])
    np.testing.assert_array_equal(tree.classes_, [0, 1])

    tree.partial_fit([[2], [3]], [2, 3])
    np.testing.assert_array_equal(tree.classes_, [2, 3])
    np.testing.assert_array_equal(tree.predict([[0], [2], [3]]), [2, 2, 3])

    tree.partial_fit([[3]], [4])
    np.testing.assert_array_equal(tree.predict([[3]]), [3])


def test_partial_fit_copies_pending_features_and_labels():
    tree = DecisionTreeClassifier(buffer_size=2)
    X = np.array([[0]])
    y = np.array([0])
    tree.partial_fit(X, y)
    X[0, 0] = 10
    y[0] = 1

    tree.partial_fit([[10]], [1])
    np.testing.assert_array_equal(tree.predict([[0], [10]]), [0, 1])


def test_partial_fit_rejects_invalid_chunks_without_changing_pending_data():
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit([[0]], [0])

    with pytest.raises(ValueError, match="Expected 1 features, got 2"):
        tree.partial_fit([[1, 2]], [1])
    with pytest.raises(ValueError, match="one label per sample"):
        tree.partial_fit(np.empty((0, 1)), [])
    with pytest.raises(ValueError, match="one label per sample"):
        tree.partial_fit([[1]], [[1]])
    with pytest.raises(ValueError, match="finite feature values"):
        tree.partial_fit([[np.nan]], [1])
    with pytest.raises(ValueError, match="numeric labels or string labels"):
        tree.partial_fit([[1]], ["one"])
    with pytest.raises(RuntimeError, match="not fitted"):
        tree.predict([[0]])

    tree.partial_fit([[1]], [1])
    with pytest.raises(ValueError, match="Expected 1 features, got 2"):
        tree.partial_fit([[1, 2]], [1])
    np.testing.assert_array_equal(tree.predict([[0], [1]]), [0, 1])


def test_fit_clears_pending_chunks_only_after_success():
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit([[0]], [0])
    with pytest.raises(ValueError, match="one label per sample"):
        tree.fit([[1]], [])
    tree.partial_fit([[1]], [1])
    np.testing.assert_array_equal(tree.predict([[0], [1]]), [0, 1])

    tree.partial_fit([[2]], [2])
    tree.fit([[0, 1]], [7])
    tree.partial_fit([[1, 2]], [8])
    np.testing.assert_array_equal(tree.predict([[1, 2]]), [7])
    tree.partial_fit([[2, 3]], [8])
    np.testing.assert_array_equal(tree.predict([[1, 2]]), [8])


def test_partial_fit_keeps_model_and_buffer_when_retraining_fails():
    boundary = 2**53
    tree = DecisionTreeClassifier(buffer_size=2).fit([[0]], [7])
    tree.partial_fit([[boundary + 1]], [0])

    with pytest.raises(ValueError, match="losing integer precision"):
        tree.partial_fit([[float(boundary)]], [1])
    np.testing.assert_array_equal(tree.predict([[boundary]]), [7])

    tree.partial_fit([[boundary]], [1])
    np.testing.assert_array_equal(tree.predict([[boundary], [boundary + 1]]), [1, 0])


def test_tree_rejects_invalid_buffer_size():
    for buffer_size in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="buffer_size"):
            DecisionTreeClassifier(buffer_size=buffer_size)


def test_partial_fit_rejects_integer_rounding_past_dtype_limit():
    for dtype in (np.int64, np.uint64):
        maximum = int(np.iinfo(dtype).max)
        tree = DecisionTreeClassifier(buffer_size=2).fit([[0]], [7])
        tree.partial_fit(np.array([[maximum]], dtype=dtype), [0])

        with pytest.raises(ValueError, match="losing integer precision"):
            tree.partial_fit([[float(maximum + 1)]], [1])
        np.testing.assert_array_equal(tree.predict([[0]]), [7])

        X = np.array([[maximum - 1], [maximum]], dtype=dtype)
        tree.partial_fit(X[:1], [1])
        np.testing.assert_array_equal(tree.predict(X), [1, 0])


def test_partial_fit_accepts_exact_integer_values_in_mixed_dtypes():
    lower = -(2**63)
    tree = DecisionTreeClassifier(buffer_size=2)
    tree.partial_fit(np.array([[lower]], dtype=np.int64), [0])
    tree.partial_fit([[0.0]], [1])

    np.testing.assert_array_equal(tree.predict([[float(lower)], [0.0]]), [0, 1])


def test_partial_fit_rejects_mixed_byte_and_unicode_labels_before_buffering():
    tree = DecisionTreeClassifier(buffer_size=3).fit([[0]], [b"old"])
    tree.partial_fit([[0]], [b"\xff"])

    with pytest.raises(ValueError, match="cannot mix byte and Unicode labels"):
        tree.partial_fit([[1]], ["new"])
    np.testing.assert_array_equal(tree.predict([[0]]), [b"old"])

    tree.partial_fit([[1]], [b"new"])
    np.testing.assert_array_equal(tree.predict([[0]]), [b"old"])
    tree.partial_fit([[2]], [b"new"])
    np.testing.assert_array_equal(tree.predict([[0], [1], [2]]), [b"\xff", b"new", b"new"])
