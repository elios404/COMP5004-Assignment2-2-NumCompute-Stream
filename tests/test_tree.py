"""Unit tests for classification trees."""

import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier, _best_split, _impurity


def test_impurity_for_pure_and_balanced_labels():
    pure = np.array([1, 1, 1, 1])
    balanced = np.array([0, 0, 1, 1])

    assert _impurity(pure, "gini") == 0.0
    assert _impurity(pure, "entropy") == 0.0
    assert _impurity(balanced, "gini") == 0.5
    assert _impurity(balanced, "entropy") == 1.0


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
