"""Unit tests for classification tree split selection."""

import numpy as np
import pytest

from numcompute_stream.tree import _best_split, _impurity


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
