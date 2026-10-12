import numpy as np
import pytest

from numcompute.sort_search import (
    binary_search,
    multi_key_sort,
    quickselect,
    stable_sort,
    topk,
)


class TestStableSort:
    def test_sorts_ascending(self):
        arr = np.array([3, 1, 2])
        out = stable_sort(arr)
        np.testing.assert_array_equal(out, [1, 2, 3])

    def test_preserves_relative_order_of_ties(self):
        dtype = [("key", int), ("tag", int)]
        arr = np.array([(1, 0), (0, 1), (1, 2)], dtype=dtype)
        out = stable_sort(arr, axis=0)
        np.testing.assert_array_equal(out["key"], [0, 1, 1])
        np.testing.assert_array_equal(out["tag"], [1, 0, 2])

    def test_axis(self):
        arr = np.array([[3, 1], [2, 4]])
        out = stable_sort(arr, axis=0)
        np.testing.assert_array_equal(out, [[2, 1], [3, 4]])


class TestMultiKeySort:
    def test_sort_by_two_columns_ascending(self):
        arr = np.array([
            [1, 5],
            [0, 2],
            [1, 2],
            [0, 9],
        ])
        out = multi_key_sort(arr, columns=[0, 1])
        expected = np.array([
            [0, 2],
            [0, 9],
            [1, 2],
            [1, 5],
        ])
        np.testing.assert_array_equal(out, expected)

    def test_sort_descending(self):
        arr = np.array([
            [1, 5],
            [0, 2],
            [1, 2],
        ])
        out = multi_key_sort(arr, columns=[0, 1], ascending=False)
        expected = np.array([
            [1, 5],
            [1, 2],
            [0, 2],
        ])
        np.testing.assert_array_equal(out, expected)

    def test_single_column_matches_stable_sort(self):
        arr = np.array([[2], [1], [3]])
        out = multi_key_sort(arr, columns=[0])
        np.testing.assert_array_equal(out, [[1], [2], [3]])


class TestTopK:
    def test_largest_values_and_indices(self):
        values = np.array([5, 1, 9, 3, 7])
        vals, idx = topk(values, k=3, largest=True, return_indices=True)
        np.testing.assert_array_equal(vals, [9, 7, 5])
        np.testing.assert_array_equal(idx, [2, 4, 0])
        np.testing.assert_array_equal(values[idx], vals)

    def test_smallest_values(self):
        values = np.array([5, 1, 9, 3, 7])
        vals, idx = topk(values, k=2, largest=False)
        np.testing.assert_array_equal(vals, [1, 3])
        np.testing.assert_array_equal(idx, [1, 3])

    def test_return_indices_false(self):
        values = np.array([5, 1, 9, 3, 7])
        vals, idx = topk(values, k=2, return_indices=False)
        assert idx is None
        np.testing.assert_array_equal(vals, [9, 7])

    def test_k_zero_returns_empty(self):
        values = np.array([5, 1, 9])
        vals, idx = topk(values, k=0)
        assert vals.size == 0
        assert idx.size == 0

    def test_k_larger_than_length_clips(self):
        values = np.array([5, 1, 9])
        vals, idx = topk(values, k=10, largest=True)
        np.testing.assert_array_equal(vals, [9, 5, 1])

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            topk(np.array([1, 2, 3]), k=-1)


class TestQuickselect:
    def test_kth_smallest_matches_sorted(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(-100, 100, size=50)
        sorted_arr = np.sort(arr)
        for k in range(len(arr)):
            assert quickselect(arr, k) == pytest.approx(sorted_arr[k])

    def test_does_not_mutate_input(self):
        arr = np.array([3.0, 1.0, 2.0])
        original = arr.copy()
        quickselect(arr, 1)
        np.testing.assert_array_equal(arr, original)

    def test_single_element(self):
        assert quickselect(np.array([42.0]), 0) == 42.0

    def test_with_duplicates(self):
        arr = np.array([2, 2, 2, 1, 3])
        assert quickselect(arr, 0) == 1
        assert quickselect(arr, 4) == 3

    def test_out_of_bounds_raises(self):
        with pytest.raises(ValueError):
            quickselect(np.array([1, 2, 3]), 3)
        with pytest.raises(ValueError):
            quickselect(np.array([1, 2, 3]), -1)


class TestBinarySearch:
    def test_exact_match(self):
        arr = np.array([1, 3, 5, 7, 9])
        idx, exists = binary_search(arr, 5)
        assert idx == 2
        assert exists is True

    def test_not_found_insertion_point(self):
        arr = np.array([1, 3, 5, 7, 9])
        idx, exists = binary_search(arr, 4)
        assert idx == 2
        assert exists is False

    def test_before_first_element(self):
        arr = np.array([1, 3, 5])
        idx, exists = binary_search(arr, 0)
        assert idx == 0
        assert exists is False

    def test_after_last_element(self):
        arr = np.array([1, 3, 5])
        idx, exists = binary_search(arr, 10)
        assert idx == 3
        assert exists is False

    def test_returns_python_bool(self):
        arr = np.array([1, 3, 5])
        _, exists = binary_search(arr, 3)
        assert isinstance(exists, bool)
