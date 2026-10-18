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

    def test_empty_array(self):
        out = stable_sort(np.array([]))
        assert out.size == 0

    def test_all_equal_values(self):
        arr = np.array([7, 7, 7, 7])
        out = stable_sort(arr)
        np.testing.assert_array_equal(out, [7, 7, 7, 7])

    def test_non_contiguous_input(self):
        backing = np.array([9, 1, 8, 2, 7, 3, 6, 4])
        strided = backing[::2]
        assert not strided.flags["C_CONTIGUOUS"]
        out = stable_sort(strided)
        np.testing.assert_array_equal(out, [6, 7, 8, 9])


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

    def test_empty_rows(self):
        arr = np.empty((0, 2), dtype=int)
        out = multi_key_sort(arr, columns=[0, 1])
        assert out.shape == (0, 2)


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

    def test_k_equals_n_returns_full_sorted_array(self):
        values = np.array([5, 1, 9])
        vals, idx = topk(values, k=3, largest=True)
        np.testing.assert_array_equal(vals, [9, 5, 1])
        np.testing.assert_array_equal(values[idx], vals)

    def test_k_larger_than_length_clips(self):
        values = np.array([5, 1, 9])
        vals, idx = topk(values, k=10, largest=True)
        np.testing.assert_array_equal(vals, [9, 5, 1])

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            topk(np.array([1, 2, 3]), k=-1)

    def test_empty_array(self):
        vals, idx = topk(np.array([]), k=5)
        assert vals.size == 0
        assert idx.size == 0

    def test_all_equal_values(self):
        values = np.array([4, 4, 4, 4])
        vals, idx = topk(values, k=2, largest=True)
        np.testing.assert_array_equal(vals, [4, 4])
        np.testing.assert_array_equal(values[idx], vals)

    def test_non_contiguous_input(self):
        backing = np.array([9, 1, 8, 2, 7, 3, 6, 4])
        strided = backing[::2]  # [9, 8, 7, 6]
        vals, idx = topk(strided, k=2, largest=True)
        np.testing.assert_array_equal(vals, [9, 8])
        np.testing.assert_array_equal(strided[idx], vals)

    def test_nan_counts_as_largest(self):
        values = np.array([3.0, np.nan, 1.0, 2.0])
        vals, _ = topk(values, k=1, largest=True)
        assert np.isnan(vals[0])


class TestQuickselect:
    def test_kth_smallest_matches_sorted(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(-100, 100, size=50)
        sorted_arr = np.sort(arr)
        for k in range(len(arr)):
            assert quickselect(arr, k) == pytest.approx(sorted_arr[k])

    def test_with_duplicates(self):
        arr = np.array([2, 2, 2, 1, 3])
        assert quickselect(arr, 0) == 1
        assert quickselect(arr, 4) == 3

    def test_out_of_bounds_raises(self):
        with pytest.raises(ValueError):
            quickselect(np.array([1, 2, 3]), 3)
        with pytest.raises(ValueError):
            quickselect(np.array([1, 2, 3]), -1)

    def test_empty_array_raises(self):
        with pytest.raises(ValueError):
            quickselect(np.array([]), 0)

    def test_all_equal_values(self):
        arr = np.array([5, 5, 5, 5])
        assert quickselect(arr, 2) == 5

    def test_extreme_k_first_and_last(self):
        arr = np.array([5, 1, 9, 3, 7])
        assert quickselect(arr, 0) == 1
        assert quickselect(arr, arr.size - 1) == 9

    def test_non_contiguous_input(self):
        backing = np.array([9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0])
        strided = backing[::2] 
        assert not strided.flags["C_CONTIGUOUS"]
        assert quickselect(strided, 0) == 6.0
        assert quickselect(strided, 3) == 9.0

    def test_nan_sorts_as_largest(self):
        arr = np.array([3.0, np.nan, 1.0, 2.0])
        assert np.isnan(quickselect(arr, 3))
        assert quickselect(arr, 0) == 1.0


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

    def test_empty_array(self):
        idx, exists = binary_search(np.array([]), 5)
        assert idx == 0
        assert exists is False

    def test_all_equal_values_finds_leftmost(self):
        arr = np.array([2, 2, 2, 2])
        idx, exists = binary_search(arr, 2)
        assert idx == 0
        assert exists is True

    def test_non_contiguous_input(self):
        backing = np.array([1, 10, 3, 20, 5, 30, 7])
        strided = backing[::2] 
        assert not strided.flags["C_CONTIGUOUS"]
        idx, exists = binary_search(strided, 5)
        assert idx == 2
        assert exists is True
