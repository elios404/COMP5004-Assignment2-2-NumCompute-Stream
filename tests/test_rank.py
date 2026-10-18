import numpy as np
import pytest

from numcompute.rank import percentile, rank

class TestRank:
    def test_average_method_with_ties(self):
        data = np.array([3, 1, 2, 2])
        out = rank(data, method="average")
        np.testing.assert_allclose(out, [4.0, 1.0, 2.5, 2.5])

    def test_ordinal_method_breaks_ties_by_position(self):
        data = np.array([3, 1, 2, 2])
        out = rank(data, method="ordinal")
        np.testing.assert_allclose(out, [4.0, 1.0, 2.0, 3.0])

    def test_no_ties_all_methods_agree(self):
        data = np.array([10, 30, 20, 40])
        expected = [1.0, 3.0, 2.0, 4.0]
        for method in ("average", "ordinal", "dense"):
            np.testing.assert_allclose(rank(data, method=method), expected)

    def test_axis_rows(self):
        data = np.array([[3, 1, 2], [1, 1, 2]])
        out = rank(data, method="average", axis=-1)
        np.testing.assert_allclose(out[0], [3.0, 1.0, 2.0])
        np.testing.assert_allclose(out[1], [1.5, 1.5, 3.0])

    def test_axis_columns(self):
        data = np.array([[3, 1], [1, 2], [2, 3]])
        out = rank(data, method="average", axis=0)
        np.testing.assert_allclose(out[:, 0], [3.0, 1.0, 2.0])
        np.testing.assert_allclose(out[:, 1], [1.0, 2.0, 3.0])

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            rank(np.array([1, 2, 3]), method="invalid_method")

    def test_empty_array(self):
        for method in ("average", "ordinal", "dense"):
            out = rank(np.array([]), method=method)
            assert out.size == 0

    def test_empty_axis_multidimensional(self):
        data = np.empty((3, 0))
        for method in ("average", "ordinal", "dense"):
            out = rank(data, method=method, axis=1)
            assert out.shape == (3, 0)

    def test_all_equal_values(self):
        data = np.array([5.0, 5.0, 5.0, 5.0])
        n = data.size
        np.testing.assert_allclose(
            rank(data, method="average"), [(n + 1) / 2.0] * n
        )
        np.testing.assert_allclose(rank(data, method="dense"), [1.0] * n)
        np.testing.assert_allclose(rank(data, method="ordinal"), [1.0, 2.0, 3.0, 4.0])

    def test_non_contiguous_input(self):
        backing = np.array([[3, 1, 2, 2], [9, 9, 9, 9]])
        strided = backing[:, ::2] 
        assert not strided.flags["C_CONTIGUOUS"]
        out = rank(strided, method="average", axis=-1)
        np.testing.assert_allclose(out[0], [2.0, 1.0])
        np.testing.assert_allclose(out[1], [1.5, 1.5])

    def test_nan_no_change(self):
        data = np.array([np.nan, np.nan, np.nan])
        for method in ("average", "ordinal", "dense"):
            out = rank(data, method=method)
            assert np.all(np.isnan(out))


class TestPercentile:
    def test_sequence_of_quantiles(self):
        data = np.array([5, 1, 9, 3, 7])
        got = percentile(data, [0, 50, 100])
        expected = np.percentile(data, [0, 50, 100])
        np.testing.assert_allclose(got, expected)

    def test_axis_rows(self):
        data = np.array([[1, 2, 3], [4, 5, 6]])
        got = percentile(data, 50, axis=1)
        expected = np.percentile(data, 50, axis=1)
        np.testing.assert_allclose(got, expected)

    def test_flattened_when_axis_none(self):
        data = np.array([[1, 2], [3, 4]])
        got = percentile(data, 50, axis=None)
        expected = np.percentile(data, 50, axis=None)
        assert got == pytest.approx(expected)

    def test_scalar_q_returns_scalar(self):
        data = np.array([1, 2, 3])
        got = percentile(data, 50)
        assert np.isscalar(got) or got.shape == ()

    def test_out_of_range_q_raises(self):
        with pytest.raises(ValueError):
            percentile(np.array([1, 2, 3]), 150)

    def test_nan_q_raises(self):
        with pytest.raises(ValueError):
            percentile(np.array([1, 2, 3]), np.nan)

    def test_invalid_interpolation_raises(self):
        with pytest.raises(ValueError):
            percentile(np.array([1, 2, 3]), 50, interpolation="invalid")

    def test_empty_array_returns_nan(self):
        assert np.isnan(percentile(np.array([]), 50))

    def test_all_equal_values(self):
        data = np.array([4.0, 4.0, 4.0, 4.0])
        for q in (0, 25, 50, 75, 100):
            assert percentile(data, q) == 4.0

    def test_non_contiguous_input(self):
        backing = np.array([9, 1, 8, 2, 7, 3, 6, 4])
        strided = backing[::2]  # [9, 8, 7, 6]
        assert not strided.flags["C_CONTIGUOUS"]
        got = percentile(strided, 50)
        expected = np.percentile(strided, 50)
        assert got == pytest.approx(expected)

    def test_all_nan_slice_returns_nan(self):
        data = np.array([np.nan, np.nan])
        for q in (0, 50, 100):
            assert np.isnan(percentile(data, q))

    def test_nan_dropped_per_row_with_axis(self):
        data = np.array([[1.0, 2.0, np.nan, 4.0, 5.0], [np.nan, np.nan, np.nan, np.nan, np.nan]])
        out = percentile(data, 50, axis=1)
        assert out[0] == pytest.approx(3.0)
        assert np.isnan(out[1])
