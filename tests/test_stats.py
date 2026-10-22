
import numpy as np
from numcompute.stats import mean, std, median, minimum, maximum, histogram, quantiles
from numcompute.stats import  WelfordStats


def test_mean_basic():
    arr = np.array([2, 4, 6, 8])

    expected = np.mean(arr)
    actual = mean(arr)

    assert actual == expected

def test_std_basic():
    arr = np.array([2, 4, 6, 8])

    expected = np.std(arr)
    actual = std(arr)

    assert actual == expected

def test_std_default_ddof():
    arr = np.array([1,2,3,4,5])

    default_result = std(arr)
    population_std = std(arr, ddof=0)
    sample_std = std(arr, ddof=1)

    assert default_result == population_std
    assert default_result != sample_std

def test_median_wrapper():
    arr = np.array([2, 4, 6, 8])

    expected = np.median(arr)
    actual = median(arr)

    assert actual == expected

def test_median_axis():
    arr = np.array([[2, 4, 6, 8], 
                   [1, 3, 7, 9]])

    expected = np.median(arr, axis=1)
    actual = median(arr, axis=1)

    assert np.array_equal(actual, expected)


def test_min_wrapper():
    arr = np.array([2, 4, 6, 8])

    expected = np.min(arr)
    actual = minimum(arr)

    assert actual == expected

def test_min_axis():
    arr = np.array([[2, 4, 6, 8], 
                    [1, 3, 7, 9]])
    
    expected = np.min(arr, axis=1)
    actual = minimum(arr, axis=1)
    
    assert np.array_equal(actual, expected)
    
    
def test_max_wrapper():
    arr = np.array([2, 4, 6, 8])

    expected = np.max(arr)
    actual = maximum(arr)

    assert actual == expected

def test_max_axis():
    arr = np.array([[2, 4, 6, 8], 
                    [1, 3, 7, 9]])
    
    expected = np.max(arr, axis=1)
    actual = maximum(arr, axis=1)
    
    assert np.array_equal(actual, expected)

def test_histogram():
    data = np.array([[1,2,3,4],
                     [5,6,7,8]])

    expected_counts, expected_edges = np.histogram(data, bins=3)
    actual_counts, actual_edges = histogram(data, bins=3)

    assert np.array_equal(actual_counts, expected_counts)
    assert np.array_equal(actual_edges, expected_edges)


def test_histogram_custom_range():
    data = np.array([[1,2,3,4],
                     [5,6,7,8]])

    expected_counts, expected_edges = np.histogram(data, bins=3, range=(0,6))
    actual_counts, actual_edges = histogram(data, bins=3, bin_range=(0,6))

    assert np.array_equal(actual_counts, expected_counts)
    assert np.array_equal(actual_edges, expected_edges)

def test_quantiles():
    data = np.array([10,20,30,40,50])

    expected = np.quantile(data, 0.5)
    actual = quantiles(data, 0.5)

    assert actual == expected

def test_quantiles_multiple():
    data = np.array([10,20,30,40,50])
    q = [0.25, 0.5, 0.75]

    expected = np.quantile(data, q)
    actual = quantiles(data, q)

    np.testing.assert_allclose(actual, expected)

def test_quantiles_axis():
    data = np.array([[10,20,30], 
                    [40,50,60]])

    expected = np.quantile(data, 0.5, axis=1)
    actual = quantiles(data, 0.5, axis=1)

    assert np.array_equal(actual, expected)

def test_quantiles_nan_handled():
    data = np.array([10, 20, np.nan, 40, 50])

    expected = np.nanquantile(data, 0.5)
    actual = quantiles(data, 0.5, handle_nan=True)

    assert actual == expected

def test_quantiles_nan_not_handled():
    data = np.array([10, 20, np.nan, 40, 50])

    actual = quantiles(data, 0.5, handle_nan=False)

    assert np.isnan(actual)

def testing_welford_mean():
    data = np.array([1, 3, 5, 7, 9])
    compute_mean = WelfordStats()

    compute_mean.update(data)

    assert np.isclose(compute_mean.mean_, np.mean(data))

def testing_welford_sampleVariance():
    data = np.array([1, 3, 5, 7, 9])
    var = WelfordStats()

    var.update(data)

    expected = np.var(data, ddof=1)

    assert np.isclose(var.variance, expected)


# Additional statistics tests (14 cases)

def test_mean_axis_zero():
    arr = np.array([[1, 2, 3], [4, 5, 6]])

    actual = mean(arr, axis=0)

    np.testing.assert_allclose(actual, [2.5, 3.5, 4.5])


def test_mean_keepdims():
    arr = np.array([[1, 2], [3, 4]])

    actual = mean(arr, axis=1, keepdims=True)

    np.testing.assert_allclose(actual, [[1.5], [3.5]])
    assert actual.shape == (2, 1)


def test_std_axis_zero():
    arr = np.array([[1, 2], [3, 4]])

    actual = std(arr, axis=0)

    np.testing.assert_allclose(actual, [1.0, 1.0])


def test_std_keepdims():
    arr = np.array([[1, 3], [5, 7]])

    actual = std(arr, axis=1, keepdims=True)

    np.testing.assert_allclose(actual, [[1.0], [1.0]])
    assert actual.shape == (2, 1)


def test_median_odd_number_of_values():
    arr = np.array([9, 1, 5, 3, 7])

    assert median(arr) == 5


def test_minimum_negative_values():
    arr = np.array([-8, -2, -5, 0])

    assert minimum(arr) == -8


def test_maximum_negative_values():
    arr = np.array([-8, -2, -5, -1])

    assert maximum(arr) == -1


def test_histogram_counts_all_values_in_range():
    data = np.array([1, 2, 3, 4, 5])

    counts, edges = histogram(data, bins=5, bin_range=(1, 5))

    assert counts.sum() == len(data)
    assert len(edges) == 6


def test_quantiles_lower_and_upper():
    data = np.array([10, 20, 30, 40, 50])

    actual = quantiles(data, [0.0, 1.0])

    np.testing.assert_allclose(actual, [10, 50])


def test_quantiles_keep_nan_handling():
    data = np.array([1.0, np.nan, 3.0])

    actual = quantiles(data, 0.5, handle_nan=True)

    assert actual == 2.0


def test_welford_updates_in_batches():
    first_batch = np.array([1.0, 2.0, 3.0])
    second_batch = np.array([4.0, 5.0])
    stats = WelfordStats()

    stats.update(first_batch)
    stats.update(second_batch)

    expected = np.mean([1.0, 2.0, 3.0, 4.0, 5.0])

    assert np.isclose(stats.mean_, expected)


def test_welford_single_update_at_a_time():
    data = [2.0, 4.0, 6.0]
    stats = WelfordStats()

    for value in data:
        stats.update(np.array([value]))

    assert np.isclose(stats.mean_, np.mean(data))


def test_welford_variance_after_multiple_batches():
    stats = WelfordStats()
    stats.update(np.array([1.0, 2.0]))
    stats.update(np.array([3.0, 4.0]))

    expected = np.var([1.0, 2.0, 3.0, 4.0], ddof=1)

    assert np.isclose(stats.variance, expected)


def test_welford_mean_with_negative_values():
    data = np.array([-5.0, -1.0, 3.0])
    stats = WelfordStats()

    stats.update(data)

    assert np.isclose(stats.mean_, np.mean(data))
