
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