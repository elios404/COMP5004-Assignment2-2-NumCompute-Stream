import unittest
import numpy as np

from numcompute.utils import (euclidean_distance, cosine_similarity, softmax, logsumexp, batch_generator)


class TestEuclideanDistance(unittest.TestCase):

    def test_euclidean_distance_normal(self):
        X = np.array([[0, 0], [3, 4]])
        Y = np.array([[3, 4]])

        result = euclidean_distance(X, Y)

        expected = np.array([[5.0], [0.0]])

        np.testing.assert_allclose(result, expected)

    
    def test_euclidean_mismatched_features(self):
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([[1, 2, 3]])

        with self.assertRaises(ValueError):
            euclidean_distance(X, Y)

    def test_euclidean_nan(self):
        X = np.array([[1.0, np.nan]])
        Y = np.array([[2.0, 3.0]])

        with self.assertRaises(ValueError):
            euclidean_distance(X, Y)

    def test_euclidean_non_contiguous(self):
        original = np.array([
            [0, 99, 0, 99],
            [3, 99, 4, 99]
        ])

        X = original[:, ::2]
        Y = np.array([[3, 4]])

        result = euclidean_distance(X, Y)

        expected = np.array([[5.0], [0.0]])

        np.testing.assert_allclose(result, expected)


class TestCosineSimilarity(unittest.TestCase):

    def test_cosine_same_direction(self):
        X = np.array([[1, 2]])
        Y = np.array([[2, 4]])

        result = cosine_similarity(X, Y)

        expected = np.array([[1.0]])

        np.testing.assert_allclose(result, expected)

    def test_cosine_opposite_direction(self):
        X = np.array([[1, 0]])
        Y = np.array([[-1, 0]])

        result = cosine_similarity(X, Y)

        expected = np.array([[-1.0]])

        np.testing.assert_allclose(result, expected)

    def test_cosine_zero_vector(self):
        X = np.array([[0, 0]])
        Y = np.array([[3, 4]])

        result = cosine_similarity(X, Y)

        expected = np.array([[0.0]])

        np.testing.assert_allclose(result, expected)

    def test_cosine_mismatched_features(self):
        X = np.array([[1, 2]])
        Y = np.array([[1, 2, 3]])

        with self.assertRaises(ValueError):
            cosine_similarity(X, Y)


class TestSoftmax(unittest.TestCase):

    def test_softmax_normal(self):
        x = np.array([1.0, 2.0, 3.0])

        result = softmax(x)

        self.assertAlmostEqual(np.sum(result), 1.0)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))

    def test_softmax_large_values(self):
        x = np.array([1000.0, 1001.0, 1002.0])

        result = softmax(x)

        shifted = np.array([-2.0, -1.0, 0.0])
        expected = np.exp(shifted) / np.sum(np.exp(shifted))

        np.testing.assert_allclose(result, expected)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_softmax_axis(self):
        x = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])

        result = softmax(x, axis=1)

        self.assertEqual(result.shape, x.shape)
        np.testing.assert_allclose(
            np.sum(result, axis=1),
            np.array([1.0, 1.0])
        )

    def test_softmax_equal_values(self):
        x = np.array([5.0, 5.0, 5.0])

        result = softmax(x)

        expected = np.array([1/3, 1/3, 1/3])

        np.testing.assert_allclose(result, expected)

    def test_softmax_nan(self):
        x = np.array([1.0, np.nan, 3.0])

        with self.assertRaises(ValueError):
            softmax(x)


class TestLogSumExp(unittest.TestCase):

    def test_logsumexp_normal(self):
        x = np.array([1.0, 2.0, 3.0])

        result = logsumexp(x)

        expected = np.log(np.sum(np.exp(x)))

        self.assertAlmostEqual(float(result), expected)

    def test_logsumexp_large_values(self):
        x = np.array([1000.0, 1001.0, 1002.0])

        result = logsumexp(x)

        expected = 1002 + np.log(
            np.exp(-2) + np.exp(-1) + np.exp(0)
        )

        self.assertAlmostEqual(float(result), expected)
        self.assertTrue(np.isfinite(result))

    def test_logsumexp_negative_values(self):
        x = np.array([-3.0, -2.0, -1.0])

        result = logsumexp(x)

        expected = np.log(np.sum(np.exp(x)))

        self.assertAlmostEqual(float(result), expected)

    def test_logsumexp_axis(self):
        x = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])

        result = logsumexp(x, axis=1)

        expected = np.array([
            3.40760596,
            6.40760596
        ])

        np.testing.assert_allclose(
            result, expected, rtol=1e-7
        )

    def test_logsumexp_nan(self):
        x = np.array([1.0, np.nan, 3.0])

        with self.assertRaises(ValueError):
            logsumexp(x)


class TestBatchGenerator(unittest.TestCase):

    def test_batch_generator_normal(self):
        X = np.arange(10)

        batches = list(
            batch_generator(X, batch_size=4, shuffle=False)
        )

        self.assertEqual(len(batches), 3)

        np.testing.assert_array_equal(
            batches[0][0], np.array([0, 1, 2, 3])
        )

        np.testing.assert_array_equal(
            batches[1][0], np.array([4, 5, 6, 7])
        )

        np.testing.assert_array_equal(
            batches[2][0], np.array([8, 9])
        )

    def test_batch_generator_with_labels(self):
        X = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 20, 30, 40, 50])

        batches = list(
            batch_generator(X, y, batch_size=2, shuffle=False)
        )

        self.assertEqual(len(batches), 3)

        np.testing.assert_array_equal(
            batches[0][1], np.array([10, 20])
        )

    def test_batch_generator_shuffle(self):
        X = np.arange(10)
        y = X * 10

        batches = list(
            batch_generator(X, y, batch_size=3, shuffle=True)
        )

        X_combined = np.concatenate([b[0] for b in batches])
        y_combined = np.concatenate([b[1] for b in batches])

        np.testing.assert_array_equal(
            np.sort(X_combined), X
        )

        np.testing.assert_array_equal(
            y_combined, X_combined * 10
        )

    def test_batch_generator_invalid_size(self):
        X = np.arange(10)

        with self.assertRaises(ValueError):
            list(batch_generator(X, batch_size=0))

    def test_batch_generator_mismatched_labels(self):
        X = np.arange(10)
        y = np.arange(5)

        with self.assertRaises(ValueError):
            list(batch_generator(X, y, batch_size=2))


if __name__ == "__main__":
    unittest.main()
