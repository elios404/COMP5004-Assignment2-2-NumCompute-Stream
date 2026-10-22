
import unittest

from unittest.mock import patch

from numcompute.benchmarking import (
    benchmark,
    compare_implementations
)


class TestBenchmark(unittest.TestCase):

    def test_benchmark_normal(self):
        def square(x):
            return x * x

        result = benchmark(square, 5, repeat=3, number=10)

        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)

    def test_benchmark_execution_count(self):
        counter = {"count": 0}

        def counting_function():
            counter["count"] += 1

        benchmark(counting_function, repeat=3, number=4)

        self.assertEqual(counter["count"], 12)

    def test_benchmark_keyword_arguments(self):
        def add(a, b=0):
            return a + b

        result = benchmark(
            add, 5, b=10, repeat=2, number=5
        )

        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)

    def test_benchmark_invalid_repeat(self):
        def sample():
            return 10

        with self.assertRaises(ValueError):
            benchmark(sample, repeat=0)

        with self.assertRaises(TypeError):
            benchmark(sample, repeat=2.5)

    def test_benchmark_invalid_function(self):
        with self.assertRaises(TypeError):
            benchmark(123)


class TestCompareImplementations(unittest.TestCase):

    def test_compare_normal(self):
        def loop_fn(x):
            return x * 2

        def vec_fn(x):
            return x * 2

        result = compare_implementations(
            loop_fn, vec_fn, 5, repeat=2
        )

        self.assertIn("loop_time", result)
        self.assertIn("vec_time", result)
        self.assertIn("speedup", result)

        self.assertGreaterEqual(result["loop_time"], 0)
        self.assertGreaterEqual(result["vec_time"], 0)
        self.assertGreater(result["speedup"], 0)

    def test_compare_speedup_calculation(self):
        with patch(
            "numcompute.benchmarking.benchmark",
            side_effect=[0.02, 0.005]
        ):
            result = compare_implementations(
                lambda: None,
                lambda: None
            )

        self.assertAlmostEqual(result["loop_time"], 0.02)
        self.assertAlmostEqual(result["vec_time"], 0.005)
        self.assertAlmostEqual(result["speedup"], 4.0)

    def test_compare_invalid_function(self):
        def sample():
            return 10

        with self.assertRaises(TypeError):
            compare_implementations(123, sample)

        with self.assertRaises(TypeError):
            compare_implementations(sample, 456)

    def test_compare_invalid_repeat(self):
        def sample():
            return 10

        with self.assertRaises(ValueError):
            compare_implementations(
                sample, sample, repeat=0
            )

        with self.assertRaises(TypeError):
            compare_implementations(
                sample, sample, repeat=2.5
            )

if __name__ == "__main__":
    unittest.main()
