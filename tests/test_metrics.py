
import unittest
import numpy as np

from numcompute.metrics import (
    accuracy, precision, recall, f1, confusion_matrix, mse, roc_curve, auc
)


class TestAccuracy(unittest.TestCase):

    def test_accuracy_normal(self):
        y_true = np.array([1, 0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0])

        result = accuracy(y_true, y_pred)

        self.assertAlmostEqual(result, 0.8)

    
    def test_accuracy_perfect(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])

        result = accuracy(y_true, y_pred)

        self.assertAlmostEqual(result, 1.0)

    def test_accuracy_empty(self):
        y_true = np.array([])
        y_pred = np.array([])

        with self.assertRaises(ValueError):
            accuracy(y_true, y_pred)

    
    def test_accuracy_mismatched_shapes(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1])

        with self.assertRaises(ValueError):
            accuracy(y_true, y_pred)
        
    
class TestPrecision(unittest.TestCase):

    def test_precision_normal(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 1, 1, 0, 0])

        result = precision(y_true, y_pred)

        self.assertAlmostEqual(result, 2 / 3)

    
    def test_precision_zero_division(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([0, 0, 0, 0])

        result = precision(y_true, y_pred)

        self.assertAlmostEqual(result, 0.0)

    
    def test_precision_mismatched_shapes(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0])

        with self.assertRaises(ValueError):
            precision(y_true, y_pred)

    def test_precision_invalid_labels(self):
        y_true = np.array([1, 0, 2, 1])
        y_pred = np.array([1, 0, 1, 1])

        with self.assertRaises(ValueError):
            precision(y_true, y_pred)


class TestRecall(unittest.TestCase):

    def test_recall_normal(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1])

        result = recall(y_true, y_pred)

        self.assertAlmostEqual(result, 2 / 3)

    
    def test_recall_zero_division(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 1, 0, 1])

        result = recall(y_true, y_pred)

        self.assertAlmostEqual(result, 0.0)

    def test_recall_mismatched_shapes(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0])

        with self.assertRaises(ValueError):
            recall(y_true, y_pred)

    def test_recall_invalid_labels(self):
        y_true = np.array([1, 0, 2, 1])
        y_pred = np.array([1, 0, 1, 1])

        with self.assertRaises(ValueError):
            recall(y_true, y_pred)


class TestF1(unittest.TestCase):

    def test_f1_normal(self):
        y_true = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1])

        result = f1(y_true, y_pred)

        self.assertAlmostEqual(result, 2 / 3)

    def test_f1_zero_division(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])

        result = f1(y_true, y_pred)

        self.assertAlmostEqual(result, 0.0)



class TestConfusionMatrix(unittest.TestCase):

    def test_confusion_matrix_binary(self):
        y_true = np.array([0, 0, 1, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 0])

        expected = np.array([
            [2, 1],
            [1, 2]
        ])

        result = confusion_matrix(y_true, y_pred)

        np.testing.assert_array_equal(result, expected)

    def test_confusion_matrix_multiclass(self):
        y_true = np.array([0, 1, 2, 2, 1])
        y_pred = np.array([0, 2, 2, 1, 1])

        expected = np.array([
            [1, 0, 0],
            [0, 1, 1],
            [0, 1, 1]
        ])

        result = confusion_matrix(y_true, y_pred)

        np.testing.assert_array_equal(result, expected)

class TestMSE(unittest.TestCase):

    def test_mse_normal(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 3, 5])

        result = mse(y_true, y_pred)

        self.assertAlmostEqual(result, 5 / 3)

    def test_mse_perfect(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([10, 20, 30])

        result = mse(y_true, y_pred)

        self.assertAlmostEqual(result, 0.0)

    def test_mse_mismatched_shapes(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])

        with self.assertRaises(ValueError):
            mse(y_true, y_pred)

    def test_mse_nan(self):
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        with self.assertRaises(ValueError):
            mse(y_true, y_pred)
   

class TestROCCurve(unittest.TestCase):

    def test_roc_curve_normal(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])

        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        expected_fpr = np.array([0, 0, 0.5, 0.5, 1])
        expected_tpr = np.array([0, 0.5, 0.5, 1, 1])

        np.testing.assert_allclose(fpr, expected_fpr)
        np.testing.assert_allclose(tpr, expected_tpr)

        self.assertTrue(np.isinf(thresholds[0]))
        self.assertEqual(len(thresholds), 5)


class TestAUC(unittest.TestCase):

    def test_auc_normal(self):
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, 0.5, 1.0])

        result = auc(x, y)

        self.assertAlmostEqual(result, 0.5)

    def test_auc_from_roc(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.4, 0.35, 0.8])

        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        result = auc(fpr, tpr)

        self.assertAlmostEqual(result, 0.75)


if __name__ == "__main__":
    unittest.main()
