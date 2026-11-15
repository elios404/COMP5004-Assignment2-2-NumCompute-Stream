"""Integration tests for chunk-wise evaluation and training."""

import tracemalloc

import numpy as np
import pytest

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier


def test_trainer_warms_up_and_scores_before_retraining():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(buffer_size=2)),
    ])
    trainer = StreamTrainer(pipe, window_size=2)
    first = trainer.fit_chunk([[0]], [0])
    second = trainer.fit_chunk([[1]], [1])

    assert not first["evaluated"] and not second["evaluated"]
    assert first["accuracy"] is None and np.isnan(second["cumulative_accuracy"])
    assert not first["retrained"] and second["retrained"]

    third = trainer.fit_chunk([[0]], [1])
    fourth = trainer.fit_chunk([[1]], [0])

    assert third["accuracy"] == fourth["accuracy"] == 0
    assert not third["retrained"] and fourth["retrained"]
    assert fourth["fit_seconds"] >= 0
    assert fourth["peak_traced_bytes"] > 0
    assert trainer.metrics.result()["count"] == 2
    assert trainer.metrics.result()["auc"] == 0
    assert trainer.metrics.rolling_result()["count"] == 2
    np.testing.assert_array_equal(trainer.last_evaluation["y_pred"], [1])
    np.testing.assert_array_equal(pipe.predict([[0], [1]]), [1, 0])


def test_score_chunk_is_read_only_and_uses_configured_positive_label():
    pipe = Pipeline([("model", DecisionTreeClassifier(max_depth=0))])
    pipe.fit([[0], [1], [2]], ["a", "a", "b"])
    trainer = StreamTrainer(pipe, labels=["b", "a"])
    result = trainer.score_chunk([[0], [1]], ["a", "b"])

    np.testing.assert_allclose(result["y_score"], [2 / 3, 2 / 3])
    assert result["auc"] == 0.5
    assert trainer.metrics.result()["count"] == 0
    assert trainer.logs == []


def test_trainer_scores_single_class_models_for_either_class():
    pipe = Pipeline([("model", DecisionTreeClassifier())])
    trainer = StreamTrainer(pipe)
    for label in (0, 1):
        pipe.fit([[0]], [label])
        result = trainer.score_chunk([[0], [1]], [0, 1])

        np.testing.assert_array_equal(result["y_score"], [label, label])
        assert result["auc"] == 0.5


def test_failed_training_does_not_record_evaluation_or_leave_tracing_enabled():
    pipe = Pipeline([("model", DecisionTreeClassifier(buffer_size=2))])
    trainer = StreamTrainer(pipe)
    trainer.fit_chunk([[0], [1]], [0, 1])
    trainer.fit_chunk([[0]], [1])
    original_partial_fit = pipe.partial_fit

    def fail(X, y):
        raise ValueError("Training failed.")

    pipe.partial_fit = fail
    was_tracing = tracemalloc.is_tracing()
    with pytest.raises(ValueError, match="Training failed"):
        trainer.fit_chunk([[1]], [0])
    assert tracemalloc.is_tracing() == was_tracing
    assert len(trainer.logs) == 2
    assert trainer.metrics.result()["count"] == 1

    pipe.partial_fit = original_partial_fit
    assert trainer.fit_chunk([[1]], [0])["retrained"]
    assert trainer.metrics.result()["count"] == 2


def test_trainer_rejects_unknown_labels_before_buffering():
    pipe = Pipeline([("model", DecisionTreeClassifier(buffer_size=2))])
    trainer = StreamTrainer(pipe)
    with pytest.raises(ValueError, match="unknown class"):
        trainer.fit_chunk([[0]], [2])

    assert trainer.logs == []
    assert not trainer.fit_chunk([[0]], [0])["retrained"]
    assert trainer.fit_chunk([[1]], [1])["retrained"]


def test_tree_predict_proba_preserves_class_order_and_predict_ties():
    tree = DecisionTreeClassifier(max_depth=0).fit([[0], [1]], ["z", "a"])

    np.testing.assert_array_equal(tree.classes_, ["a", "z"])
    np.testing.assert_allclose(tree.predict_proba([[3], [4]]), [[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_array_equal(tree.predict([[3], [4]]), ["a", "a"])
    assert tree.predict_proba(np.empty((0, 1))).shape == (0, 2)
