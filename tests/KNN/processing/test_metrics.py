from typing import Sequence

import pytest

from src.homeworks.KNN.processing.metrics import accuracy, f1_score


class TestMetrics:

    @pytest.mark.parametrize(
        "y_true, y_prediction, expected_accuracy",
        [
            ([(1.0,), (0.0,), (1.0,), (0.0,)], [(1.0,), (0.0,), (1.0,), (0.0,)], 1.0),
            ([(1.0,), (0.0,), (1.0,), (0.0,)], [(0.0,), (1.0,), (0.0,), (1.0,)], 0.0),
            ([(1.0,), (0.0,), (1.0,), (0.0,)], [(1.0,), (1.0,), (0.0,), (0.0,)], 0.5),
            ([(True,), (False,)], [(True,), (False,)], 1.0),
            ([(True,), (False,)], [(False,), (True,)], 0.0),
            ([(True,), (False,)], [(True,), (True,)], 0.5),
            ([(1.0,), (1.0,), (1.0,)], [(1.0,), (1.0,), (1.0,)], 1.0),
            ([(0.0,), (0.0,), (0.0,)], [(0.0,), (0.0,), (0.0,)], 1.0),
            (
                [(1.0,), (0.0,), (1.0,), (0.0,), (1.0,)],
                [(1.0,), (0.0,), (0.0,), (1.0,), (1.0,)],
                0.6,
            ),
            ([(0.0,)] * 10, [(0.0,)] * 10, 1.0),
        ],
    )
    def test_accuracy_metric(
        self,
        y_true: Sequence[tuple[float]],
        y_prediction: Sequence[tuple[float]],
        expected_accuracy: float,
    ) -> None:
        """Test accuracy metric with various inputs."""
        assert accuracy(y_prediction, y_true) == expected_accuracy

    @pytest.mark.parametrize(
        "y_true, y_prediction, expected_f1",
        [
            ([(1.0,), (0.0,), (1.0,), (0.0,)], [(1.0,), (0.0,), (1.0,), (0.0,)], 1.0),
            ([(1.0,), (0.0,), (1.0,), (0.0,)], [(0.0,), (1.0,), (0.0,), (1.0,)], 0.0),
            ([(1.0,), (1.0,), (1.0,), (0.0,)], [(1.0,), (1.0,), (0.0,), (0.0,)], 0.86),
            ([(1.0,), (0.0,), (0.0,), (0.0,)], [(1.0,), (1.0,), (1.0,), (1.0,)], 0.4),
            ([(1.0,), (1.0,), (1.0,), (1.0,)], [(1.0,), (1.0,), (1.0,), (1.0,)], 1.0),
            ([(1.0,), (1.0,), (0.0,), (0.0,)], [(1.0,), (1.0,), (1.0,), (1.0,)], 0.67),
            (
                [(1.0,), (0.0,), (1.0,), (0.0,), (1.0,)],
                [(1.0,), (0.0,), (0.0,), (1.0,), (1.0,)],
                0.75,
            ),
        ],
    )
    def test_f1_score_metric(
        self,
        y_true: Sequence[tuple[float]],
        y_prediction: Sequence[tuple[float]],
        expected_f1: float,
    ) -> None:
        """Test F1 score metric with various inputs."""
        calculated_f1 = f1_score(y_prediction, y_true)
        assert round(calculated_f1, 2) == expected_f1

    def test_accuracy_different_lengths_exception(self) -> None:
        """Test if ValueError is raised when lengths of y_prediction and y_true are different in accuracy."""
        y_true = [(1.0,), (0.0,), (1.0,)]
        y_prediction = [(1.0,), (0.0,)]
        with pytest.raises(ValueError):
            accuracy(y_prediction, y_true)

    def test_f1_score_different_lengths_exception(self) -> None:
        """Test if ValueError is raised when lengths of y_prediction and y_true are different in f1_score."""
        y_true = [(1.0,), (0.0,), (1.0,)]
        y_prediction = [(1.0,), (0.0,)]
        with pytest.raises(ValueError):
            f1_score(y_prediction, y_true)
