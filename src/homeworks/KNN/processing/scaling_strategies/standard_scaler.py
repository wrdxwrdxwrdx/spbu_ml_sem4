from typing import Optional, Sequence

from src.homeworks.KNN.kdt_tree.kdt_node import Point
from src.homeworks.KNN.processing.scaling_strategies.abstract_scaler import (
    AbstractScaler,
)


class StandardScaler(AbstractScaler):
    """
    StandardScaler normalizes features by removing the mean and scaling to unit variance.
    """

    def __init__(self):
        self.mean: Optional[Point] = None
        self.std: Optional[Point] = None

    def _compute_std(self, X: Sequence[Point], mean: Optional[Point] = None) -> Point:
        """
        Computes the standard deviation for each feature in the dataset.

        :param X: Sequence of data points.
        :param mean: Pre-computed mean for each feature. If None, computes mean internally.

        :return: Standard deviation for each feature.
        """

        if mean is None:
            mean = self._compute_mean(X)

        std = [0.0] * len(X[0])
        for point in X:
            for i, coordinate in enumerate(point):
                std[i] += (coordinate - mean[i]) ** 2

        for i, coordinate in enumerate(std):
            std[i] = (coordinate / len(X)) ** 0.5

        return tuple(std)

    def _compute_mean(self, X: Sequence[Point]) -> Point:
        """
        Computes the mean for each feature in the dataset.

        :param X: Sequence of data points.

        :return: Mean value for each feature.
        """

        mean = [0.0] * len(X[0])
        for point in X:
            for i, coordinate in enumerate(point):
                mean[i] += coordinate
        mean = [coord / len(X) for coord in mean]

        return tuple(mean)

    def fit(self, X: Sequence[Point]):
        if len(X) == 0:
            raise ValueError("X is empty")
        if len(set(map(len, X))) != 1:
            raise ValueError("there are points with different dimensions in X")

        self.mean = self._compute_mean(X)
        self.std = self._compute_std(X, self.mean)

    def transform(self, X: Sequence[Point]) -> list[Point]:
        if len(X) == 0:
            raise ValueError("X is empty")
        if len(set(map(len, X))) != 1:
            raise ValueError("there are points with different dimensions in X")

        if self.mean is None or self.std is None:
            raise ValueError(
                "StandardScaler is not trained. To get started, use the fit method"
            )

        result = []
        for point in X:
            new_point = [
                (point[i] - self.mean[i]) / self.std[i] if self.std[i] else 0.0
                for i in range(len(point))
            ]
            result.append(tuple(new_point))
        return result
