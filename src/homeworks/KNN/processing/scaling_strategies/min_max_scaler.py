from typing import Optional, Sequence

from src.homeworks.KNN.kdt_tree.kdt_node import Point
from src.homeworks.KNN.processing.scaling_strategies.abstract_scaler import \
    AbstractScaler


class MinMaxScaler(AbstractScaler):
    """
    Scales input data to a given range [min, max].

    This scaler transforms each feature to a given range, typically [0, 1].
    """

    def __init__(self):
        self.min: Optional[Point] = None
        self.max: Optional[Point] = None

    @staticmethod
    def _get_min_by_axis(X: Sequence[Point], axis: int) -> float:
        """
        Finds the minimum value along a specified axis.

        :param X: The sequence of points.
        :param axis: The axis along which to find the minimum value.

        :return: The minimum value along the specified axis.

        :raises ValueError: If X is empty or contains points with different dimensions.
        """

        result = X[0][axis]
        for point in X:
            result = min(result, point[axis])
        return result

    @staticmethod
    def _get_max_by_axis(X: Sequence[Point], axis: int) -> float:
        """
        Finds the maximum value along a specified axis.

        :param X: The sequence of points.
        :param axis: The axis along which to find the maximum value.

        :return: The maximum value along the specified axis.

        :raises ValueError: If X is empty or contains points with different dimensions.
        """

        result = X[0][axis]
        for point in X:
            result = max(result, point[axis])
        return result

    def fit(self, X: Sequence[Point]):
        if len(X) == 0:
            raise ValueError("X is empty")
        if len(set(map(len, X))) != 1:
            raise ValueError("there are points with different dimensions in X")

        self.min = tuple([self._get_min_by_axis(X, axis) for axis in range(len(X[0]))])
        self.max = tuple([self._get_max_by_axis(X, axis) for axis in range(len(X[0]))])

    def transform(self, X: Sequence[Point]) -> list[Point]:
        if len(X) == 0:
            raise ValueError("X is empty")
        if len(set(map(len, X))) != 1:
            raise ValueError("there are points with different dimensions in X")

        if self.min is None or self.max is None:
            raise ValueError(
                "MinMaxScaler is not trained. To get started, use the fit method"
            )

        result = []
        for point in X:
            new_point = []
            for axis, coordinate in enumerate(point):
                if self.max[axis] - self.min[axis] == 0:
                    new_point.append(0.0)
                else:
                    new_point.append(
                        (coordinate - self.min[axis])
                        / (self.max[axis] - self.min[axis])
                    )
            result.append(tuple(new_point))
        return result
