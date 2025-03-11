from typing import Optional, Sequence

from src.homeworks.KNN.kdt_tree.kdt_node import Point
from src.homeworks.KNN.processing.scaling_strategies.abstract_scaler import AbstractScaler


class MaxAbsScaler(AbstractScaler):
    """
    A scaling strategy that scales each feature by its maximum absolute value.
    This transformation ensures that each feature is in the range [-1, 1],
    preserving the sparsity of data.
    """

    def __init__(self):
        self.max_abs: Optional[Point] = None

    @staticmethod
    def _get_max_abs_by_axis(X: Sequence[Point], axis: int) -> float:
        """
        Finds the maximum abs value along a specified axis.

        :param X: The sequence of points.
        :param axis: The axis along which to find the maximum value.

        :return: The maximum value along the specified axis.
        """

        result = abs(X[0][axis])
        for point in X:
            result = max(result, abs(point[axis]))
        return result

    def fit(self, X: Sequence[Point]):
        if len(X) == 0:
            raise ValueError("X is empty")
        if len(set(map(len, X))) != 1:
            raise ValueError("there are points with different dimensions in X")

        self.max_abs = tuple([self._get_max_abs_by_axis(X, axis) for axis in range(len(X[0]))])

    def transform(self, X: Sequence[Point]) -> list[Point]:
        if len(X) == 0:
            raise ValueError("X is empty")
        if len(set(map(len, X))) != 1:
            raise ValueError("there are points with different dimensions in X")

        if self.max_abs is None:
            raise ValueError("MinMaxScaler is not trained. To get started, use the fit method")

        result = []
        for point in X:
            new_point = []
            for axis, coordinate in enumerate(point):
                new_point.append((coordinate / self.max_abs[axis]) if self.max_abs[axis] else 0)
            result.append(tuple(new_point))
        return result
