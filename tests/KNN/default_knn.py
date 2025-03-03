import math
from typing import Callable, Optional, Sequence

from src.homeworks.KNN.knn_tree.KDTNode import Point


class DefaultKNN:
    def __init__(
        self,
        x: Sequence[Point],
        metric: Optional[Callable[[Point, Point], float]] = None,
    ):
        self.points = x
        self.metric = metric if metric else self._default_metric

    @staticmethod
    def _default_metric(point_1: Point, point_2: Point) -> float:
        result = 0.0
        for axis in range(len(point_1)):
            result += (point_1[axis] - point_2[axis]) ** 2
        return math.sqrt(result)

    def k_nearest_neighbours(self, point: Point, k: int) -> Sequence[Point]:
        points = []
        for x in self.points:
            points.append((self.metric(point, x), x))
        points.sort()
        return [p[1] for p in points[:k]]

    def query(self, X: Sequence[Point], k: int) -> Sequence[Sequence[Point]]:
        result = []
        for point in X:
            result.append(self.k_nearest_neighbours(point, k))
        return result
