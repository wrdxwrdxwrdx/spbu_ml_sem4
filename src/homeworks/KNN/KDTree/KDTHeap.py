import heapq
from typing import Callable

from src.homeworks.KNN.KDTree.KDTNode import Point


class MaxHeap:
    """Max heap realization for KDTree"""

    def __init__(self, metric: Callable[[Point, Point], float], capacity: int):
        """MaxHeap init

        :param metric: function for determining the distance between two points
        :param capacity: maximum number of items in a heap"""
        self.metric = metric
        self.heap: list[tuple[float, Point]] = []
        self.capacity = capacity

    def push(self, reference_point: Point, point: Point):
        """Add point and distance between reference_point and point to heap

        :param reference_point: the reference point around which the points are being searched for
        :param point: the point we want to add"""
        dist = self.metric(point, reference_point)
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, (-dist, point))
        elif dist < self.max_dist():
            heapq.heappop(self.heap)
            heapq.heappush(self.heap, (-dist, point))

    def is_empty(self) -> bool:
        """Is heap empty"""
        return len(self.heap) == 0

    def max_dist(self) -> float:
        """Maximum distance stored in the heap"""
        return -self.heap[0][0] if not self.is_empty() else 0

    def points(self) -> list[Point]:
        """Get a list of points stored in the heap"""
        return [point[1] for point in self.heap]

    def __len__(self):
        return len(self.heap)

    def __str__(self):
        return str([(-pair[0], pair[1]) for pair in self.heap])
