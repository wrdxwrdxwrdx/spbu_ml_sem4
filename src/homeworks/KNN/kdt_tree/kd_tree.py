import math
from typing import Callable, Optional, Sequence

from src.homeworks.KNN.kdt_tree.kdt_heap import MaxHeap
from src.homeworks.KNN.kdt_tree.kdt_node import KDTNode, Point


class KDTree:
    """K Dimensional Tree for efficient nearest neighbour search."""

    def __init__(
        self,
        X: Sequence[Point],
        leaf_size: int,
        metric: Optional[Callable[[Point, Point], float]] = None,
    ):
        """kdt_tree init.

        :param X: Given points.
        :param leaf_size: Max size of each leaf in tree.
        :param metric: Function for determining the distance between two points.

        :raises ValueError: If points with different dimensions are provided.
        """

        self._points = X
        if len(set(map(len, X))) != 1:
            print(X)
            raise ValueError("Points with different dimensions are given")
        self._leaf_size = leaf_size
        self._root = KDTNode(X, leaf_size)
        self.metric = metric if metric else self._default_metric

    @staticmethod
    def _default_metric(point_1: Point, point_2: Point) -> float:
        """
        Computes the Euclidean distance between two points.

        :param point_1: First point.
        :param point_2: Second point.

        :return: Euclidean distance between the points.

        :raises ValueError: If the points are of different dimensions.
        """

        if len(point_1) != len(point_2):
            raise ValueError("Points must have the same dimension")

        result = 0.0
        for axis in range(len(point_1)):
            result += (point_1[axis] - point_2[axis]) ** 2
        return math.sqrt(result)

    def __str__(self):
        return str(self._root)

    def __repr__(self):
        return repr(self._root)

    def _median_by_axis(self, axis: int) -> Point:
        """
        Calculating the median of points along the axis.

        :param axis: Axis along which to compute the median.

        :return: Median point along the specified axis.
        """

        points = list(sorted(self._points, key=lambda p: p[axis]))
        return points[len(points) // 2]

    def _knn_recursion(
        self, point: Point, node: Optional[KDTNode], k: int, max_heap: MaxHeap
    ):
        """
        Recursive helper function to find k nearest neighbors.

        :param point: Point for which neighbors are being found.
        :param node: Current node in the kdt_tree.
        :param k: Number of nearest neighbors.
        :param max_heap: Max heap to store nearest neighbors.
        """

        # None
        if node is None:
            return max_heap

        # Leaf
        if node.is_leaf:
            for node_point in node.points:
                max_heap.push(point, node_point)
            return max_heap

        axis = node.axis
        max_heap.push(point, node.points[0])

        # Node
        if point[axis] <= node.points[0][axis]:
            self._knn_recursion(point, node.left, k, max_heap)
            axis = node.axis
            hyperplane_dist = abs(point[axis] - node.points[0][axis])
            if hyperplane_dist < max_heap.max_dist() or (
                node and node.right and not node.is_leaf and node.right.is_leaf
            ):
                self._knn_recursion(point, node.right, k, max_heap)
        else:
            self._knn_recursion(point, node.right, k, max_heap)
            axis = node.axis
            hyperplane_dist = abs(point[axis] - node.points[0][axis])
            if hyperplane_dist < max_heap.max_dist() or (
                node and node.left and not node.is_leaf and node.left.is_leaf
            ):
                self._knn_recursion(point, node.left, k, max_heap)

    def k_nearest_neighbours(self, point: Point, k: int) -> Sequence[Point]:
        """
        Finding the k nearest neighbors for a given point.

        :param point: Point for which neighbors are being found.
        :param k: Number of nearest neighbors.

        :return: Sequence of k nearest points.
        """

        max_heap = MaxHeap(self.metric, k)
        self._knn_recursion(point, self._root, k, max_heap)
        return max_heap.points()

    def query(self, X: Sequence[Point], k: int) -> Sequence[Sequence[Point]]:
        """
        Finds the k nearest neighbors for each point in the input sequence.

        :param X: Sequence of points.
        :param k: Number of nearest neighbors.

        :return: sequence of lists of k nearest points for each of the given points
        """

        result = []
        for point in X:
            result.append(self.k_nearest_neighbours(point, k))
        return result
