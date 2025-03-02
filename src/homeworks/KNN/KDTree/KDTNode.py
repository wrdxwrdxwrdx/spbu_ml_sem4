from typing import Optional, Sequence

import numpy as np

Point = tuple[float, ...]


class KDTNode:
    """Node for K Dimensional Tree"""

    def __init__(self, points: Sequence[Point], leaf_size: int, is_leaf=True):
        """"""
        self.leaf_size = leaf_size
        self.points = points
        self.axis = self._chose_axis()
        self.left: Optional["KDTNode"] = None
        self.right: Optional["KDTNode"] = None
        self.is_leaf = is_leaf
        self._maintain_init_invariance()

    def __str__(self):
        return f"(p:{self.points}, l:{self.left}, r:{self.right})"

    def __repr__(self):
        return f"(p:{self.points}, l:{repr(self.left)}, r:{repr(self.right)}, a:{self.axis})"

    def _count_variance(self, axis: int) -> float:
        """Calculating the variance along the axis

        :param axis: the number of the vector's coordinate for comparison

        :return: variance along the axis"""
        axis_values = [point[axis] for point in self.points]
        average_axis_value = sum(axis_values) / len(axis_values)
        variance = 0.0
        for axis_value in axis_values:
            variance += (axis_value - average_axis_value) ** 2
        variance /= len(axis_values)
        return variance

    def _maintain_init_invariance(self) -> None:
        """Maintaining init invariance"""
        if len(self.points) <= self.leaf_size:
            return
        self.points = list(sorted(self.points, key=lambda point: point[self.axis]))
        mid_pointer = len(self.points) // 2
        median = self.points[mid_pointer]
        self.left = KDTNode(self.points[:mid_pointer], self.leaf_size)
        self.right = KDTNode(self.points[mid_pointer + 1 :], self.leaf_size)
        self.points = [median]
        self.is_leaf = False

    def _chose_axis(self) -> int:
        """Choosing the best axis to create a hyperplane

        :return: best axis"""
        if len(self.points) == 0:
            return 0
        variances = [self._count_variance(axis) for axis in range(len(self.points[0]))]
        return int(np.argmax(variances))
