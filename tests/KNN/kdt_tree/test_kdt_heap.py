# --- START OF FILE test_kdt_Heap.py ---
import heapq
import random
from typing import Callable

import pytest

from src.homeworks.KNN.kdt_tree.kdt_Heap import MaxHeap
from src.homeworks.KNN.kdt_tree.kdt_Node import \
    Point  # Assuming Point is defined in kdt_Node.py


def euclidean_distance(point1: Point, point2: Point) -> float:
    """Simple euclidean distance for 2D points for testing."""
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimension")
    return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)) ** 0.5


class TestMaxHeap:
    """Tests for MaxHeap class."""

    @pytest.mark.parametrize("capacity", list(range(100)))
    def test_init(self, capacity):
        """Test MaxHeap initialization."""
        metric = euclidean_distance
        max_heap = MaxHeap(metric, capacity)
        assert max_heap.metric == metric
        assert max_heap.capacity == capacity
        assert max_heap.heap == []

    @pytest.mark.parametrize("capacity", list(range(20, 50)))
    def test_push_basic(self, capacity):
        """Basic test for push method."""
        metric = euclidean_distance
        max_heap = MaxHeap(metric, capacity)
        points_to_push = [
            (random.randint(-500, 500), random.randint(-500, 500))
            for _ in range(capacity)
        ]

        ref_point = (0, 0)
        for point in points_to_push:
            max_heap.push(ref_point, point)

        assert max_heap.max_dist() == max(
            [metric(ref_point, point) for point in points_to_push]
        )
        assert not max_heap.is_empty()

    @pytest.mark.parametrize("capacity", list(range(20, 50)))
    def test_push_capacity_limit(self, capacity):
        """Test push method when capacity is reached and exceeded."""
        metric = euclidean_distance
        max_heap = MaxHeap(metric, capacity)
        points_to_push = [
            (random.randint(-500, 500), random.randint(-500, 500))
            for _ in range(2 * capacity)
        ]

        ref_point = (0, 0)
        for point in points_to_push:
            max_heap.push(ref_point, point)

        assert max_heap.max_dist() == max(
            [metric(ref_point, point) for point in points_to_push]
        )
        assert not max_heap.is_empty()
        assert len(max_heap) == capacity

    def test_is_empty(self):
        """Test is_empty method."""
        metric = euclidean_distance
        max_heap = MaxHeap(metric, 3)
        assert max_heap.is_empty()

        max_heap.push((0, 0), (1, 1))
        assert not max_heap.is_empty()

    @pytest.mark.parametrize("capacity", list(range(20, 50)))
    def test_max_dist_basic(self, capacity):
        """Basic test for max_dist method."""
        metric = euclidean_distance
        max_heap = MaxHeap(metric, capacity)
        points_to_push = [
            (random.randint(-500, 500), random.randint(-500, 500))
            for _ in range(capacity)
        ]

        ref_point = (0, 0)
        for point in points_to_push:
            max_heap.push(ref_point, point)
        expected_max_dist = max([metric(ref_point, point) for point in points_to_push])
        assert not max_heap.is_empty()
        assert max_heap.max_dist() == expected_max_dist

    def test_max_dist_empty_heap(self):
        """Test max_dist method for empty heap."""
        metric = euclidean_distance
        max_heap = MaxHeap(metric, 3)
        assert max_heap.max_dist() == 0

    @pytest.mark.parametrize("capacity", list(range(20, 50)))
    def test_points_basic(self, capacity):
        """Basic test for points method."""
        metric = euclidean_distance
        max_heap = MaxHeap(metric, capacity)
        ref_point = (0, 0)
        points_to_push = [
            (random.randint(-500, 500), random.randint(-500, 500))
            for _ in range(capacity)
        ]

        for point in points_to_push:
            max_heap.push(ref_point, point)

        heap_points = max_heap.points()
        assert set(heap_points) == set(
            points_to_push
        )  # Order is not guaranteed in heap

    def test_points_empty_heap(self):
        """Test points method for empty heap."""
        metric = euclidean_distance
        max_heap = MaxHeap(metric, 3)
        assert max_heap.points() == []

    def test_len_basic(self):
        """Basic test for __len__ method."""
        metric = euclidean_distance
        max_heap = MaxHeap(metric, 3)
        assert len(max_heap) == 0

        max_heap.push((0, 0), (1, 1))
        assert len(max_heap) == 1
        max_heap.push((0, 0), (2, 2))
        assert len(max_heap) == 2
        max_heap.push((0, 0), (3, 3))
        assert len(max_heap) == 3
        max_heap.push((0, 0), (0.1, 0.1))
        assert len(max_heap) == 3
