import heapq
import math

import pytest

from src.homeworks.KNN.kdt_tree.kdt_heap import MaxHeap
from src.homeworks.KNN.kdt_tree.kdt_node import Point


def euclidean_distance(point1: Point, point2: Point) -> float:
    """Euclidean distance metric for testing."""
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimension")
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))


class TestMaxHeap:
    """Property-based tests for MaxHeap class."""

    @pytest.mark.parametrize(
        "capacity",
        [0, 1, 5, 10, 100, -1, 2, 7, 15, 30],
    )
    def test_init(self, capacity: int):
        """Test MaxHeap initialization."""
        metric = euclidean_distance
        heap = MaxHeap(metric, capacity)
        assert heap.capacity == capacity
        assert heap.metric == metric
        assert heap.heap == []

    @pytest.mark.parametrize(
        "capacity, points_to_push",
        [
            (3, [(1, 1), (2, 2), (3, 3)]),
            (2, [(4, 4), (5, 5), (1, 1)]),
            (1, [(1, 1)]),
            (4, []),
            (2, [(5, 5), (4, 4), (6, 6), (3, 3)]),
            (0, [(1, 1), (2, 2)]),  # Capacity 0, nothing should be added
            (3, [(0, 0), (0, 0), (0, 0)]),  # Identical points
            (2, [(-1, -1), (-2, -2), (-3, -3)]),  # Negative coordinates
            (3, [(1.5, 2.5), (2.5, 3.5), (3.5, 4.5)]),  # Float coordinates
        ],
    )
    def test_push(self, capacity, points_to_push):
        """Test push method."""
        metric = euclidean_distance
        heap = MaxHeap(metric, capacity)
        ref_point = (0, 0)
        expected = []
        for point in points_to_push:
            heap.push(ref_point, point)
            heapq.heappush(expected, (euclidean_distance(ref_point, point), point))
            assert set(sorted(expected)[:capacity]) == set(
                map(lambda x: (-x[0], x[1]), heap.heap)
            )

    @pytest.mark.parametrize(
        "capacity, points_to_push, expected_empty",
        [
            (3, [], True),
            (3, [(1, 1)], False),
            (0, [], True),
            (1, [], True),
            (5, [(1, 1), (2, 2), (3, 3)], False),
            (2, [], True),
            (
                4,
                [(1,)],
                False,
            ),  # different dimension point, but we are just checking is_empty after init and push
            (3, [(-1, -1)], False),
            (2, [(1.5, 2.5)], False),
        ],
    )
    def test_is_empty(
        self, capacity: int, points_to_push: list[Point], expected_empty: bool
    ):
        """Test is_empty method."""
        metric = euclidean_distance
        heap = MaxHeap(metric, capacity)
        assert heap.is_empty()

        for point in points_to_push:
            heap.push(point, point)
        assert heap.is_empty() == expected_empty

    @pytest.mark.parametrize(
        "capacity, points_to_push, expected_max_dist",
        [
            (3, [], 0.0),
            (3, [(1, 0)], 1.0),
            (2, [(1, 1), (0, 0)], math.sqrt(2)),
            (1, [(5, 0), (1, 0)], 1.0),
            (
                4,
                [(3, 4), (1, 2), (5, 6)],
                math.sqrt(36 + 25),
            ),  # max distance from (0,0) to (5,6)
            (
                2,
                [(1, 1), (2, 2), (0.5, 0.5)],
                math.sqrt(2),
            ),  # capacity 2, so max of top 2 distances
            (0, [], 0.0),
            (3, [(0, 0), (0, 0), (0, 0)], 0.0),
            (2, [(-3, 0), (-1, 0)], 3.0),
            (3, [(1.5, 0), (0.5, 0), (2.5, 0)], 2.5),
        ],
    )
    def test_max_dist(
        self, capacity: int, points_to_push: list[Point], expected_max_dist: float
    ):
        """Test max_dist method."""
        metric = euclidean_distance
        heap = MaxHeap(metric, capacity)
        assert heap.max_dist() == 0.0  # initially 0

        ref_point = (0, 0)
        for point in points_to_push:
            heap.push(ref_point, point)

        if capacity <= 0 or not points_to_push:
            assert heap.max_dist() == 0.0
        else:
            v = heap.max_dist()
            assert heap.max_dist() == expected_max_dist

    @pytest.mark.parametrize(
        "capacity, points_to_push,_",
        [
            (3, [], []),
            (3, [(1, 0)], [(1, 0)]),
            (2, [(1, 1), (0, 0)], [(1, 1), (0, 0)]),
            (4, [(3, 4), (1, 2), (5, 6)], [(5, 6), (3, 4), (1, 2)]),
            (0, [], []),
            (3, [(0, 0), (0, 0), (0, 0)], [(0, 0), (0, 0), (0, 0)]),
            (2, [(-3, 0), (-1, 0)], [(-3, 0), (-1, 0)]),
            (3, [(1.5, 0), (0.5, 0), (2.5, 0)], [(2.5, 0), (1.5, 0), (0.5, 0)]),
        ],
    )
    def test_points(self, capacity: int, points_to_push: list[Point], _):
        """Test points method."""
        metric = euclidean_distance
        heap = MaxHeap(metric, capacity)

        ref_point = (0, 0)
        for point in points_to_push:
            heap.push(ref_point, point)

        returned_points = heap.points()
        if capacity > 0 and points_to_push:
            expected_points_with_distances = []
            for p in points_to_push:
                expected_points_with_distances.append((-metric(ref_point, p), p))
            expected_points_with_distances.sort(
                key=lambda x: x[0]
            )  # sort by distance DESC
            expected_points = [item[1] for item in expected_points_with_distances]

            if capacity < len(expected_points):
                expected_points = expected_points[:capacity]

            assert (
                len(returned_points) == min(capacity, len(points_to_push))
                if capacity > 0
                else 0
            )

            # check if all expected points are in returned points. Order doesn't strictly matter for points(), only content.
            returned_points_set = set(
                tuple(p) for p in returned_points
            )  # convert to tuples to make hashable
            expected_points_set = set(tuple(p) for p in expected_points)
            assert returned_points_set == expected_points_set
        else:
            assert returned_points == []

    @pytest.mark.parametrize(
        "capacity, points_to_push, expected_len",
        [
            (3, [], 0),
            (3, [(1, 0)], 1),
            (2, [(1, 1), (0, 0)], 2),
            (1, [(5, 0), (1, 0)], 1),
            (
                4,
                [(3, 4), (1, 2), (5, 6), (7, 8), (9, 10)],
                4,
            ),  # push more than capacity
            (2, [(1, 1), (2, 2), (0.5, 0.5)], 2),  # capacity 2, so len should be max 2
            (0, [], 0),
            (3, [(0, 0), (0, 0), (0, 0)], 3),
            (2, [(-3, 0), (-1, 0), (-5, 0)], 2),
            (3, [(1.5, 0), (0.5, 0), (2.5, 0), (3.5, 0)], 3),
        ],
    )
    def test_len(self, capacity: int, points_to_push: list[Point], expected_len: int):
        """Test __len__ method."""
        metric = euclidean_distance
        heap = MaxHeap(metric, capacity)
        assert len(heap) == 0  # initially 0

        ref_point = (0, 0)
        for point in points_to_push:
            heap.push(ref_point, point)
        assert len(heap) == expected_len
