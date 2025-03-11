import math

import pytest

from src.homeworks.KNN.kdt_tree.kd_tree import KDTree


class TestKDTree:
    @pytest.mark.parametrize(
        "points, leaf_size, metric, expected_error",
        [
            ([(1, 2), (3, 4), (5, 6)], 1, None, None),
            ([(1, 2, 3), (4, 5, 6), (7, 8, 9)], 2, None, None),
            ([(1,), (2,), (3,)], 1, None, None),
            ([(1.5, 2.5), (3.5, 4.5), (5.5, 6.5)], 1, None, None),
            (
                [(1, 2), (3, 4), (5, 6)],
                3,
                lambda x, y: sum(abs(a - b) for a, b in zip(x, y)),
                None,
            ),
            (
                [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
                2,
                lambda x, y: max(abs(a - b) for a, b in zip(x, y)),
                None,
            ),
            ([(1,), (2,), (3,)], 1, lambda x, y: abs(x[0] - y[0]), None),
            (
                [(1.5, 2.5), (3.5, 4.5), (5.5, 6.5)],
                2,
                lambda x, y: math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y))),
                None,
            ),
            ([(1, 2), (3, 4, 5)], 1, None, ValueError),
            ([(1,), (2, 3)], 1, None, ValueError),
        ],
    )
    def test_kd_tree_initialization(self, points, leaf_size, metric, expected_error):
        if expected_error:
            with pytest.raises(expected_error):
                KDTree(points, leaf_size, metric)
        else:
            tree = KDTree(points, leaf_size, metric)
            assert tree._points == points
            assert tree._leaf_size == leaf_size
            assert tree.metric == (metric if metric else tree._default_metric)

    @pytest.mark.parametrize(
        "point_1, point_2, expected_distance, expected_error",
        [
            ((1, 2), (3, 4), math.sqrt((3 - 1) ** 2 + (4 - 2) ** 2), None),
            ((1,), (2,), 1.0, None),
            (
                (1.5, 2.5),
                (3.5, 4.5),
                math.sqrt((3.5 - 1.5) ** 2 + (4.5 - 2.5) ** 2),
                None,
            ),
            (
                (1, 2, 3),
                (4, 5, 6),
                math.sqrt((4 - 1) ** 2 + (5 - 2) ** 2 + (6 - 3) ** 2),
                None,
            ),
            ((1, 2), (3, 4, 5), None, ValueError),
            ((1,), (2, 3), None, ValueError),
        ],
    )
    def test_default_metric(self, point_1, point_2, expected_distance, expected_error):
        tree = KDTree([(1, 2)], 1)  # Инициализация с любыми данными
        if expected_error:
            with pytest.raises(expected_error):
                tree._default_metric(point_1, point_2)
        else:
            distance = tree._default_metric(point_1, point_2)
            assert math.isclose(distance, expected_distance, rel_tol=1e-9)

    @pytest.mark.parametrize(
        "points, axis, expected_median",
        [
            ([(1, 2), (3, 4), (5, 6)], 0, (3, 4)),
            ([(1, 2), (3, 4), (5, 6)], 1, (3, 4)),
            ([(1, 2, 3), (4, 5, 6), (7, 8, 9)], 0, (4, 5, 6)),
            ([(1, 2, 3), (4, 5, 6), (7, 8, 9)], 1, (4, 5, 6)),
            ([(1,), (2,), (3,)], 0, (2,)),
            ([(1.5, 2.5), (3.5, 4.5), (5.5, 6.5)], 0, (3.5, 4.5)),
            ([(1.5, 2.5), (3.5, 4.5), (5.5, 6.5)], 1, (3.5, 4.5)),
        ],
    )
    def test_median_by_axis(self, points, axis, expected_median):
        tree = KDTree(points, leaf_size=1)
        median = tree._median_by_axis(axis)
        assert median == expected_median

    @pytest.mark.parametrize(
        "point, k, expected_neighbours",
        [
            ((2, 3), 1, [(1.5, 2.5)]),
            ((4, 5), 2, [(3.5, 4.5), (5, 6)]),
            ((3.5, 4.5), 1, [(3.5, 4.5)]),
            ((2, 3), 3, [(1, 2), (3, 4), (1.5, 2.5)]),
            ((3.5, 4.5), 2, [(3.5, 4.5), (3, 4)]),
            ((2, 3), 2, [(1.5, 2.5), (3, 4)]),
        ],
    )
    def test_k_nearest_neighbours(self, point, k, expected_neighbours):
        points = [(1, 2), (3, 4), (5, 6), (1.5, 2.5), (3.5, 4.5), (5.5, 6.5)]
        tree = KDTree(points, leaf_size=1)

        result = tree.k_nearest_neighbours(point, k)
        assert set(result) == set(expected_neighbours)

    @pytest.mark.parametrize(
        "points, k, expected_results",
        [
            (
                ((3.5, 4.5), (2, 3), (4, 5)),
                2,
                [[(3.5, 4.5), (3, 4)], [(1.5, 2.5), (3, 4)], [(3.5, 4.5), (5, 6)]],
            ),
            (((3.5, 4.5), (2, 3)), 1, [[(3.5, 4.5)], [(1.5, 2.5)]]),
            (((2, 3),), 3, [[(1, 2), (3, 4), (1.5, 2.5)]]),
        ],
    )
    def test_query(self, points, k, expected_results):
        tree_points = [(1, 2), (3, 4), (5, 6), (1.5, 2.5), (3.5, 4.5), (5.5, 6.5)]
        tree = KDTree(tree_points, leaf_size=1)
        result = tree.query(points, k)
        if len(result) == len(expected_results):
            for i in range(len(result)):
                assert set(result[i]) == set(expected_results[i])
        else:
            assert False
