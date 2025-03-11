import random

import pytest

from src.homeworks.KNN.kdt_tree.kd_tree import KDTree
from src.homeworks.KNN.kdt_tree.kdt_node import Point
from tests.KNN.default_knn import DefaultKNN


class TestKDTree:
    TEST_PART = 0.15

    @pytest.mark.parametrize(
        "k,selection_size,dimension,leaf_size",
        [(5, 100, 5, 5), (5, 200, 5, 5), (5, 200, 20, 20), (25, 200, 100, 10)],
    )
    def test_query(self, k: int, selection_size: int, dimension: int, leaf_size: int):
        X_train = self._generate_selection(selection_size, dimension)
        X_test = self._generate_selection(
            int(selection_size * self.TEST_PART), dimension
        )

        kdtree = KDTree(X_train, leaf_size)
        knn = DefaultKNN(X_train)

        kdtree_result = kdtree.query(X_test, k)
        knn_result = knn.query(X_test, k)

        for i, test_point in enumerate(X_test):
            kdtree_result_dist = [
                kdtree.metric(test_point, point) for point in kdtree_result[i]
            ]
            knn_result_dist = [
                kdtree.metric(test_point, point) for point in knn_result[i]
            ]
            assert sum(kdtree_result_dist) == sum(knn_result_dist)

    @pytest.mark.parametrize(
        "k,selection_size,dimension,leaf_size",
        [(5, 100, 5, 5), (5, 200, 5, 5), (5, 200, 20, 20), (25, 200, 100, 10)],
    )
    def test_k_nearest_neighbours(
        self, k: int, selection_size: int, dimension: int, leaf_size: int
    ):
        X_train = self._generate_selection(selection_size, dimension)
        test_point = tuple([random.randint(-500, 500) for _ in range(dimension)])

        kdtree = KDTree(X_train, leaf_size)
        knn = DefaultKNN(X_train)

        kdtree_result = kdtree.k_nearest_neighbours(test_point, k)
        knn_result = knn.k_nearest_neighbours(test_point, k)

        kdtree_result_dist = [
            kdtree.metric(test_point, point) for point in kdtree_result
        ]
        knn_result_dist = [knn.metric(test_point, point) for point in knn_result]

        assert sum(kdtree_result_dist) == sum(knn_result_dist)

    def _generate_selection(self, selection_size: int, dimension: int) -> list[Point]:
        return [
            tuple([random.randint(-500, 500) for _ in range(dimension)])
            for _ in range(selection_size)
        ]
