from typing import Optional, Sequence

import numpy as np

from src.homeworks.KNN.knn_tree.KDTNode import Point
from src.homeworks.KNN.knn_tree.KDTree import KDTree


class KNNClassifier:
    """
    K-Nearest Neighbors (KNN) classifier.

    This classifier learns from input data and predicts the class of new points
    based on the k nearest neighbors.
    """

    def __init__(self, k: int, leaf_size: int):
        """
        KNN classifier init

        :param k: Number of nearest neighbors.
        :param leaf_size: Leaf size for the knn_tree.
        """
        self.k = k
        self.leaf_size = leaf_size
        self.kdtree: Optional[KDTree] = None
        self.point_to_class: dict[Point, int] = dict()
        self._class_number: int = 0

    def fit(self, X: Sequence[Point], Y: Sequence[int]):
        """
        Trains the classifier by storing input points and their classes.

        :param X: Sequence of input points.
        :param Y: Corresponding class labels of the points.

        :raises ValueError: If the sizes of X and Y do not match.
        """
        if len(X) != len(Y):
            raise ValueError(
                "The size of the sequence of points is not equal to the size of the target sequence"
            )

        self.kdtree = KDTree(X, self.leaf_size)
        for i in range(len(X)):
            self.point_to_class[X[i]] = Y[i]
        self._class_number = max(Y)

    def _get_point_class(self, point: Point) -> int:
        """
        Returns the class of a given point.

        :param point: The point whose class is needed.

        :return: The class of the point.

        :raises ValueError: If the point is not in the training sample.
        """
        if point not in self.point_to_class:
            raise ValueError(f"{point} not in the training sample")
        return self.point_to_class[point]

    def predict_proba(self, X: Sequence[Point]) -> list[list[float]]:
        """
        Computes class membership probabilities for each point.

        :param X: Sequence of points to predict.

        :return: List of class membership probabilities for each point.

        :raises ValueError: If the model has not been trained.
        """
        if self.kdtree is None:
            raise ValueError(
                "The model has not been trained yet. First, use the fit method"
            )

        k_nearest_points = self.kdtree.query(X, self.k)
        result = []
        for i, point in enumerate(X):
            point_neighbours = k_nearest_points[i]
            class_amount = [0] * (self._class_number + 1)
            for neighbour in point_neighbours:
                neighbour_class = self._get_point_class(neighbour)
                class_amount[neighbour_class] += 1
            probabilities = list(map(lambda amount: amount / self.k, class_amount))
            result.append(probabilities)
        return result

    def predict(self, X: Sequence[Point]) -> list[int]:
        """
        Predicts the class for each point in X.

        :param X: Sequence of points to predict.

        :return: List of predicted class labels.
        """
        result = []
        for probabilities in self.predict_proba(X):
            result.append(int(np.argmax(probabilities)))
        return result
