from abc import ABC
from typing import Sequence

from src.homeworks.KNN.kdt_tree.kdt_node import Point


class AbstractScaler(ABC):
    """
    Abstract class for data scaling.

    Defines abstract class for all normalization strategies, including the
    fit, transform, and fit_transform methods.
    """

    def fit(self, X: Sequence[Point]):
        """
        Computes statistics based on the input data X.

        :param X: A sequence of data points.

        :raises ValueError: If X is empty or contains points with different dimensions.
        """

        raise NotImplementedError

    def transform(self, X: Sequence[Point]) -> list[Point]:
        """
        Transforms the input data X based on the statistics learned from the fit method.

        :param X: A sequence of data points to transform.

        :return: A list of transformed data points.

        :raises ValueError: If X is empty or contains points with different dimensions.
        """

        raise NotImplementedError

    def fit_transform(self, X: Sequence[Point]) -> list[Point]:
        """
        Computes statistics from the input data and immediately applies the transformation.

        :param X: A sequence of data points.

        :return: A list of transformed data points.
        """
        self.fit(X)
        return self.transform(X)
