import random
from collections.abc import MutableSequence
from typing import Sequence

from src.homeworks.KNN.knn_tree.KDTNode import Point


def train_test_split(
    X: Sequence[Point], y: Sequence[int], test_size: float = 0.2, to_shuffle: bool=True
) -> tuple[Sequence[Point], Sequence[Point], Sequence[int], Sequence[int]]:
    """
    Splits the dataset into training and testing sets.

    :param X: Sequence of input features.
    :param y: Sequence of target labels.
    :param test_size: Proportion of the dataset to include in the test split. Default is 0.2.
    :param to_shuffle: Shuffle . Default is True.

    :return: A tuple containing:
            - X_train: Training input features.
            - X_test: Testing input features.
            - y_train: Training target labels.
            - y_test: Testing target labels.
    """

    split_pointer = int(len(X) * (1 - test_size))

    if to_shuffle:
        pairs = list(zip(X, y))
        random.shuffle(pairs)
        X, y = zip(*pairs)


    X_train, X_test = X[:split_pointer], X[split_pointer:]
    y_train, y_test = y[:split_pointer], y[split_pointer:]
    return X_train, X_test, y_train, y_test
