from typing import Sequence

from src.homeworks.KNN.kdt_tree.kdt_Node import Point


def accuracy(y_prediction: Sequence[Point], y_true: Sequence[Point]) -> float:
    """Accuracy metric

    :param y_prediction: target predicted by the model
    :param y_true: correct target values
    """

    if len(y_prediction) != len(y_true):
        raise ValueError("y_prediction and y_true are of different lengths")

    return sum([y_prediction[i] == y_true[i] for i in range(len(y_true))]) / len(y_true)


def f1_score(y_prediction: Sequence[Point], y_true: Sequence[Point]) -> float:
    """F1 score metric

    :param y_prediction: target predicted by the model
    :param y_true: correct target values
    """

    if len(y_prediction) != len(y_true):
        raise ValueError("y_prediction and y_true are of different lengths")

    true_positive, true_negative, false_positive, false_negative = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_prediction[i] == y_true[i]:
            if y_prediction[i]:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if y_prediction[i]:
                false_positive += 1
            else:
                false_negative += 1
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
