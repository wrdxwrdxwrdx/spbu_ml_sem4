from typing import Sequence

import pytest

from src.homeworks.KNN.kdt_tree.kdt_node import Point
from src.homeworks.KNN.processing.scaling_strategies.max_abs_scaler import MaxAbsScaler


class TestMaxAbsScaler:

    @pytest.mark.parametrize(
        "X, axis, expected_max_abs",
        [
            ([(1.0, 2.0), (3.0, 4.0)], 0, 3.0),
            ([(1.0, 2.0), (3.0, 4.0)], 1, 4.0),
            ([(-1.0, -2.0), (-3.0, -4.0)], 0, 3.0),
            ([(-1.0, -2.0), (-3.0, -4.0)], 1, 4.0),
            ([(0.0, 0.0), (0.0, 0.0)], 0, 0.0),
            ([(0.0, 0.0), (0.0, 0.0)], 1, 0.0),
            ([(1.0, 2.0), (-3.0, -4.0)], 0, 3.0),
            ([(1.0, 2.0), (-3.0, -4.0)], 1, 4.0),
            ([(5.0, -1.0), (2.0, -7.0)], 0, 5.0),
            ([(5.0, -1.0), (2.0, -7.0)], 1, 7.0),
        ],
    )
    def test_get_max_abs_by_axis(self, X: Sequence[Point], axis: int, expected_max_abs: float):
        scaler = MaxAbsScaler()
        assert scaler._get_max_abs_by_axis(X, axis) == expected_max_abs

    @pytest.mark.parametrize(
        "X",
        [
            ([(1.0, 2.0), (3.0, 4.0)]),
            ([(-1.0, -2.0), (-3.0, -4.0)]),
            ([(0.0, 0.0), (0.0, 0.0)]),
            ([(1.0, 2.0), (-3.0, -4.0)]),
            ([(5.0, -1.0), (2.0, -7.0)]),
            ([(10.0, 20.0), (3.0, 4.0)]),
            ([(-0.1, -0.2), (-0.3, -0.4)]),
            ([(0.5, 0.5), (0.5, 0.5)]),
            ([(1.0, -2.0), (-1.0, 2.0)]),
            ([(1e6, 2e6), (3e6, 4e6)]),
        ],
    )
    def test_fit_valid_input(self, X: Sequence[Point]):
        scaler = MaxAbsScaler()
        scaler.fit(X)
        assert scaler.max_abs is not None

    @pytest.mark.parametrize(
        "X",
        [
            ([]),
        ],
    )
    def test_fit_empty_input(self, X: Sequence[Point]):
        scaler = MaxAbsScaler()
        with pytest.raises(ValueError) as exc_info:
            scaler.fit(X)
        assert str(exc_info.value) == "X is empty"

    @pytest.mark.parametrize(
        "X_train, X_test, expected_transformed",
        [
            (
                [(1.0, 2.0), (3.0, 4.0)],
                [(1.0, 2.0), (3.0, 4.0)],
                [(1.0 / 3.0, 2.0 / 4.0), (3.0 / 3.0, 4.0 / 4.0)],
            ),
            (
                [(-1.0, -2.0), (-3.0, -4.0)],
                [(-1.0, -2.0), (-3.0, -4.0)],
                [(-1.0 / 3.0, -2.0 / 4.0), (-3.0 / 3.0, -4.0 / 4.0)],
            ),
            (
                [(0.0, 0.0), (0.0, 0.0)],
                [(0.0, 0.0), (0.0, 0.0)],
                [(0.0, 0.0), (0.0, 0.0)],
            ),
            (
                [(1.0, 2.0), (-3.0, -4.0)],
                [(1.0, 2.0), (-3.0, -4.0)],
                [(1.0 / 3.0, 2.0 / 4.0), (-3.0 / 3.0, -4.0 / 4.0)],
            ),
            (
                [(5.0, -1.0), (2.0, -7.0)],
                [(5.0, -1.0), (2.0, -7.0)],
                [(5.0 / 5.0, -1.0 / 7.0), (2.0 / 5.0, -7.0 / 7.0)],
            ),
            (
                [(-0.1, -0.2), (-0.3, -0.4)],
                [(-0.1, -0.2), (-0.3, -0.4)],
                [(-0.1 / 0.3, -0.2 / 0.4), (-0.3 / 0.3, -0.4 / 0.4)],
            ),
            (
                [(0.5, 0.5), (0.5, 0.5)],
                [(0.5, 0.5), (0.5, 0.5)],
                [(0.5 / 0.5, 0.5 / 0.5), (0.5 / 0.5, 0.5 / 0.5)],
            ),
            (
                [(2.0, 4.0), (6.0, 8.0)],
                [(1.0, 2.0), (3.0, 4.0)],
                [(1.0 / 6.0, 2.0 / 8.0), (3.0 / 6.0, 4.0 / 8.0)],
            ),
        ],
    )
    def test_transform_valid_input(
        self,
        X_train: Sequence[Point],
        X_test: Sequence[Point],
        expected_transformed: list[Point],
    ):
        scaler = MaxAbsScaler()
        scaler.fit(X_train)
        transformed_X = scaler.transform(X_test)
        for transformed_point, expected_point in zip(transformed_X, expected_transformed):
            assert len(transformed_point) == len(expected_point)
            for i in range(len(transformed_point)):
                assert pytest.approx(transformed_point[i]) == expected_point[i]

    @pytest.mark.parametrize(
        "X",
        [
            ([]),
        ],
    )
    def test_transform_empty_input(self, X: Sequence[Point]):
        scaler = MaxAbsScaler()
        scaler.fit([(1.0, 2.0)])
        with pytest.raises(ValueError) as exc_info:
            scaler.transform(X)
        assert str(exc_info.value) == "X is empty"

    def test_transform_not_fitted(self):
        scaler = MaxAbsScaler()
        X = [(1.0, 2.0), (3.0, 4.0)]
        with pytest.raises(ValueError) as exc_info:
            scaler.transform(X)
        assert str(exc_info.value) == "MinMaxScaler is not trained. To get started, use the fit method"
