import pytest

from src.homeworks.KNN.processing.scaling_strategies.min_max_scaler import \
    MinMaxScaler


class TestMinMaxScaler:

    @pytest.mark.parametrize(
        "X, axis, expected",
        [
            ([(1, 2), (3, 4), (0, 5)], 0, 0),
            ([(1, 2), (3, 4), (0, 5)], 1, 2),
            ([(5, 5), (5, 5), (5, 5)], 0, 5),
            ([(5, 5), (5, 5), (5, 5)], 1, 5),
            ([(-1, -2), (-3, -4), (-5, -6)], 0, -5),
            ([(-1, -2), (-3, -4), (-5, -6)], 1, -6),
            ([(1.5, 2.5), (3.5, 4.5), (0.5, 5.5)], 0, 0.5),
            ([(1.5, 2.5), (3.5, 4.5), (0.5, 5.5)], 1, 2.5),
            ([(1, 2, 3), (4, 5, 6)], 2, 3),
            ([(10, 2), (1, 2)], 0, 1),
        ],
    )
    def test_get_min_by_axis(self, X, axis, expected):
        points = [p for p in X]
        assert MinMaxScaler._get_min_by_axis(points, axis) == expected

    @pytest.mark.parametrize(
        "X, axis, expected",
        [
            ([(1, 2), (3, 4), (0, 5)], 0, 3),
            ([(1, 2), (3, 4), (0, 5)], 1, 5),
            ([(5, 5), (5, 5), (5, 5)], 0, 5),
            ([(5, 5), (5, 5), (5, 5)], 1, 5),
            ([(-1, -2), (-3, -4), (-5, -6)], 0, -1),
            ([(-1, -2), (-3, -4), (-5, -6)], 1, -2),
            ([(1.5, 2.5), (3.5, 4.5), (0.5, 5.5)], 0, 3.5),
            ([(1.5, 2.5), (3.5, 4.5), (0.5, 5.5)], 1, 5.5),
            ([(1, 2, 3), (4, 5, 6)], 2, 6),
            ([(10, 2), (1, 2)], 0, 10),
        ],
    )
    def test_get_max_by_axis(self, X, axis, expected):
        points = [p for p in X]
        assert MinMaxScaler._get_max_by_axis(points, axis) == expected

    @pytest.mark.parametrize(
        "X",
        [
            ([(1, 2), (3, 4), (0, 5)]),
            ([(5, 5), (5, 5), (5, 5)]),
            ([(-1, -2), (-3, -4), (-5, -6)]),
            ([(1.5, 2.5), (3.5, 4.5), (0.5, 5.5)]),
            ([(1, 2, 3), (4, 5, 6)]),
            ([(10, 2), (1, 2)]),
            ([(0, 0), (1, 1), (2, 2)]),
            ([(-10, 10), (0, 0), (10, -10)]),
            ([(100, 200), (300, 400)]),
            ([(-0.5, -0.5), (0.5, 0.5)]),
        ],
    )
    def test_fit(self, X):
        points = [p for p in X]
        scaler = MinMaxScaler()
        scaler.fit(points)
        expected_min = tuple(
            MinMaxScaler._get_min_by_axis(points, axis) for axis in range(len(X[0]))
        )
        expected_max = tuple(
            MinMaxScaler._get_max_by_axis(points, axis) for axis in range(len(X[0]))
        )
        assert scaler.min == expected_min
        assert scaler.max == expected_max

    @pytest.mark.parametrize(
        "X",
        [
            ([(1, 2), (3, 4), (0, 5)]),
            ([(5, 5), (5, 5), (5, 5)]),
            ([(-1, -2), (-3, -4), (-5, -6)]),
            ([(1.5, 2.5), (3.5, 4.5), (0.5, 5.5)]),
            ([(1, 2, 3), (4, 5, 6)]),
            ([(10, 2), (1, 2)]),
            ([(0, 0), (1, 1), (2, 2)]),
            ([(-10, 10), (0, 0), (10, -10)]),
            ([(100, 200), (300, 400)]),
            ([(-0.5, -0.5), (0.5, 0.5)]),
        ],
    )
    def test_transform(self, X):
        points = [p for p in X]
        scaler = MinMaxScaler()
        scaler.fit(points)
        transformed_points = scaler.transform(points)

        expected_transformed_points = []
        for point in points:
            new_point = []
            for axis, coordinate in enumerate(point):
                expected_val = (
                    (coordinate - scaler.min[axis])
                    / (scaler.max[axis] - scaler.min[axis])
                    if scaler.max[axis] != scaler.min[axis]
                    else 0.0
                )
                new_point.append(expected_val)
            expected_transformed_points.append(tuple(new_point))

        for i in range(len(transformed_points)):
            for j in range(len(transformed_points[i])):
                assert transformed_points[i][j] == pytest.approx(
                    expected_transformed_points[i][j]
                )

    def test_fit_empty_X(self):
        scaler = MinMaxScaler()
        with pytest.raises(ValueError, match="X is empty"):
            scaler.fit([])

    def test_transform_empty_X(self):
        scaler = MinMaxScaler()
        with pytest.raises(ValueError, match="X is empty"):
            scaler.transform([])

    def test_fit_different_dimensions(self):
        scaler = MinMaxScaler()
        with pytest.raises(
            ValueError, match="there are points with different dimensions in X"
        ):
            scaler.fit([(1, 2), (3,), (4, 5, 6)])

    def test_transform_different_dimensions(self):
        scaler = MinMaxScaler()
        scaler.fit([(1, 2), (3, 4)])  # Fit first to avoid "not trained" error
        with pytest.raises(
            ValueError, match="there are points with different dimensions in X"
        ):
            scaler.transform([(1, 2), (3,), (4, 5, 6)])

    def test_transform_not_fitted(self):
        scaler = MinMaxScaler()
        with pytest.raises(ValueError, match="MinMaxScaler is not trained"):
            scaler.transform([(1, 2), (3, 4)])
