import pytest

from src.homeworks.KNN.processing.scaling_strategies.standard_scaler import \
    StandardScaler


class TestStandardScaler:

    @pytest.mark.parametrize(
        "X, expected_mean",
        [
            ([], ValueError),
            (
                [
                    (1,),
                ],
                (1.0,),
            ),
            ([(1, 2)], (1.0, 2.0)),
            ([(1, 2), (3, 4)], (2.0, 3.0)),
            ([(0, 0), (0, 0), (0, 0)], (0.0, 0.0)),
            ([(1, 2, 3), (4, 5, 6), (7, 8, 9)], (4.0, 5.0, 6.0)),
            ([(-1, -2), (1, 2)], (0.0, 0.0)),
            ([(1.5, 2.5), (3.5, 4.5)], (2.5, 3.5)),
            ([(100, 200), (300, 400)], (200.0, 300.0)),
            ([(0.1, 0.2), (0.3, 0.4)], (0.2, 0.3)),
        ],
    )
    def test_compute_mean(self, X, expected_mean):
        scaler = StandardScaler()
        if not X:
            with pytest.raises(IndexError):
                scaler._compute_mean(X)
        elif X:
            actual_mean = scaler._compute_mean(X)
            assert actual_mean == pytest.approx(expected_mean)

    @pytest.mark.parametrize(
        "X, expected_std",
        [
            (
                [
                    (1,),
                ],
                (0.0,),
            ),
            ([(1, 2)], (0.0, 0.0)),
            ([(2, 4), (4, 8)], (1.0, 2.0)),
            ([(0, 0), (0, 0), (0, 0)], (0.0, 0.0)),
            (
                [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
                (2.449489742783178, 2.449489742783178, 2.449489742783178),
            ),
            ([(-1, -2), (1, 2)], (1.0, 2.0)),
            ([(1.0, 2.0), (3.0, 4.0)], (1.0, 1.0)),
            ([(10, 20), (30, 40)], (10.0, 10.0)),
            ([(0.1, 0.2), (0.3, 0.4)], (0.1, 0.1)),
        ],
    )
    def test_compute_std(self, X, expected_std):
        scaler = StandardScaler()
        if expected_std == ValueError:
            with pytest.raises(ValueError):
                scaler._compute_std(X)
        else:
            if not X:
                with pytest.raises(IndexError):
                    scaler._compute_std(X)
            elif X:
                actual_std = scaler._compute_std(
                    X, scaler._compute_mean(X) if X else None
                )
                assert actual_std == pytest.approx(expected_std, abs=1e-6)

    @pytest.mark.parametrize(
        "X",
        [
            [],
            [
                (1,),
            ],
            [(1, 2)],
            [(1, 2), (3, 4)],
            [(0, 0), (0, 0), (0, 0)],
            [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
            [(-1, -2), (1, 2)],
            [(1.5, 2.5), (3.5, 4.5)],
            [(100, 200), (300, 400)],
            [(0.1, 0.2), (0.3, 0.4)],
        ],
    )
    def test_fit(self, X):
        scaler = StandardScaler()
        if not X:
            with pytest.raises(ValueError):
                scaler.fit(X)
        elif len(set(map(len, X))) != 1 and X:
            with pytest.raises(ValueError):
                scaler.fit(X)
        else:
            scaler.fit(X)
            if X:
                expected_mean = scaler._compute_mean(X)
                expected_std = scaler._compute_std(X, expected_mean)
                assert scaler.mean == pytest.approx(expected_mean)
                assert scaler.std == pytest.approx(expected_std, abs=1e-6)
            else:  # When X is empty, mean and std should remain None
                assert scaler.mean is None
                assert scaler.std is None

    @pytest.mark.parametrize(
        "X",
        [
            [
                (1,),
            ],
            [(1, 2)],
            [(1, 2), (3, 4)],
            [(0, 0), (0, 0), (0, 0)],
            [(1, 2, 3), (4, 5, 6), (7, 8, 9)],
            [(-1, -2), (1, 2)],
            [(1.5, 2.5), (3.5, 4.5)],
            [(100, 200), (300, 400)],
            [(0.1, 0.2), (0.3, 0.4)],
        ],
    )
    def test_transform(self, X):
        scaler = StandardScaler()

        with pytest.raises(ValueError, match="StandardScaler is not trained"):
            scaler.transform(X)

        if not X:
            with pytest.raises(ValueError, match="X is empty"):
                scaler.transform(X)
        elif len(set(map(len, X))) != 1 and X:
            with pytest.raises(
                ValueError, match="there are points with different dimensions in X"
            ):
                scaler.transform(X)
        else:
            if X:
                scaler.fit(X)
                transformed_X = scaler.transform(X)
                expected_mean = scaler.mean
                expected_std = scaler.std

                expected_transformed_X = []
                for point in X:
                    new_point = tuple(
                        [
                            (
                                (point[i] - expected_mean[i]) / expected_std[i]
                                if expected_std[i]
                                else 0
                            )
                            for i in range(len(point))
                        ]
                    )
                    expected_transformed_X.append(new_point)

                for actual_point, expected_point in zip(
                    transformed_X, expected_transformed_X
                ):
                    assert actual_point == pytest.approx(expected_point, abs=1e-6)
            else:
                with pytest.raises(ValueError, match="X is empty"):
                    scaler.transform(X)
