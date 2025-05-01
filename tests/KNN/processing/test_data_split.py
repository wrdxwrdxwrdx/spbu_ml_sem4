from typing import Sequence

import pytest

from src.homeworks.KNN.processing.data_split import train_test_split


class TestDataSplit:
    @pytest.mark.parametrize(
        "test_size, to_shuffle, seed",
        [
            (0.0, True, 42),
            (0.2, True, 42),
            (0.5, True, 42),
            (1.0, True, 42),
            (0.3, False, 42),
            (0.7, False, 42),
            (0.1, True, 100),
            (0.4, True, 5),
        ],
    )
    def test_train_test_split_size(self, test_size: float, to_shuffle: bool, seed: int) -> None:
        """Test if the split size is correct."""
        X = [(float(i),) for i in range(100)]
        y = list(range(100))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, to_shuffle=to_shuffle, seed=seed)
        assert len(X_test) == int(len(X) * test_size)
        assert len(X_train) == len(X) - int(len(X) * test_size)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    @pytest.mark.parametrize(
        "test_size, to_shuffle, seed",
        [
            (0.2, False, 42),
            (0.5, False, 42),
            (0.8, False, 42),
            (0.0, False, 42),
            (1.0, False, 42),
            (0.3, False, 100),
            (0.7, False, 5),
            (0.1, False, 42),
            (0.9, False, 42),
            (0.4, False, 42),
        ],
    )
    def test_train_test_split_no_shuffle_order(self, test_size: float, to_shuffle: bool, seed: int) -> None:
        """Test if the order is preserved when shuffle is False."""
        X = [(float(i),) for i in range(10)]
        y = list(range(10))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, to_shuffle=to_shuffle, seed=seed)
        split_pointer = int(len(X) * (1 - test_size))
        assert X_train == X[:split_pointer]
        assert X_test == X[split_pointer:]
        assert y_train == y[:split_pointer]
        assert y_test == y[split_pointer:]

    @pytest.mark.parametrize(
        "test_size, to_shuffle, seed, input_len",
        [
            (0.2, True, 42, 100),
            (0.5, True, 42, 50),
            (0.8, True, 42, 200),
            (0.0, True, 42, 10),
            (1.0, True, 42, 5),
            (0.3, False, 100, 150),
            (0.7, False, 5, 75),
            (0.1, True, 42, 300),
            (0.9, False, 42, 25),
            (0.4, True, 5, 120),
        ],
    )
    def test_train_test_split_total_length(self, test_size: float, to_shuffle: bool, seed: int, input_len: int) -> None:
        """Test if the total length is preserved after split."""
        X = [(float(i),) for i in range(input_len)]
        y = list(range(input_len))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, to_shuffle=to_shuffle, seed=seed)
        assert len(X_train) + len(X_test) == input_len
        assert len(y_train) + len(y_test) == input_len

    @pytest.mark.parametrize(
        "test_size, to_shuffle, seed, input_data",
        [
            (0.2, True, 42, ([(1.0,), (2.0,), (3.0,)], [0, 1, 0])),
            (0.5, True, 42, ([(4.0,), (5.0,)], [1, 1])),
            (
                0.8,
                True,
                42,
                ([(6.0,), (7.0,), (8.0,), (9.0,), (10.0,)], [0, 0, 1, 1, 0]),
            ),
            (0.0, True, 42, ([(11.0,)], [1])),
            (1.0, True, 42, ([(12.0,), (13.0,)], [0, 1])),
            (0.3, False, 100, ([(14.0,), (15.0,), (16.0,), (17.0,)], [1, 0, 1, 0])),
            (0.7, False, 5, ([(18.0,), (19.0,), (20.0,)], [0, 1, 1])),
            (
                0.1,
                True,
                42,
                (
                    [(21.0,), (22.0,), (23.0,), (24.0,), (25.0,), (26.0,)],
                    [1, 0, 0, 1, 1, 0],
                ),
            ),
            (0.9, False, 42, ([(27.0,), (28.0,)], [0, 0])),
            (
                0.4,
                True,
                5,
                ([(29.0,), (30.0,), (31.0,), (32.0,), (33.0,)], [1, 1, 0, 0, 1]),
            ),
        ],
    )
    def test_train_test_split_data_integrity(
        self,
        test_size: float,
        to_shuffle: bool,
        seed: int,
        input_data: tuple[Sequence[tuple[float]], Sequence[int]],
    ) -> None:
        """Test if data integrity is maintained after split (pairs of X and y are kept together after shuffling)."""
        X, y = input_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, to_shuffle=to_shuffle, seed=seed)
        if to_shuffle:
            combined_original = list(zip(X, y))
            combined_train = list(zip(X_train, y_train))
            combined_test = list(zip(X_test, y_test))

            reconstructed_data = combined_train + combined_test
            # Sort both lists of tuples by the original index (if we could track it, but shuffle is random, so we just check the elements are the same)
            assert set(tuple(sorted(combined_original))) == set(tuple(sorted(reconstructed_data)))
        else:
            split_pointer = int(len(X) * (1 - test_size))
            assert list(X_train) == list(X[:split_pointer])
            assert list(X_test) == list(X[split_pointer:])
            assert list(y_train) == list(y[:split_pointer])
            assert list(y_test) == list(y[split_pointer:])
