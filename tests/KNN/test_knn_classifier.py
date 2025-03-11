import pytest

from src.homeworks.KNN.knn_classifier.knn_classifier import KNNClassifier


class TestKNNClassifier:
    @pytest.mark.parametrize(
        "k, leaf_size",
        [
            (1, 1),
            (3, 5),
            (5, 10),
            (10, 1),
            (1, 100),
            (15, 15),
            (2, 7),
            (7, 2),
            (100, 100),
            (3, 1),
        ],
    )
    def test_init(self, k, leaf_size):
        """Test KNNClassifier initialization."""
        classifier = KNNClassifier(k=k, leaf_size=leaf_size)
        assert classifier.k == k
        assert classifier.leaf_size == leaf_size
        assert classifier.kdtree is None
        assert classifier.point_to_class == {}
        assert classifier._class_number == 0

    @pytest.mark.parametrize(
        "X, Y",
        [
            ([(1.0, 2.0), (3.0, 4.0)], [0, 1]),
            ([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], [1, 0, 1]),
            ([(-1.0, -1.0), (1.0, 1.0)], [0, 0]),
            ([(5.0, 5.0), (6.0, 6.0), (7.0, 7.0), (8.0, 8.0)], [1, 2, 1, 2]),
            ([(0.5, 0.5), (1.5, 1.5)], [0, 1]),
            ([(10.0, 20.0), (30.0, 40.0)], [1, 0]),
            ([(-5.0, -10.0), (-15.0, -20.0)], [0, 1]),
            ([(1.0, 0.0), (0.0, 1.0)], [1, 0]),
            (
                [(2.0, 3.0), (4.0, 5.0), (6.0, 7.0), (8.0, 9.0), (10.0, 11.0)],
                [0, 1, 0, 1, 0],
            ),
            ([(0.0, 0.0)], [0]),
        ],
    )
    def test_fit_valid_input(self, X, Y):
        """Test fit method with valid input data."""
        classifier = KNNClassifier(k=3, leaf_size=10)
        classifier.fit(X, Y)
        assert classifier.kdtree is not None
        assert classifier.point_to_class == dict(zip(X, Y))
        assert classifier._class_number == max(Y) if Y else 0

    @pytest.mark.parametrize(
        "X, Y",
        [
            ([(1.0, 2.0), (3.0, 4.0)], []),
            ([], [0, 1]),
            ([(1.0, 2.0)], [0, 1]),
            ([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)], [0, 1]),
            ([], []),
            ([(1.0, 1.0)], []),
            ([], [1]),
            ([(1.0, 2.0), (3.0, 4.0)], [0]),
            ([(1.0, 2.0)], [0, 1, 2]),
            ([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)], [0, 1, 2]),
        ],
    )
    def test_fit_invalid_input_length_mismatch(self, X, Y):
        """Test fit method with invalid input data (length mismatch)."""
        classifier = KNNClassifier(k=3, leaf_size=10)
        with pytest.raises(ValueError):
            classifier.fit(X, Y)

    def test_get_point_class_valid_point(self):
        """Test _get_point_class method with a valid point."""
        X = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        Y = [0, 1, 0]
        classifier = KNNClassifier(k=3, leaf_size=10)
        classifier.fit(X, Y)
        assert classifier._get_point_class((1.0, 2.0)) == 0
        assert classifier._get_point_class((3.0, 4.0)) == 1
        assert classifier._get_point_class((5.0, 6.0)) == 0

    def test_get_point_class_invalid_point(self):
        """Test _get_point_class method with an invalid point."""
        X = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        Y = [0, 1, 0]
        classifier = KNNClassifier(k=3, leaf_size=10)
        classifier.fit(X, Y)
        with pytest.raises(ValueError):
            classifier._get_point_class((2.0, 3.0))
        with pytest.raises(ValueError):
            classifier._get_point_class((0.0, 0.0))
        with pytest.raises(ValueError):
            classifier._get_point_class((10.0, 10.0))

    def test_get_point_class_empty_training(self):
        """Test _get_point_class method when the classifier is not trained."""
        classifier = KNNClassifier(k=3, leaf_size=10)
        with pytest.raises(ValueError):
            classifier._get_point_class((1.0, 2.0))

    @pytest.mark.parametrize(
        "X_train, Y_train, X_predict",
        [
            ([(1.0, 2.0), (3.0, 4.0)], [0, 1], [(2.0, 3.0)]),
            ([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], [0, 1, 0], [(1.5, 1.5)]),
            ([(-1.0, -1.0), (1.0, 1.0)], [0, 1], [(0.0, 0.0)]),
            (
                [(5.0, 5.0), (6.0, 6.0), (7.0, 7.0), (8.0, 8.0)],
                [0, 1, 0, 1],
                [(6.5, 6.5)],
            ),
            ([(0.5, 0.5), (1.5, 1.5)], [0, 1], [(1.0, 1.0)]),
            ([(10.0, 20.0), (30.0, 40.0)], [0, 1], [(20.0, 30.0)]),
            ([(-5.0, -10.0), (-15.0, -20.0)], [0, 1], [(-10.0, -15.0)]),
            ([(1.0, 0.0), (0.0, 1.0)], [0, 1], [(0.5, 0.5)]),
            ([(2.0, 3.0), (4.0, 5.0), (6.0, 7.0)], [0, 1, 0], [(5.0, 6.0)]),
        ],
    )
    def test_predict_proba_valid_input(self, X_train, Y_train, X_predict):
        """Test predict_proba method with valid input data."""
        classifier = KNNClassifier(k=2, leaf_size=1)
        classifier.fit(X_train, Y_train)
        probas = classifier.predict_proba(X_predict)
        assert isinstance(probas, list)
        assert len(probas) == len(X_predict)
        for point_probas in probas:
            assert isinstance(point_probas, list)
            assert sum(point_probas) == pytest.approx(1.0)
            for proba in point_probas:
                assert 0 <= proba <= 1

    def test_predict_proba_not_fitted(self):
        """Test predict_proba method when the classifier is not fitted."""
        classifier = KNNClassifier(k=3, leaf_size=10)
        with pytest.raises(ValueError):
            classifier.predict_proba([(1.0, 2.0)])

    @pytest.mark.parametrize(
        "X_train, Y_train, X_predict",
        [
            ([(1.0, 2.0), (3.0, 4.0)], [0, 1], [(2.0, 3.0)]),
            ([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], [0, 1, 0], [(1.5, 1.5)]),
            ([(-1.0, -1.0), (1.0, 1.0)], [0, 1], [(0.0, 0.0)]),
            (
                [(5.0, 5.0), (6.0, 6.0), (7.0, 7.0), (8.0, 8.0)],
                [0, 1, 0, 1],
                [(6.5, 6.5)],
            ),
            ([(0.5, 0.5), (1.5, 1.5)], [0, 1], [(1.0, 1.0)]),
            ([(10.0, 20.0), (30.0, 40.0)], [0, 1], [(20.0, 30.0)]),
            ([(-5.0, -10.0), (-15.0, -20.0)], [0, 1], [(-10.0, -15.0)]),
            ([(1.0, 0.0), (0.0, 1.0)], [0, 1], [(0.5, 0.5)]),
            ([(2.0, 3.0), (4.0, 5.0), (6.0, 7.0)], [0, 1, 0], [(5.0, 6.0)]),
            ([(0.0, 0.0)], [0], [(1.0, 1.0)]),
        ],
    )
    def test_predict_valid_input(self, X_train, Y_train, X_predict):
        """Test predict method with valid input data."""
        classifier = KNNClassifier(k=2, leaf_size=1)
        classifier.fit(X_train, Y_train)
        predictions = classifier.predict(X_predict)
        assert isinstance(predictions, list)
        assert len(predictions) == len(X_predict)
        for prediction in predictions:
            assert isinstance(prediction, int)
            assert prediction in Y_train or prediction in range(max(Y_train) + 1) if Y_train else prediction == 0

    def test_predict_not_fitted(self):
        """Test predict method when the classifier is not fitted."""
        classifier = KNNClassifier(k=3, leaf_size=10)
        with pytest.raises(ValueError):
            classifier.predict([(1.0, 2.0)])
