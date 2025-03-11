import pytest

from src.homeworks.KNN.kdt_tree.kdt_Node import KDTNode


class TestKDTNode:
    @pytest.mark.parametrize(
        "points, leaf_size, expected_leaf, expected_points_len",
        [
            ([], 5, True, 0),
            ([(1, 2)], 5, True, 1),
            ([(1, 2), (3, 4)], 5, True, 2),
            ([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)], 5, True, 5),
            ([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)], 5, False, 1),
            (
                [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)],
                3,
                False,
                1,
            ),
            ([(1, 2), (1, 2), (1, 2)], 2, False, 1),
            ([(1, 2), (3, 4), (1, 2), (5, 6)], 2, False, 1),
            (
                [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)],
                4,
                False,
                1,
            ),
            (
                [
                    (1, 2),
                    (3, 4),
                    (5, 6),
                    (7, 8),
                    (9, 10),
                    (11, 12),
                    (13, 14),
                    (15, 16),
                    (17, 18),
                ],
                3,
                False,
                1,
            ),
        ],
    )
    def test_init(self, points, leaf_size, expected_leaf, expected_points_len):
        node = KDTNode(points, leaf_size)
        assert node.leaf_size == leaf_size
        assert node.is_leaf == expected_leaf
        assert len(node.points) == expected_points_len
        if not expected_leaf:
            assert node.left is not None
            assert node.right is not None
        else:
            assert node.left is None
            assert node.right is None

    @pytest.mark.parametrize(
        "points, axis, expected_variance",
        [
            ([(1, 2), (3, 4), (5, 6)], 0, 2.6666666666666665),
            ([(1, 2), (3, 4), (5, 6)], 1, 2.6666666666666665),
            ([(1, 1), (1, 1), (1, 1)], 0, 0.0),
            ([(1, 2), (1, 4), (1, 6)], 1, 2.6666666666666665),
            ([(2, 1), (4, 1), (6, 1)], 0, 2.6666666666666665),
            ([(1, 2)], 0, 0.0),
            ([(1, 2), (2, 2)], 0, 0.25),
            ([(1, 2), (2, 2), (3, 2), (4, 2)], 0, 1.25),
            ([(0, 0), (1, 1), (2, 0), (3, 1)], 0, 1.25),
            ([(5, 1), (5, 2), (5, 3)], 0, 0.0),
        ],
    )
    def test_count_variance(self, points, axis, expected_variance):
        node = KDTNode(points, leaf_size=5)
        variance = node._count_variance(axis)
        assert pytest.approx(variance) == expected_variance

    @pytest.mark.parametrize(
        "points, leaf_size, expected_is_leaf, has_children",
        [
            ([(1, 2), (3, 4)], 3, True, False),
            ([(1, 2), (3, 4), (5, 6)], 2, False, True),
            ([(1, 2), (3, 4), (5, 6), (7, 8)], 2, False, True),
            ([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)], 3, False, True),
            ([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)], 5, True, False),
            ([], 5, True, False),
            ([(1, 2)], 1, True, False),
            ([(1, 2), (3, 4)], 1, False, True),
            ([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)], 3, False, True),
            (
                [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)],
                4,
                False,
                True,
            ),
        ],
    )
    def test_maintain_init_invariance(
        self, points, leaf_size, expected_is_leaf, has_children
    ):
        node = KDTNode(points, leaf_size, is_leaf=True)
        node._maintain_init_invariance()
        assert node.is_leaf == expected_is_leaf
        if has_children:
            assert node.left is not None
            assert node.right is not None
            assert len(node.points) == 1
        else:
            assert node.left is None
            assert node.right is None
            assert len(node.points) == len(points)

    @pytest.mark.parametrize(
        "points, expected_axis",
        [
            ([(1, 2), (3, 4), (5, 6)], 0),
            ([(1, 2), (1, 4), (1, 6)], 1),
            ([(2, 1), (4, 1), (6, 1)], 0),
            ([(1, 1), (2, 2), (3, 1)], 0),
            ([(1, 1), (1, 2), (1, 3)], 1),
            ([(1, 1)], 0),
            ([], 0),
            ([(1, 1), (2, 1), (1, 2), (2, 2)], 0),
            ([(1, 1, 5), (1, 2, 5), (1, 3, 1)], 2),
            ([(5, 1, 1), (1, 2, 1), (1, 3, 1)], 0),
        ],
    )
    def test_chose_axis(self, points, expected_axis):
        node = KDTNode(points, leaf_size=5)
        chosen_axis = node._chose_axis()
        assert chosen_axis == expected_axis
