import numpy as np
import pytest

from awb import GrayEdge, GrayWorld, MaxRGB


def test_grayworld_returns_white_point_vector():
    image = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 2.0, 6.0]],
            [[7.0, 2.0, 9.0], [10.0, 2.0, 12.0]],
        ],
        dtype=np.float32,
    )
    white_point = GrayWorld(image)
    assert white_point.shape == (3,)
    assert np.isclose(white_point[1], 1.0)


def test_maxrgb_returns_white_point_vector():
    image = np.array(
        [
            [[0.5, 1.0, 1.5], [0.2, 1.0, 2.0]],
            [[0.1, 1.0, 0.5], [1.5, 1.0, 0.3]],
        ],
        dtype=np.float32,
    )
    white_point = MaxRGB(image)
    assert white_point.shape == (3,)
    assert np.isclose(white_point[1], 1.0)


@pytest.mark.parametrize(
    "image, expected",
    [
        (
            np.array(
                [
                    [[2.0, 1.0, 4.0], [4.0, 1.0, 2.0]],
                    [[6.0, 1.0, 6.0], [8.0, 1.0, 8.0]],
                ],
                dtype=np.float32,
            ),
            np.array([5.0, 1.0, 5.0], dtype=np.float32),
        ),
        (
            np.array(
                [
                    [[1.0, 2.0, 3.0], [3.0, 2.0, 5.0]],
                    [[5.0, 2.0, 7.0], [7.0, 2.0, 9.0]],
                ],
                dtype=np.float32,
            ),
            np.array([2.0, 1.0, 3.0], dtype=np.float32),
        ),
    ],
)
def test_grayworld_hardcoded_expected_values(image, expected):
    white_point = GrayWorld(image)
    np.testing.assert_allclose(white_point, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "image, expected",
    [
        (
            np.array(
                [
                    [[2.0, 1.0, 4.0], [4.0, 1.0, 2.0]],
                    [[6.0, 1.0, 6.0], [8.0, 1.0, 8.0]],
                ],
                dtype=np.float32,
            ),
            np.array([8.0, 1.0, 8.0], dtype=np.float32),
        ),
        (
            np.array(
                [
                    [[1.0, 4.0, 3.0], [5.0, 4.0, 2.0]],
                    [[4.0, 4.0, 10.0], [2.0, 4.0, 8.0]],
                ],
                dtype=np.float32,
            ),
            np.array([1.25, 1.0, 2.5], dtype=np.float32),
        ),
    ],
)
def test_maxrgb_hardcoded_expected_values(image, expected):
    white_point = MaxRGB(image)
    np.testing.assert_allclose(white_point, expected, rtol=1e-6, atol=1e-6)

