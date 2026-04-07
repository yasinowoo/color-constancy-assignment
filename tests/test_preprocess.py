import numpy as np

from preprocess import get_color_responses, linrgb2srgb, xyz2linrgb, xyz2srgb


def test_get_color_responses_shape():
    hsi = np.random.RandomState(0).rand(4, 5, 7).astype(np.float32)
    sens = np.random.RandomState(1).rand(7, 3).astype(np.float32)

    colors = get_color_responses(hsi, sens)

    assert colors.shape == (4, 5, 3)


def test_xyz2linrgb_shape_dtype_and_range():
    xyz = np.array(
        [
            [[1.2, -0.1, 0.2], [0.5, 0.5, 0.5]],
            [[0.2, 0.1, 1.4], [0.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    linrgb = xyz2linrgb(xyz)

    assert linrgb.shape == xyz.shape
    assert linrgb.dtype == np.float32
    assert np.all(linrgb >= 0.0)
    assert np.all(linrgb <= 1.0)


def test_linrgb2srgb_shape_dtype_and_range():
    linrgb = np.array(
        [
            [[0.0, 0.0031308, 0.5], [1.0, 0.1, 0.01]],
            [[0.2, 0.4, 0.8], [0.001, 0.9, 0.7]],
        ],
        dtype=np.float32,
    )

    srgb = linrgb2srgb(linrgb.copy())

    assert srgb.shape == linrgb.shape
    assert srgb.dtype == np.float32
    assert np.all(srgb >= 0.0)
    assert np.all(srgb <= 1.0)


def test_xyz2srgb_shape_dtype_and_range():
    xyz = np.random.RandomState(2).rand(3, 2, 3).astype(np.float32)

    srgb = xyz2srgb(xyz)

    assert srgb.shape == xyz.shape
    assert srgb.dtype == np.float32
    assert np.all(srgb >= 0.0)
    assert np.all(srgb <= 1.0)
