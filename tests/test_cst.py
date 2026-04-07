import numpy as np
import pytest

from cst import (
    PolynomialColorCorrection,
    RootPolynomialColorCorrection,
    ScalableRationalColorCorrection,
)


@pytest.mark.parametrize(
    "model_cls",
    [PolynomialColorCorrection, RootPolynomialColorCorrection, ScalableRationalColorCorrection],
)
def test_fit_transform_preserves_image_shape(model_cls):
    src = np.random.RandomState(0).rand(4, 5, 3).astype(np.float32)
    dst = np.random.RandomState(1).rand(4, 5, 3).astype(np.float32)
    model = model_cls(degree=2)

    output = model.fit_transform(src, dst)

    assert output.shape == src.shape


@pytest.mark.parametrize(
    "model_cls",
    [PolynomialColorCorrection, RootPolynomialColorCorrection, ScalableRationalColorCorrection],
)
def test_fit_transform_requires_matching_shapes(model_cls):
    src = np.random.RandomState(0).rand(4, 5, 3).astype(np.float32)
    dst = np.random.RandomState(1).rand(4, 6, 3).astype(np.float32)
    model = model_cls(degree=2)

    with pytest.raises(ValueError, match="shapes must match"):
        model.fit_transform(src, dst)


@pytest.mark.parametrize(
    "model",
    [
        PolynomialColorCorrection(degree=2),
        RootPolynomialColorCorrection(degree=2),
        ScalableRationalColorCorrection(degree=2),
    ],
)
def test_transform_requires_fit(model):
    input_colors = np.random.RandomState(2).rand(10, 3).astype(np.float32)
    with pytest.raises(ValueError, match="must be fitted"):
        model.transform(input_colors)


@pytest.mark.parametrize(
    "model_cls,input_shape",
    [
        (PolynomialColorCorrection, (20, 3)),
        (PolynomialColorCorrection, (4, 5, 3)),
        (RootPolynomialColorCorrection, (20, 3)),
        (RootPolynomialColorCorrection, (4, 5, 3)),
        (ScalableRationalColorCorrection, (20, 3)),
        (ScalableRationalColorCorrection, (4, 5, 3)),
    ],
)
def test_fit_and_transform_support_2d_and_3d_inputs(model_cls, input_shape):
    src = np.random.RandomState(3).rand(*input_shape).astype(np.float32)
    dst = np.random.RandomState(4).rand(*input_shape).astype(np.float32)
    model = model_cls(degree=2)

    model.fit(src, dst)
    output = model.transform(src)

    assert output.shape == src.shape


@pytest.mark.parametrize(
    "model_cls, expected",
    [
        (
            PolynomialColorCorrection,
            np.array(
                [
                    [0.3565880736, 0.2535817939, 0.5563320607],
                    [0.6525852199, 0.1805836952, 0.3505387032],
                    [0.3120413463, 0.7762191521, 0.7119078805],
                ],
                dtype=np.float64,
            ),
        ),
        (
            RootPolynomialColorCorrection,
            np.array(
                [
                    [0.3514090687, 0.2633399997, 0.5514090687],
                    [0.5982066680, 0.2956319059, 0.2982066680],
                    [0.5303892477, 0.2381079184, 0.9303892477],
                ],
                dtype=np.float64,
            ),
        ),
        (
            ScalableRationalColorCorrection,
            np.array(
                [
                    [0.3447329602, 0.2827073016, 0.5447329602],
                    [0.5502597779, 0.4275805925, 0.2502597779],
                    [0.5580377839, 0.1268034699, 0.9580377839],
                ],
                dtype=np.float64,
            ),
        ),
    ],
)
def test_transform_matches_hardcoded_values(model_cls, expected):
    src = np.array(
        [
            [0.2, 0.3, 0.4],
            [0.6, 0.1, 0.2],
            [0.3, 0.8, 0.5],
            [0.9, 0.7, 0.1],
            [0.4, 0.5, 0.9],
            [0.1, 0.2, 0.7],
        ],
        dtype=np.float64,
    )
    dst = np.array(
        [
            [0.3, 0.2, 0.5],
            [0.7, 0.2, 0.3],
            [0.4, 0.9, 0.6],
            [1.0, 0.8, 0.2],
            [0.5, 0.6, 1.0],
            [0.2, 0.3, 0.8],
        ],
        dtype=np.float64,
    )
    probe = np.array(
        [
            [0.25, 0.35, 0.45],
            [0.55, 0.15, 0.25],
            [0.2, 0.7, 0.6],
        ],
        dtype=np.float64,
    )

    model = model_cls(degree=2)
    model.fit(src, dst)
    predicted = model.transform(probe)

    np.testing.assert_allclose(predicted, expected, rtol=1e-8, atol=1e-8)
