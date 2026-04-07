import numpy as np

from metrics import ColorCorrectionMetrics


def test_angular_error_returns_hw_shape():
    predicted = np.ones((3, 4, 3), dtype=np.float32)
    target = np.ones((3, 4, 3), dtype=np.float32)

    angle = ColorCorrectionMetrics.angular_error(predicted, target)

    assert angle.shape == (3, 4)


def test_deltae2000_returns_hw_shape():
    predicted = np.ones((2, 3, 3), dtype=np.float32)
    target = np.ones((2, 3, 3), dtype=np.float32)

    de = ColorCorrectionMetrics.deltaE2000(predicted, target)

    assert de.shape == (2, 3)


def test_reproduction_error_returns_vector_shape():
    predicted = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    target = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)

    repro = ColorCorrectionMetrics.reproduction_error(predicted, target)

    assert repro.shape == (1,)


def test_evaluate_metrics_have_expected_keys_and_shapes():
    predicted = np.ones((2, 2, 3), dtype=np.float32)
    target = np.ones((2, 2, 3), dtype=np.float32)
    predicted_white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    target_white = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    cst_metrics = ColorCorrectionMetrics.evaluate_cst_metrics(predicted, target)
    awb_metrics = ColorCorrectionMetrics.evaluate_awb_metrics(predicted_white, target_white)

    assert set(cst_metrics.keys()) == {"angle", "dE"}
    assert cst_metrics["angle"].shape == (2, 2)
    assert cst_metrics["dE"].shape == (2, 2)

    assert set(awb_metrics.keys()) == {"angle", "dE", "reproduction_error"}
    assert awb_metrics["angle"].shape == (1, 1)
    assert awb_metrics["dE"].shape == (1, 1)
    assert awb_metrics["reproduction_error"].shape == (1, 1)


def test_angular_error_hardcoded_values():
    predicted = np.array(
        [
            [[0.9, 0.2, 0.3], [0.4, 0.7, 0.1]],
            [[0.2, 0.3, 0.8], [0.6, 0.5, 0.4]],
        ],
        dtype=np.float64,
    )
    target = np.array(
        [
            [[0.8, 0.3, 0.2], [0.5, 0.6, 0.2]],
            [[0.3, 0.2, 0.7], [0.5, 0.4, 0.6]],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [9.1240183860, 12.2778911860],
            [10.2100535804, 16.0462345345],
        ],
        dtype=np.float64,
    )

    result = ColorCorrectionMetrics.angular_error(predicted, target)
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-8)


def test_deltae2000_hardcoded_values():
    predicted = np.array(
        [
            [[0.9, 0.2, 0.3], [0.4, 0.7, 0.1]],
            [[0.2, 0.3, 0.8], [0.6, 0.5, 0.4]],
        ],
        dtype=np.float64,
    )
    target = np.array(
        [
            [[0.8, 0.3, 0.2], [0.5, 0.6, 0.2]],
            [[0.3, 0.2, 0.7], [0.5, 0.4, 0.6]],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [14.126417, 16.754913],
            [48.160023, 19.08776],
        ],
        dtype=np.float64,
    )

    result = ColorCorrectionMetrics.deltaE2000(predicted, target)
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)


def test_reproduction_error_hardcoded_values():
    predicted = np.array([[1.2, 1.0, 0.8]], dtype=np.float64)
    target = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
    expected = np.array([9.2745009598], dtype=np.float64)

    result = ColorCorrectionMetrics.reproduction_error(predicted, target)
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-8)


def test_awb_metrics_hardcoded_values():
    predicted_white = np.array([1.2, 1.0, 0.8], dtype=np.float64)
    target_white = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    metrics = ColorCorrectionMetrics.evaluate_awb_metrics(predicted_white, target_white)

    np.testing.assert_allclose(metrics["angle"], np.array([[9.2745002092]]), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(metrics["dE"], np.array([[16.175917]]), rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(
        metrics["reproduction_error"],
        np.array([[9.2745009598]]),
        rtol=1e-8,
        atol=1e-8,
    )
