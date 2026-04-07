import numpy as np
from _deltaE import DeltaE

class ColorCorrectionMetrics:
    @staticmethod
    def angular_error(predicted, target):
        """
        Angular error between predicted and target 3-vectors (degrees).

        Args:
            predicted: Array of shape (..., 3).
            target: Same shape as predicted.

        Returns:
            ndarray: Angles in degrees, shape predicted.shape[:-1].
        """
        angle_deg = ...
        return angle_deg


    @staticmethod
    def reproduction_error(predicted, target):
        """
        Reproduction angular error between predicted and target white points (degrees).

        Args:
            predicted: (..., 3) array of predicted white points
            target: (..., 3) array of target white points

        Returns:
            ndarray: Shape predicted.shape[:-1], angles in degrees.
        """
        angle_deg = ...
        return angle_deg


    @staticmethod
    def deltaE2000(predicted, target):
        """
        Calculate CIEDE2000 (Delta E 00) between predicted and target XYZ colors.

        Args:
            predicted: Array of shape (..., 3) with XYZ (or compatible) values.
            target: Same shape as predicted.

        Returns:
            ndarray: Per-pixel Delta E map, shape predicted.shape[:-1], dtype float32.
        """
        dE_pixelwise: np.ndarray = DeltaE().compute(predicted, target)
        return dE_pixelwise


    @staticmethod
    def evaluate_cst_metrics(predicted, target):
        """
        Evaluate all metrics and return as dictionary
        
        Args:
            predicted: (H, W, 3) array of predicted colors
            target: (H, W, 3) array of target colors
            
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        metrics = {
            'angle': ColorCorrectionMetrics.angular_error(predicted, target),
            'dE': ColorCorrectionMetrics.deltaE2000(predicted, target),
        }
        return metrics


    @staticmethod
    def evaluate_awb_metrics(predicted, target):
        """
        Evaluate AWB metrics for a single white-point estimate.
        
        Args:
            predicted: (3,) array of predicted white point
            target: (3,) array of target white point
            
        Returns:
            dict: Dictionary with AWB metrics ('angle', 'dE', 'reproduction_error')
        """
        predicted = predicted.reshape(1, 1, 3)
        target = target.reshape(1, 1, 3)
        metrics = {
            'angle': ColorCorrectionMetrics.angular_error(predicted, target),
            'dE': ColorCorrectionMetrics.deltaE2000(predicted, target),
            'reproduction_error': ColorCorrectionMetrics.reproduction_error(predicted, target),
        }
        return metrics
