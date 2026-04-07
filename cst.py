import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def _flatten_colors(arr: np.ndarray):
    arr = np.asarray(arr)
    if arr.ndim < 2 or arr.shape[-1] != 3:
        raise ValueError("Input must have shape (N, 3) or (..., 3)")
    if arr.ndim == 2:
        return arr, None
    return arr.reshape(-1, 3), arr.shape


def _restore_colors(flat: np.ndarray, original_shape):
    if original_shape is None:
        return flat
    return flat.reshape(original_shape)


class PolynomialColorCorrection:
    """
    Polynomial color correction
    """
    
    def __init__(self, degree=2):
        self.degree = degree
        self.regressor = None

    def _transform_features(self, X):
        transformed = ...
        return transformed

    def fit(self, src, dst):
        """
        Fit transformation from source to destination colors.

        Args:
            src: (*, 3) array of source colors
            dst: Same shape as src, corresponding destination colors.
        """
        src_flat, _ = _flatten_colors(src)
        dst_flat, _ = _flatten_colors(dst)
        if src_flat.shape != dst_flat.shape:
            raise ValueError("Source and destination shapes must match")

        # Create polynomial features
        transformed_src = self._transform_features(src_flat)
        
        # Fit linear regression
        self.regressor = LinearRegression(fit_intercept=False)
        self.regressor.fit(transformed_src, dst_flat)
        return self


    def transform(self, input):
        """
        Transform source colors using learned transformation

        Args:
            input: (*, 3) array of source colors

        Returns:
            Array of transformed colors with the same shape as input.
        """
        if self.regressor is None:
            raise ValueError("Model must be fitted before transformation")

        input_flat, input_shape = _flatten_colors(input)
        input_transformed = self._transform_features(input_flat)
        predicted = self.regressor.predict(input_transformed)
        return _restore_colors(predicted, input_shape)


    def fit_transform(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step

        Args:
            src: (*, 3) array of source domain colors
            dst: Same shape as src, target domain colors

        Returns:
            Transformed colors with the same shape as src.
        """
        src_arr = np.asarray(src)
        dst_arr = np.asarray(dst)
        if src_arr.shape != dst_arr.shape:
            raise ValueError("Source and destination shapes must match")
        return self.fit(src_arr, dst_arr).transform(src_arr)


class RootPolynomialColorCorrection:
    """
    Root-Polynomial Color Correction (RPCC) based on:
    "Colour Correction using Root-Polynomial Regression" by Finlayson et al.
    """
    
    def __init__(self, degree=2):
        self.degree = degree
        self.regressor = None
        self.terms = None

    def _transform_features(self, X):
        """Transform input features using root-polynomial terms"""
        transformed = ...
        return transformed

    def fit(self, src, dst):
        """
        Fit transformation from source to destination colors.

        Args:
            src: (*, 3) array of source colors
            dst: Same shape as src, corresponding destination colors.
        """
        src_flat, _ = _flatten_colors(src)
        dst_flat, _ = _flatten_colors(dst)
        if src_flat.shape != dst_flat.shape:
            raise ValueError("Source and destination shapes must match")

        # Transform features
        transformed_src = self._transform_features(src_flat)

        # Fit linear regression
        self.regressor = LinearRegression(fit_intercept=False)
        self.regressor.fit(transformed_src, dst_flat)
        return self

    def transform(self, input):
        """
        Transform source colors using learned transformation

        Args:
            input: (*, 3) array of source colors

        Returns:
            Array of transformed colors with the same shape as input.
        """
        if self.regressor is None:
            raise ValueError("Model must be fitted before transformation")

        input_flat, input_shape = _flatten_colors(input)
        input_transformed = self._transform_features(input_flat)
        predicted = self.regressor.predict(input_transformed)
        return _restore_colors(predicted, input_shape)

    def fit_transform(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step

        Args:
            src: (*, 3) array of source domain colors
            dst: Same shape as src, target domain colors

        Returns:
            Transformed colors with the same shape as src.
        """
        src_arr = np.asarray(src)
        dst_arr = np.asarray(dst)
        if src_arr.shape != dst_arr.shape:
            raise ValueError("Source and destination shapes must match")
        return self.fit(src_arr, dst_arr).transform(src_arr)


class ScalableRationalColorCorrection:
    """
    Scalable-Rational Color Correction (SRCC) based on:
    "A comparative study of color space transform methods" by Soshin et al.
    """
    
    def __init__(self, degree=2, eps=1e-9):
        self.degree = degree
        self.eps = eps
        self.regressor = None
        self.terms = None
        
    
    def _transform_features(self, X):
        """Transform input features using scalable-rational terms"""
        transformed = ...
        return transformed
    
    def fit(self, src, dst):
        """
        Fit transformation from source to destination colors.

        Args:
            src: (*, 3) array of source colors
            dst: Same shape as src, corresponding destination colors.
        """
        src_flat, _ = _flatten_colors(src)
        dst_flat, _ = _flatten_colors(dst)
        if src_flat.shape != dst_flat.shape:
            raise ValueError("Source and destination shapes must match")

        # Transform features
        transformed_src = self._transform_features(src_flat)

        # Fit linear regression
        self.regressor = LinearRegression(fit_intercept=False)
        self.regressor.fit(transformed_src, dst_flat)
        return self

    def transform(self, input):
        """
        Transform source colors using learned transformation

        Args:
            input: (*, 3) array of source colors

        Returns:
            Array of transformed colors with the same shape as input.
        """
        if self.regressor is None:
            raise ValueError("Model must be fitted before transformation")

        input_flat, input_shape = _flatten_colors(input)
        input_transformed = self._transform_features(input_flat)
        predicted = self.regressor.predict(input_transformed)
        return _restore_colors(predicted, input_shape)

    def fit_transform(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step

        Args:
            src: (*, 3) array of source domain colors
            dst: Same shape as src, target domain colors

        Returns:
            Transformed colors with the same shape as src.
        """
        src_arr = np.asarray(src)
        dst_arr = np.asarray(dst)
        if src_arr.shape != dst_arr.shape:
            raise ValueError("Source and destination shapes must match")
        return self.fit(src_arr, dst_arr).transform(src_arr)


