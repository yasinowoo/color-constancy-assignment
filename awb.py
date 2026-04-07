import numpy as np 


def GrayWorld(image: np.ndarray):
    """
    GrayWorld algorithm for automatic white balance.
    Args:
        image: Image (height, width, 3)
    Returns:
        white_point: Estimated white point multipliers normalized
            by the green channel, shape (3,)
    """
    white_point = ...
    return white_point / white_point[1]


def MaxRGB(image: np.ndarray):
    """
    MaxRGB algorithm for automatic white balance.
    Args:
        image: Image (height, width, 3)
    Returns:
        white_point: Estimated white point multipliers normalized
            by the green channel, shape (3,)
    """
    white_point = ...
    return white_point / white_point[1]


def GrayEdge(image: np.ndarray):
    """
    Gray-Edge style AWB: average RGB on high-gradient pixels (luminance), then
    normalize by the green channel

    Args:
        image: Image (height, width, 3)

    Returns:
        white_point: Shape (3,), illuminant estimate normalized by green.
    """
    white_point = ...
    return white_point / white_point[1]
