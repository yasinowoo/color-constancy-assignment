import numpy as np
from pathlib import Path 
from typing import Literal

this_folder = Path(__file__).parent.resolve()
data_folder = this_folder / 'data'

def load_hsi(idx: int):
    return np.load(data_folder / f'hsi/{idx}.npy', allow_pickle=True).item()


def load_sens(name: Literal['xyz_matching_fun', 'canon600d']):
    return np.load(data_folder / f'sensitivity/{name}.npy')


def get_color_responses(hsi: np.ndarray, sens: np.ndarray):
    """
    Project HSI  onto sensor sensitivities.

    Args:
        hsi: (height, width, bands)
        sens: (bands, 3)

    Returns:
        (height, width, 3) color responses
    """
    colors = ...
    return colors


XYZ2LINRGB = np.array([[3.2406, -1.5372, -0.4986],
                    [-0.9689, 1.8758, 0.0416],
                    [0.0557, -0.204, 1.0570]], dtype=np.float32).T

def xyz2linrgb(xyz: np.ndarray):
    """
    Convert XYZ to linear RGB.
    Args:
        xyz: XYZ values (height, width, 3)
    Returns:
        linrgb: Linear RGB values (height, width, 3)
    """
    linrgb = ...
    return linrgb.astype(np.float32)

def linrgb2srgb(linrgb: np.ndarray):
    """Apply sRGB gamma.

    Args:
        linrgb: Linear RGB values (height, width, 3)
    Returns:
        srgb: sRGB values (height, width, 3)
    """
    srgb = ...
    return srgb.astype(np.float32)

def xyz2srgb(xyz: np.ndarray):
    """
    Convert XYZ to sRGB.
    Args:
        xyz: XYZ values (height, width, 3)
    Returns:
        srgb: sRGB values (height, width, 3)
    """
    return linrgb2srgb(xyz2linrgb(xyz)).astype(np.float32)

