import numpy as np
from pathlib import Path 
from typing import Literal

this_folder = Path(__file__).parent.resolve()
data_folder = this_folder / 'data'

def load_hsi(idx: int):
    return np.load(data_folder / f'hsi/{idx}.npy', allow_pickle=True).item()


def read_illuminants() -> dict[str, np.ndarray]:
    """
    Illuminant spectra for white-point augmentation (FC4 training).

    Returns:
        {name: (bands,) spectrum} on the HSI wavelength grid
    """
    return np.load(data_folder / 'illuminants.npy', allow_pickle=True).item()


def augment_whitepoint(hsi: np.ndarray, whitepatch_mask: np.ndarray,
                       new_illuminant: np.ndarray):
    """
    Relight a scene with a new illuminant: divide HSI by the spectrum of the
    white patch (estimate of the original illuminant) and multiply by the new one.

    Args:
        hsi: (height, width, bands)
        whitepatch_mask: (height, width) boolean mask of the white patch
        new_illuminant: (bands,) spectrum of the new illuminant
    Returns:
        (height, width, bands) relit HSI
    """
    spectra_whitepoint = ...
    return hsi / spectra_whitepoint * new_illuminant


HSI_SHAPE = (400, 700, 10)  # HSI wavelength range: 400-700 nm, 10 nm step -> 31 bands


def load_sens(name: Literal['xyz_matching_fun', 'canon600d']):
    """
    Load spectral sensitivities from colour-science, resampled to the HSI grid
    and normalised so that the green (Y / G) channel peaks at 1.

    Returns:
        (bands, 3) sensitivities
    """
    import colour

    if name == 'xyz_matching_fun':
        msds = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    elif name == 'canon600d':
        import colour_datasets
        # Jiang et al. (2013) camera spectral sensitivity database
        msds = colour_datasets.load('3245883')['Canon 600D']
    else:
        raise ValueError(f'Unknown sensitivity: {name}')

    msds = msds.copy().align(colour.SpectralShape(*HSI_SHAPE))
    sens = np.asarray(msds.values, dtype=np.float32)
    return sens / sens[:, 1].max()


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

