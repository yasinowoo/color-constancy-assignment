"""
FC4: Fully Convolutional Color Constancy with Confidence-weighted Pooling
(Hu et al., CVPR 2017). SqueezeNet backbone instead of AlexNet.

Reference implementation for inspiration: https://github.com/matteo-rizzo/fc4-pytorch
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.functional import normalize
from torchvision.models import squeezenet1_1


class FC4(nn.Module):
    """FC4 backbone + prediction head from fc4-pytorch."""

    def __init__(self, confidence_weighted_pooling: bool = True):
        super().__init__()
        self.confidence_weighted_pooling = confidence_weighted_pooling

        backbone = squeezenet1_1(weights="DEFAULT").features
        self.backbone = nn.Sequential(*list(backbone)[:12])
        self.head = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True),
            nn.Conv2d(512, 64, kernel_size=6, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 4 if confidence_weighted_pooling else 3, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 3, H, W) image
        Returns:
            (B, 3) unit-norm illuminant estimate
        """
        out = self.head(self.backbone(x))

        if self.confidence_weighted_pooling:
            rgb = normalize(out[:, :3, :, :], dim=1)
            confidence = out[:, 3:4, :, :]
            return normalize(torch.sum(rgb * confidence, dim=(2, 3)), dim=1)

        return normalize(torch.sum(out, dim=(2, 3)), dim=1)

    @torch.no_grad()
    def confidence_map(self, x: Tensor) -> Tensor:
        """
        Confidence map for visualization (only with confidence-weighted pooling).

        Args:
            x: (B, 3, H, W) image
        Returns:
            (B, h, w) confidence map on the feature grid
        """
        if not self.confidence_weighted_pooling:
            raise RuntimeError("Confidence map requires confidence_weighted_pooling=True.")
        return self.head(self.backbone(x))[:, 3, :, :]


def preprocess_raw(raw: Tensor) -> Tensor:
    """
    Bring a linear RAW image closer to the (sRGB-like) domain the ImageNet
    backbone was pretrained on.

    Args:
        raw: (..., 3, H, W) linear RAW
    Returns:
        (..., 3, H, W) gamma-encoded image
    """
    if raw.ndim < 3 or raw.shape[-3] != 3:
        raise ValueError("RAW tensor must have shape [..., 3, H, W].")
    x = raw.to(dtype=torch.float32)
    x = torch.clamp(x, min=0.0)
    return torch.pow(x, 1.0 / 2.2)


class FC4AWB(nn.Module):
    """
    FC4 wrapper:
    raw tensor -> preprocess -> FC4 -> illuminant prediction.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        confidence_weighted_pooling: bool = True,
    ):
        super().__init__()
        self.model = FC4(confidence_weighted_pooling=confidence_weighted_pooling)

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state_dict)

        self.model.eval()

    def forward(self, raw: Tensor) -> Tensor:
        """Return estimated white point(s), shape (B, 3)."""
        return self.model(preprocess_raw(raw))


def angular_loss(pred: Tensor, gt: Tensor) -> Tensor:
    """
    Differentiable angular error between illuminant vectors, used as the FC4 loss.
    Same as the numpy implementation (metrics.py) but rewritten for pytorch.

    Args:
        pred: (B, 3) predicted illuminants
        gt: (B, 3) ground-truth illuminants
    Returns:
        (B,) angles in degrees
    """
    # TODO: Implement (watch out for the gradient of arccos near +-1)
    angle_deg = ...
    return angle_deg
