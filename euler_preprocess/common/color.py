from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional
    torch = None


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """Convert display-encoded sRGB values in ``[0, 1]`` to scene-linear RGB."""
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return np.where(
        img <= 0.04045,
        img / 12.92,
        ((img + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32, copy=False)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Convert scene-linear RGB values in ``[0, 1]`` to display sRGB."""
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * np.power(img, 1.0 / 2.4) - 0.055,
    ).astype(np.float32, copy=False)


def srgb_to_linear_torch(image):
    if torch is None:  # pragma: no cover - defensive
        raise RuntimeError("Torch color conversion requested but torch is unavailable")
    img = torch.clamp(image, 0.0, 1.0)
    return torch.where(
        img <= 0.04045,
        img / 12.92,
        torch.pow((img + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb_torch(image):
    if torch is None:  # pragma: no cover - defensive
        raise RuntimeError("Torch color conversion requested but torch is unavailable")
    img = torch.clamp(image, 0.0, 1.0)
    return torch.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * torch.pow(img, 1.0 / 2.4) - 0.055,
    )
