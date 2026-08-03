from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class DCPAirlightTorch:
    """GPU variant of :class:`DCPAirlight` using pure torch operations.

    Estimates atmospheric light via the Dark Channel Prior.  Selects the
    top *top_percent* pixels from the dark channel and returns the RGB
    value of the one with the highest channel-sum intensity.

    Input must be a float32 ``(H, W, 3)`` tensor already on the target
    device.  Returns a ``(3,)`` tensor on the same device.
    """

    def __init__(self, patch_size: int = 15, top_percent: float = 0.001) -> None:
        patch_size = int(patch_size)
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        top_percent = float(top_percent)
        if not 0.0 < top_percent <= 1.0:
            raise ValueError("top_percent must be in (0, 1]")
        self.patch_size = patch_size
        self.top_percent = top_percent

    def compute(self, rgb: torch.Tensor) -> torch.Tensor:
        """Compute airlight from an ``(H, W, 3)`` float tensor."""
        image = self._prepare_rgb(rgb)
        dark_channel = self._dark_channel(image)

        num_pixels = dark_channel.numel()
        if num_pixels == 0:
            raise ValueError("rgb image is empty")

        num_select = max(1, int(math.ceil(num_pixels * self.top_percent)))
        num_select = min(num_select, num_pixels)

        flat_dark = dark_channel.reshape(-1)
        _, candidate_idx = torch.topk(flat_dark, num_select)

        intensity = image.reshape(-1, 3).sum(dim=1)
        best_local = torch.argmax(intensity[candidate_idx])
        best_idx = candidate_idx[best_local]

        return image.reshape(-1, 3)[best_idx]

    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.compute(rgb)

    @staticmethod
    def _prepare_rgb(rgb: torch.Tensor) -> torch.Tensor:
        image = rgb
        if image.ndim == 2:
            image = image.unsqueeze(-1).expand(-1, -1, 3)
        elif image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("rgb image must have shape (H, W, 3)")
        image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        return image.to(torch.float32)

    def _dark_channel(self, rgb: torch.Tensor) -> torch.Tensor:
        # min across colour channels → (H, W)
        min_channel = rgb.min(dim=-1).values
        if self.patch_size == 1:
            return min_channel
        pad = self.patch_size // 2
        # max_pool2d of negated image ≡ min-pool
        # Replicate-pad first (matching numpy edge padding), then pool
        # input shape for max_pool2d: (1, 1, H, W)
        neg = (-min_channel).unsqueeze(0).unsqueeze(0)
        neg = F.pad(neg, (pad, pad, pad, pad), mode="replicate")
        pooled = F.max_pool2d(neg, kernel_size=self.patch_size, stride=1)
        return -pooled.squeeze(0).squeeze(0)
