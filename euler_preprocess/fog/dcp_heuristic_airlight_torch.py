from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# NTSC / BT.601 luminance weights
_GRAY_WEIGHTS = torch.tensor([0.2989, 0.5870, 0.1140])
_BRIGHT_SKY_QUANTILE = 0.75
_WHITE_TARGET = torch.ones(3)
_DEFAULT_COOL_TARGET = torch.tensor([0.93, 0.97, 1.0])


class DCPHeuristicAirlightTorch:
    """GPU variant of :class:`DCPHeuristicAirlight` using pure torch operations.

    Uses the Dark Channel Prior with a robust pooled heuristic:
    the brighter half of the top *top_percent* dark-channel pixels are
    averaged with luminance weights. When a sky mask is available, the
    brightest sky pixels provide the chromaticity prior and the DCP
    estimate contributes the target luminance.

    Input must be a float32 ``(H, W, 3)`` tensor already on the target
    device.  Returns a ``(3,)`` tensor on the same device.
    """

    def __init__(
        self,
        patch_size: int = 15,
        top_percent: float = 0.001,
        white_bias: float = 0.0,
        cool_bias: float = 0.0,
        cool_target: torch.Tensor | None = None,
    ) -> None:
        patch_size = int(patch_size)
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        top_percent = float(top_percent)
        if not 0.0 < top_percent <= 1.0:
            raise ValueError("top_percent must be in (0, 1]")
        white_bias = float(white_bias)
        cool_bias = float(cool_bias)
        if not 0.0 <= white_bias <= 1.0:
            raise ValueError("white_bias must be in [0, 1]")
        if not 0.0 <= cool_bias <= 1.0:
            raise ValueError("cool_bias must be in [0, 1]")
        if white_bias + cool_bias > 1.0:
            raise ValueError("white_bias + cool_bias must be <= 1")
        self.patch_size = patch_size
        self.top_percent = top_percent
        self.white_bias = white_bias
        self.cool_bias = cool_bias
        self.cool_target = self._prepare_color(
            _DEFAULT_COOL_TARGET if cool_target is None else cool_target
        )

    def compute(self, rgb: torch.Tensor) -> torch.Tensor:
        """Compute airlight from an ``(H, W, 3)`` float tensor."""
        image = self._prepare_rgb(rgb)
        candidate_rgb, candidate_gray = self._candidate_pixels(image)
        airlight = self._estimate_from_candidates(candidate_rgb, candidate_gray)
        return self._apply_color_bias(airlight, reference_color=airlight)

    def __call__(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.compute(rgb)

    def estimate_airlight(
        self,
        image: torch.Tensor,
        sky_mask: torch.Tensor,
        sample_id: str | None = None,
    ) -> torch.Tensor:
        del sample_id
        prepared = self._prepare_rgb(image)
        candidate_rgb, candidate_gray = self._candidate_pixels(prepared)
        airlight = self._estimate_from_candidates(candidate_rgb, candidate_gray)
        sky_prior = self._estimate_sky_prior(prepared, sky_mask)
        if sky_prior is None:
            return self._apply_color_bias(airlight, reference_color=airlight)
        merged = self._merge_with_sky_prior(airlight, sky_prior)
        return self._apply_color_bias(merged, reference_color=sky_prior)

    def _brightest_pixels_count(self, num_pixels: int) -> int:
        count = int(math.floor(self.top_percent * num_pixels))
        # Force odd so the median corresponds to an actual pixel
        count = count + ((count + 1) % 2)
        return max(1, min(count, num_pixels))

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

    @staticmethod
    def _prepare_color(value: torch.Tensor) -> torch.Tensor:
        color = torch.as_tensor(value, dtype=torch.float32)
        if color.ndim != 1 or color.shape[0] != 3:
            raise ValueError("cool_target must be a length-3 color")
        color = torch.nan_to_num(color, nan=0.0, posinf=0.0, neginf=0.0)
        if float(color.max().item()) > 1.0:
            color = color / 255.0
        return torch.clamp(color, 0.0, 1.0)

    def _candidate_pixels(
        self,
        rgb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dark_channel = self._dark_channel(rgb)

        num_pixels = dark_channel.numel()
        if num_pixels == 0:
            raise ValueError("rgb image is empty")

        num_select = self._brightest_pixels_count(num_pixels)
        flat_dark = dark_channel.reshape(-1)
        _, candidate_idx = torch.topk(flat_dark, num_select)

        flat_rgb = rgb.reshape(-1, 3)
        candidate_rgb = flat_rgb[candidate_idx]
        weights = _GRAY_WEIGHTS.to(device=rgb.device, dtype=rgb.dtype)
        candidate_gray = (candidate_rgb * weights).sum(dim=1)
        return candidate_rgb, candidate_gray

    def _estimate_from_candidates(
        self,
        candidate_rgb: torch.Tensor,
        candidate_gray: torch.Tensor,
    ) -> torch.Tensor:
        median_intensity = torch.median(candidate_gray)
        bright_mask = candidate_gray >= median_intensity
        if not torch.any(bright_mask):
            bright_mask = torch.ones_like(candidate_gray, dtype=torch.bool)

        bright_rgb = candidate_rgb[bright_mask]
        bright_gray = candidate_gray[bright_mask]
        weights = bright_gray - bright_gray.min()
        weights = weights + torch.finfo(bright_gray.dtype).eps
        weighted_sum = (bright_rgb * weights[:, None]).sum(dim=0)
        return weighted_sum / weights.sum()

    def _estimate_sky_prior(
        self,
        rgb: torch.Tensor,
        sky_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if sky_mask is None:
            return None

        mask = sky_mask.to(device=rgb.device, dtype=torch.bool)
        if tuple(mask.shape) != tuple(rgb.shape[:2]):
            raise ValueError("sky_mask must have shape (H, W)")
        if not torch.any(mask):
            return None

        sky_pixels = rgb[mask]
        weights = _GRAY_WEIGHTS.to(device=rgb.device, dtype=rgb.dtype)
        sky_gray = (sky_pixels * weights).sum(dim=1)
        threshold = self._quantile_value(sky_gray, _BRIGHT_SKY_QUANTILE)
        bright_sky = sky_pixels[sky_gray >= threshold]
        if bright_sky.numel() == 0:
            bright_sky = sky_pixels
        return bright_sky.mean(dim=0)

    def _merge_with_sky_prior(
        self,
        airlight: torch.Tensor,
        sky_prior: torch.Tensor,
    ) -> torch.Tensor:
        sky_luminance = self._luminance(sky_prior)
        if sky_luminance <= torch.finfo(sky_prior.dtype).eps:
            return airlight

        target_luminance = max(self._luminance(airlight), sky_luminance)
        return sky_prior * (target_luminance / sky_luminance)

    def _apply_color_bias(
        self,
        airlight: torch.Tensor,
        reference_color: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.white_bias == 0.0 and self.cool_bias == 0.0:
            return airlight

        reference = airlight if reference_color is None else reference_color
        base_weight = 1.0 - self.white_bias - self.cool_bias
        white_target = _WHITE_TARGET.to(device=airlight.device, dtype=airlight.dtype)
        cool_target = self._correlated_cool_target(reference).to(
            device=airlight.device,
            dtype=airlight.dtype,
        )
        biased = (
            base_weight * airlight
            + self.white_bias * white_target
            + self.cool_bias * cool_target
        )
        airlight_luminance = self._luminance(airlight)
        biased_luminance = self._luminance(biased)
        if (
            airlight_luminance > torch.finfo(airlight.dtype).eps
            and biased_luminance > torch.finfo(airlight.dtype).eps
        ):
            biased = biased * (airlight_luminance / biased_luminance)
        return biased

    def _correlated_cool_target(self, reference_color: torch.Tensor) -> torch.Tensor:
        reference = self._prepare_color(reference_color)
        cool_target = self.cool_target.to(
            device=reference.device,
            dtype=reference.dtype,
        )
        return 0.5 * reference + 0.5 * cool_target

    @staticmethod
    def _quantile_value(values: torch.Tensor, quantile: float) -> torch.Tensor:
        sorted_values, _ = torch.sort(values)
        index = int(math.ceil(quantile * sorted_values.numel()) - 1)
        index = min(max(index, 0), sorted_values.numel() - 1)
        return sorted_values[index]

    @staticmethod
    def _luminance(rgb: torch.Tensor) -> float:
        weights = _GRAY_WEIGHTS.to(device=rgb.device, dtype=rgb.dtype)
        return float((rgb * weights).sum().item())

    def _dark_channel(self, rgb: torch.Tensor) -> torch.Tensor:
        min_channel = rgb.min(dim=-1).values
        if self.patch_size == 1:
            return min_channel
        pad = self.patch_size // 2
        # Replicate-pad first (matching numpy edge padding), then pool
        neg = (-min_channel).unsqueeze(0).unsqueeze(0)
        neg = F.pad(neg, (pad, pad, pad, pad), mode="replicate")
        pooled = F.max_pool2d(neg, kernel_size=self.patch_size, stride=1)
        return -pooled.squeeze(0).squeeze(0)
