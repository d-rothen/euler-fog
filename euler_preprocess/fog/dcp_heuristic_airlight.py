from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_GRAY_WEIGHTS = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
_BRIGHT_SKY_QUANTILE = 0.75
_WHITE_TARGET = np.ones(3, dtype=np.float32)
_DEFAULT_COOL_TARGET = np.array([0.93, 0.97, 1.0], dtype=np.float32)


class DCPHeuristicAirlight:
    """Estimate atmospheric light via Dark Channel Prior with a robust heuristic.

    Like :class:`DCPAirlight`, this selects the top bright pixels from the dark
    channel. Instead of collapsing to a single candidate pixel, it keeps the
    brighter half of the candidate pool and averages them with luminance
    weights. When a sky mask is available, the brightest sky pixels provide the
    chromaticity prior and the DCP estimate contributes the target luminance.
    """

    def __init__(
        self,
        patch_size: int = 15,
        top_percent: float = 0.001,
        white_bias: float = 0.0,
        cool_bias: float = 0.0,
        cool_target: np.ndarray | None = None,
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

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        return self.compute(rgb)

    def compute(self, rgb: np.ndarray) -> np.ndarray:
        image = self._prepare_rgb(rgb)
        candidate_rgb, candidate_gray = self._candidate_pixels(image)
        airlight = self._estimate_from_candidates(candidate_rgb, candidate_gray)
        return self._apply_color_bias(airlight, reference_color=airlight)

    def estimate_airlight(
        self,
        image: np.ndarray,
        sky_mask: np.ndarray,
        sample_id: str | None = None,
    ) -> np.ndarray:
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
        # Force odd so that the median matches an actual pixel
        count = count + ((count + 1) % 2)
        return max(1, min(count, num_pixels))

    def _prepare_rgb(self, rgb: np.ndarray) -> np.ndarray:
        image = np.asarray(rgb)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 4:
            image = image[..., :3]

        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("rgb image must have shape (H, W, 3)")

        image = image.astype(np.float32)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        return image

    @staticmethod
    def _prepare_color(value: np.ndarray) -> np.ndarray:
        color = np.asarray(value, dtype=np.float32)
        if color.ndim != 1 or color.shape[0] != 3:
            raise ValueError("cool_target must be a length-3 color")
        color = np.nan_to_num(color, nan=0.0, posinf=0.0, neginf=0.0)
        if float(color.max()) > 1.0:
            color = color / 255.0
        return np.clip(color, 0.0, 1.0)

    def _candidate_pixels(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dark_channel = self._dark_channel(rgb)

        num_pixels = dark_channel.size
        if num_pixels == 0:
            raise ValueError("rgb image is empty")

        num_select = self._brightest_pixels_count(num_pixels)
        flat_dark = dark_channel.ravel()

        if num_select >= num_pixels:
            candidate_idx = np.arange(num_pixels)
        else:
            candidate_idx = np.argpartition(flat_dark, -num_select)[-num_select:]

        flat_rgb = rgb.reshape(-1, 3)
        candidate_rgb = flat_rgb[candidate_idx]
        candidate_gray = flat_rgb[candidate_idx] @ _GRAY_WEIGHTS
        return candidate_rgb, candidate_gray

    def _estimate_from_candidates(
        self,
        candidate_rgb: np.ndarray,
        candidate_gray: np.ndarray,
    ) -> np.ndarray:
        median_intensity = float(np.median(candidate_gray))
        bright_mask = candidate_gray >= median_intensity
        if not np.any(bright_mask):
            bright_mask = np.ones_like(candidate_gray, dtype=bool)

        bright_rgb = candidate_rgb[bright_mask]
        bright_gray = candidate_gray[bright_mask]
        weights = bright_gray - float(bright_gray.min())
        weights = weights + np.finfo(np.float32).eps
        airlight = np.average(bright_rgb, axis=0, weights=weights)
        return airlight.astype(np.float32)

    def _estimate_sky_prior(
        self,
        rgb: np.ndarray,
        sky_mask: np.ndarray | None,
    ) -> np.ndarray | None:
        if sky_mask is None:
            return None

        mask = np.asarray(sky_mask, dtype=bool)
        if mask.shape != rgb.shape[:2]:
            raise ValueError("sky_mask must have shape (H, W)")
        if not np.any(mask):
            return None

        sky_pixels = rgb[mask]
        sky_gray = sky_pixels @ _GRAY_WEIGHTS
        threshold = float(np.quantile(sky_gray, _BRIGHT_SKY_QUANTILE))
        bright_sky = sky_pixels[sky_gray >= threshold]
        if bright_sky.size == 0:
            bright_sky = sky_pixels
        return np.mean(bright_sky, axis=0).astype(np.float32)

    def _merge_with_sky_prior(
        self,
        airlight: np.ndarray,
        sky_prior: np.ndarray,
    ) -> np.ndarray:
        sky_luminance = self._luminance(sky_prior)
        if sky_luminance <= np.finfo(np.float32).eps:
            return airlight.astype(np.float32)

        target_luminance = max(self._luminance(airlight), sky_luminance)
        return (sky_prior * (target_luminance / sky_luminance)).astype(np.float32)

    def _apply_color_bias(
        self,
        airlight: np.ndarray,
        reference_color: np.ndarray | None,
    ) -> np.ndarray:
        if self.white_bias == 0.0 and self.cool_bias == 0.0:
            return airlight.astype(np.float32)

        reference = airlight if reference_color is None else reference_color
        cool_target = self._correlated_cool_target(reference)
        base_weight = 1.0 - self.white_bias - self.cool_bias
        biased = (
            base_weight * airlight
            + self.white_bias * _WHITE_TARGET
            + self.cool_bias * cool_target
        )
        airlight_luminance = self._luminance(airlight)
        biased_luminance = self._luminance(biased)
        if (
            airlight_luminance > np.finfo(np.float32).eps
            and biased_luminance > np.finfo(np.float32).eps
        ):
            biased = biased * (airlight_luminance / biased_luminance)
        return biased.astype(np.float32)

    def _correlated_cool_target(self, reference_color: np.ndarray) -> np.ndarray:
        reference = self._prepare_color(reference_color)
        return 0.5 * reference + 0.5 * self.cool_target

    @staticmethod
    def _luminance(rgb: np.ndarray) -> float:
        return float(np.dot(rgb, _GRAY_WEIGHTS))

    def _dark_channel(self, rgb: np.ndarray) -> np.ndarray:
        min_channel = np.min(rgb, axis=2)
        if self.patch_size == 1:
            return min_channel

        pad = self.patch_size // 2
        padded = np.pad(min_channel, ((pad, pad), (pad, pad)), mode="edge")
        windows = sliding_window_view(padded, (self.patch_size, self.patch_size))
        return np.min(windows, axis=(-1, -2))
