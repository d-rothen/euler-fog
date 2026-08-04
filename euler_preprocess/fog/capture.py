from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from PIL import Image

from euler_preprocess.common.color import (
    linear_to_srgb,
    linear_to_srgb_torch,
    srgb_to_linear,
    srgb_to_linear_torch,
)
from euler_preprocess.common.noise import perlin_fbm, perlin_fbm_torch
from euler_preprocess.common.sampling import deep_merge, sample_value

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional
    torch = None


@dataclass(frozen=True)
class CaptureContext:
    """Metadata passed to camera/capture artifact stages."""

    sample_id: str | None = None
    rng: Any | None = None
    device: Any | None = None
    intrinsics: Any | None = None
    depth_m: Any | None = None
    k_map: Any | None = None
    fog_opacity: Any | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)


class CaptureArtifactStage:
    """Base class for post-render camera artifact stages.

    Stages operate on float RGB images in ``[0, 1]`` and leave physical fog
    auxiliary maps untouched. Torch inputs are supported by applying the same
    CPU implementation per image and converting back to the source device.
    """

    name = "capture_artifact"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def apply_np(self, image, context: CaptureContext):
        return image

    def apply_torch(self, image, context: CaptureContext):
        np_image = _torch_to_numpy_image(image)
        processed = self.apply_np(np_image, context)
        return _numpy_to_torch_like(processed, image)

    def apply_torch_batch(self, images, contexts: tuple[CaptureContext, ...]):
        if not contexts:
            return images
        processed = [
            self.apply_torch(images[index], context)
            for index, context in enumerate(contexts)
        ]
        return _stack_like(images, processed)


class ConfiguredCaptureStage(CaptureArtifactStage):
    """Capture stage with common ``enabled`` and ``probability`` controls."""

    DEFAULTS: dict[str, Any] = {}

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        cleaned = {
            key: value
            for key, value in dict(config or {}).items()
            if key not in {"type", "name"}
        }
        super().__init__(deep_merge(dict(self.DEFAULTS), cleaned))

    def apply_np(self, image, context: CaptureContext):
        rng = _rng(context)
        config = _conditioned_stage_config(self.config, rng)
        if not self._should_apply(rng, config):
            return image
        return self._apply_np(_as_float_rgb(image), context, rng, config)

    def apply_torch(self, image, context: CaptureContext):
        if torch is None or not torch.is_tensor(image):
            return self.apply_np(image, context)
        rng = _rng(context)
        config = _conditioned_stage_config(self.config, rng)
        if not self._should_apply(rng, config):
            return image
        return self._apply_torch(_as_float_rgb_torch(image), context, rng, config)

    def _apply_np(
        self,
        image: np.ndarray,
        context: CaptureContext,
        rng: np.random.Generator,
        config: Mapping[str, Any],
    ) -> np.ndarray:
        return image

    def _apply_torch(
        self,
        image,
        context: CaptureContext,
        rng: np.random.Generator,
        config: Mapping[str, Any],
    ):
        np_image = _torch_to_numpy_image(image)
        processed = self._apply_np(np_image, context, rng, config)
        return _numpy_to_torch_like(processed, image)

    def _should_apply(
        self,
        rng: np.random.Generator,
        config: Mapping[str, Any],
    ) -> bool:
        if not _bool_value(config.get("enabled", True)):
            return False
        probability = _sample_float(config, "probability", 1.0, rng)
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        return bool(rng.random() < probability)


class OpticsStage(ConfiguredCaptureStage):
    """Lens, windshield, and image-plane weather effects before raw sampling."""

    name = "optics"
    DEFAULTS = {
        "blur_sigma": {"dist": "uniform", "min": 0.15, "max": 0.75},
        "motion_blur": {
            "enabled": True,
            "probability": 0.25,
            "length_px": {"dist": "uniform", "min": 2.0, "max": 5.0},
            "angle_deg": {"dist": "uniform", "min": -8.0, "max": 8.0},
        },
        "vignetting_strength": {"dist": "uniform", "min": 0.04, "max": 0.22},
        "vignetting_radius": 1.15,
        "chromatic_aberration_px": {"dist": "uniform", "min": 0.0, "max": 0.7},
        "depth_chromatic_fringing": {
            "enabled": False,
            "strength_px": {"dist": "uniform", "min": 0.0, "max": 0.6},
            "depth_weight": 0.35,
            "fog_weight": 0.55,
            "dark_weight": 0.1,
            "gamma": 1.2,
            "max_alpha": 0.8,
            "blur_sigma": 1.2,
        },
        "lens_distortion": {"dist": "uniform", "min": -0.015, "max": 0.015},
        "bloom": {
            "enabled": True,
            "threshold": {"dist": "uniform", "min": 0.72, "max": 0.9},
            "strength": {"dist": "uniform", "min": 0.02, "max": 0.14},
            "sigma": {"dist": "uniform", "min": 2.0, "max": 6.0},
        },
        "veiling_glare_strength": {"dist": "uniform", "min": 0.0, "max": 0.04},
        "fog_coupled_glare": {
            "enabled": False,
            "base_strength": 0.0,
            "fog_strength": 0.0,
            "highlight_strength": 0.0,
            "airlight_strength": 0.0,
            "highlight_threshold": 0.72,
            "smooth_sigma": 12.0,
            "color": [0.92, 0.95, 1.0],
        },
        "windshield_haze": {
            "enabled": True,
            "probability": 0.35,
            "strength": {"dist": "uniform", "min": 0.02, "max": 0.12},
            "blur_sigma": {"dist": "uniform", "min": 4.0, "max": 14.0},
            "color": [0.82, 0.86, 0.88],
        },
        "droplets": {
            "enabled": False,
            "probability": 0.2,
            "count": {"dist": "uniform", "min": 4.0, "max": 18.0},
            "radius_fraction": {"dist": "uniform", "min": 0.01, "max": 0.04},
            "opacity": {"dist": "uniform", "min": 0.25, "max": 0.55},
            "refraction_px": {"dist": "uniform", "min": 1.5, "max": 5.0},
            "blur_sigma": {"dist": "uniform", "min": 1.5, "max": 4.0},
        },
    }

    def _apply_np(
        self,
        image: np.ndarray,
        context: CaptureContext,
        rng: np.random.Generator,
        config: Mapping[str, Any],
    ) -> np.ndarray:
        img = image
        intrinsics = _context_intrinsics(context)

        distortion = _sample_float(config, "lens_distortion", 0.0, rng)
        if abs(distortion) > 1e-5:
            k2 = _sample_float(config, "lens_distortion_k2", 0.0, rng)
            img = _lens_distort_np(img, distortion, k2, intrinsics)

        aberration_px = _sample_float(
            config,
            "chromatic_aberration_px",
            0.0,
            rng,
        )
        if aberration_px > 1e-4:
            img = _chromatic_aberration_np(img, aberration_px, intrinsics)

        fringing_cfg = _config_block(config, "depth_chromatic_fringing")
        if _block_enabled(fringing_cfg, rng):
            img = _apply_depth_chromatic_fringing_np(
                img,
                fringing_cfg,
                context,
                intrinsics,
                rng,
            )

        blur_sigma = _sample_float(config, "blur_sigma", 0.0, rng)
        if blur_sigma > 1e-4:
            img = _gaussian_blur_np(img, blur_sigma)

        motion_cfg = _config_block(config, "motion_blur")
        if _block_enabled(motion_cfg, rng):
            length = _sample_float(motion_cfg, "length_px", 0.0, rng)
            angle = _sample_float(motion_cfg, "angle_deg", 0.0, rng)
            if length >= 2.0:
                img = _motion_blur_np(img, length, angle)

        bloom_cfg = _config_block(config, "bloom")
        if _bool_value(bloom_cfg.get("enabled", True)):
            img = _apply_bloom_np(img, bloom_cfg, rng)

        glare = _sample_float(config, "veiling_glare_strength", 0.0, rng)
        if glare > 1e-5:
            veil = _low_frequency_field(img.shape[0], img.shape[1], rng)
            img = img * (1.0 - glare) + glare * veil[..., None]

        fog_glare_cfg = _config_block(config, "fog_coupled_glare")
        if _block_enabled(fog_glare_cfg, rng):
            img = _apply_fog_coupled_glare_np(img, fog_glare_cfg, context, rng)

        vignette = _sample_float(config, "vignetting_strength", 0.0, rng)
        if vignette > 1e-5:
            radius = _sample_float(config, "vignetting_radius", 1.15, rng)
            mask = _vignette_mask(
                img.shape[0],
                img.shape[1],
                vignette,
                radius,
                intrinsics,
            )
            img = img * mask[..., None]

        windshield_cfg = _config_block(config, "windshield_haze")
        if _block_enabled(windshield_cfg, rng):
            img = _apply_windshield_haze_np(img, windshield_cfg, rng)

        droplets_cfg = _config_block(config, "droplets")
        if _block_enabled(droplets_cfg, rng):
            img = _apply_droplets_np(img, droplets_cfg, rng)

        return _clip01(img)

    def _apply_torch(
        self,
        image,
        context: CaptureContext,
        rng: np.random.Generator,
        config: Mapping[str, Any],
    ):
        img = image
        intrinsics = _context_intrinsics_torch(context, img.device, img.dtype)

        distortion = _sample_float(config, "lens_distortion", 0.0, rng)
        if abs(distortion) > 1e-5:
            k2 = _sample_float(config, "lens_distortion_k2", 0.0, rng)
            img = _lens_distort_torch(img, distortion, k2, intrinsics)

        aberration_px = _sample_float(
            config,
            "chromatic_aberration_px",
            0.0,
            rng,
        )
        if aberration_px > 1e-4:
            img = _chromatic_aberration_torch(img, aberration_px, intrinsics)

        fringing_cfg = _config_block(config, "depth_chromatic_fringing")
        if _block_enabled(fringing_cfg, rng):
            img = _apply_depth_chromatic_fringing_torch(
                img,
                fringing_cfg,
                context,
                intrinsics,
                rng,
            )

        blur_sigma = _sample_float(config, "blur_sigma", 0.0, rng)
        if blur_sigma > 1e-4:
            img = _gaussian_blur_torch(img, blur_sigma)

        motion_cfg = _config_block(config, "motion_blur")
        if _block_enabled(motion_cfg, rng):
            length = _sample_float(motion_cfg, "length_px", 0.0, rng)
            angle = _sample_float(motion_cfg, "angle_deg", 0.0, rng)
            if length >= 2.0:
                img = _motion_blur_torch(img, length, angle)

        bloom_cfg = _config_block(config, "bloom")
        if _bool_value(bloom_cfg.get("enabled", True)):
            img = _apply_bloom_torch(img, bloom_cfg, rng)

        glare = _sample_float(config, "veiling_glare_strength", 0.0, rng)
        if glare > 1e-5:
            veil = _low_frequency_field_torch(
                int(img.shape[0]),
                int(img.shape[1]),
                rng,
                img.device,
                img.dtype,
            )
            img = img * (1.0 - float(glare)) + float(glare) * veil[..., None]

        fog_glare_cfg = _config_block(config, "fog_coupled_glare")
        if _block_enabled(fog_glare_cfg, rng):
            img = _apply_fog_coupled_glare_torch(img, fog_glare_cfg, context, rng)

        vignette = _sample_float(config, "vignetting_strength", 0.0, rng)
        if vignette > 1e-5:
            radius = _sample_float(config, "vignetting_radius", 1.15, rng)
            mask = _vignette_mask_torch(
                int(img.shape[0]),
                int(img.shape[1]),
                vignette,
                radius,
                intrinsics,
                img.device,
                img.dtype,
            )
            img = img * mask[..., None]

        windshield_cfg = _config_block(config, "windshield_haze")
        if _block_enabled(windshield_cfg, rng):
            img = _apply_windshield_haze_torch(img, windshield_cfg, rng)

        droplets_cfg = _config_block(config, "droplets")
        if _block_enabled(droplets_cfg, rng):
            img_np = _apply_droplets_np(_torch_to_numpy_image(img), droplets_cfg, rng)
            img = _numpy_to_torch_like(img_np, img)

        return _clip01_torch(img)


class ExposureStage(ConfiguredCaptureStage):
    """Small standalone exposure/white-balance stage for simple configs."""

    name = "exposure"
    DEFAULTS = {
        "gain": 1.0,
        "exposure_gain": None,
        "white_balance": [1.0, 1.0, 1.0],
        "white_balance_jitter": 0.0,
        "clip": True,
    }

    def _apply_np(
        self,
        image: np.ndarray,
        context: CaptureContext,
        rng: np.random.Generator,
        config: Mapping[str, Any],
    ) -> np.ndarray:
        gain_key = (
            "exposure_gain" if config.get("exposure_gain") is not None else "gain"
        )
        gain = _sample_float(config, gain_key, 1.0, rng)
        wb = _sample_triplet(config.get("white_balance", [1.0, 1.0, 1.0]), rng)
        jitter = _sample_float(config, "white_balance_jitter", 0.0, rng)
        if jitter > 0.0:
            wb = wb * rng.lognormal(mean=0.0, sigma=jitter, size=3).astype(np.float32)
        out = image * gain * wb.reshape(1, 1, 3)
        if _bool_value(config.get("clip", True)):
            out = _clip01(out)
        return out.astype(np.float32, copy=False)

    def _apply_torch(
        self,
        image,
        context: CaptureContext,
        rng: np.random.Generator,
        config: Mapping[str, Any],
    ):
        gain_key = (
            "exposure_gain" if config.get("exposure_gain") is not None else "gain"
        )
        gain = _sample_float(config, gain_key, 1.0, rng)
        wb = _sample_triplet(config.get("white_balance", [1.0, 1.0, 1.0]), rng)
        jitter = _sample_float(config, "white_balance_jitter", 0.0, rng)
        if jitter > 0.0:
            wb = wb * rng.lognormal(mean=0.0, sigma=jitter, size=3).astype(np.float32)
        wb_t = torch.as_tensor(wb, device=image.device, dtype=image.dtype)
        out = image * float(gain) * wb_t.view(1, 1, 3)
        if _bool_value(config.get("clip", True)):
            out = _clip01_torch(out)
        return out


class SensorStage(ConfiguredCaptureStage):
    """Raw-like camera sampling, Bayer mosaic, and heteroscedastic noise."""

    name = "sensor"
    DEFAULTS = {
        "input_space": "linear",
        "exposure_gain": {"dist": "uniform", "min": 0.85, "max": 1.25},
        "auto_exposure": {
            "enabled": False,
            "metering": "center_weighted",
            "target_luminance": 0.18,
            "center_weight": 0.55,
            "center_sigma": 0.35,
            "meter_percentile": 50.0,
            "highlight_percentile": 98.5,
            "highlight_target": 0.92,
            "highlight_protection": 0.65,
            "min_gain": 0.35,
            "max_gain": 2.6,
            "exposure_compensation_ev": 0.0,
            "manual_gain_weight": 1.0,
            "resolve_iso": False,
            "min_iso": None,
            "max_iso": 1600.0,
            "iso_activation_gain": 1.2,
            "iso_gain_power": 0.85,
            "dark_fraction_threshold": 0.12,
            "dark_iso_boost": 0.0,
            "fog_iso_boost": 0.0,
        },
        "white_balance": [1.0, 1.0, 1.0],
        "white_balance_jitter": 0.03,
        "channel_gain_sigma": 0.01,
        "camera_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "clip": 1.0,
        "bayer_pattern": {"dist": "choice", "values": ["RGGB", "BGGR", "GRBG", "GBRG"]},
        "iso": {"dist": "choice", "values": [100.0, 200.0, 400.0, 800.0]},
        "base_iso": 100.0,
        "full_well_electrons": {"dist": "uniform", "min": 8000.0, "max": 24000.0},
        "read_noise_electrons": {"dist": "uniform", "min": 1.5, "max": 7.0},
        "read_noise_sigma": None,
        "fixed_pattern_sigma": {"dist": "uniform", "min": 0.0, "max": 0.0015},
        "row_noise_sigma": {"dist": "uniform", "min": 0.0, "max": 0.0012},
        "column_noise_sigma": {"dist": "uniform", "min": 0.0, "max": 0.0007},
        "row_banding_correlation_px": {
            "dist": "uniform",
            "min": 24.0,
            "max": 96.0,
        },
        "column_banding_correlation_px": {
            "dist": "uniform",
            "min": 24.0,
            "max": 96.0,
        },
        "banding_modulation": 0.35,
        "noise_modulation": {
            "enabled": False,
            "dark_gain": 0.0,
            "depth_gain": 0.0,
            "fog_gain": 0.0,
            "gamma": 1.0,
            "max_gain": 3.0,
            "smooth_sigma": 0.0,
            "black_noise_floor": 1.0,
            "black_suppression_luminance": 0.0,
            "black_suppression_softness": 0.05,
        },
        "shadow_recovery_noise": {
            "enabled": False,
            "luminance_threshold": 0.18,
            "luminance_softness": 0.08,
            "gamma": 1.4,
            "strength": 1.0,
            "luma_sigma": 0.0,
            "chroma_sigma": 0.0,
            "chroma_mode": "balanced",
            "red_chroma_gain": 1.0,
            "blue_chroma_gain": 1.0,
            "chroma_axis_correlation": 0.0,
            "chroma_spatial_sigma": 0.0,
            "chroma_fine_fraction": 1.0,
            "chroma_luminance_preservation": 1.0,
            "blotch_sigma": 0.0,
            "fog_weight": 0.0,
            "depth_weight": 0.0,
            "smooth_sigma": 1.0,
            "max_weight": 1.0,
            "black_noise_floor": 1.0,
            "black_suppression_luminance": 0.0,
            "black_suppression_softness": 0.05,
        },
        "black_level": [0.003, 0.003, 0.003],
        "black_level_jitter": {"dist": "uniform", "min": 0.0, "max": 0.002},
        "white_level": [1.0, 1.0, 1.0],
        "white_level_jitter": {"dist": "uniform", "min": 0.0, "max": 0.004},
        "adc_bit_depth": 12,
        "post_demosaic_bit_depth": 12,
        "hot_pixel_probability": 0.00002,
        "dead_pixel_probability": 0.00001,
        "sensor_identity": {
            "enabled": False,
            "sensor_id": "default",
            "seed": 0,
            "prnu_sigma": 0.0,
            "dsnu_sigma": 0.0,
            "persistent_hot_pixel_probability": 0.0,
            "persistent_dead_pixel_probability": 0.0,
            "persistent_row_sigma": 0.0,
            "persistent_column_sigma": 0.0,
        },
        "demosaic": True,
    }

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config)
        self._sensor_identity_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def _apply_np(
        self,
        image: np.ndarray,
        context: CaptureContext,
        rng: np.random.Generator,
        config: Mapping[str, Any],
    ) -> np.ndarray:
        img = image
        if str(config.get("input_space", "linear")).lower() == "srgb":
            img = _srgb_to_linear(img)

        matrix = _sample_matrix(config.get("camera_matrix"), rng)
        img = _apply_color_matrix(img, matrix)
        pre_exposure_luminance = _linear_luminance_np(img)

        exposure = _sample_float(config, "exposure_gain", 1.0, rng)
        exposure, config = _resolve_auto_exposure_np(
            img,
            context,
            config,
            rng,
            manual_exposure_gain=exposure,
        )
        config = _apply_sensor_noise_adjustment(config, rng)
        wb = _sample_triplet(config.get("white_balance", [1.0, 1.0, 1.0]), rng)
        wb_jitter = _sample_float(config, "white_balance_jitter", 0.0, rng)
        if wb_jitter > 0.0:
            wb = wb * rng.lognormal(mean=0.0, sigma=wb_jitter, size=3).astype(
                np.float32
            )
        gain_sigma = _sample_float(config, "channel_gain_sigma", 0.0, rng)
        if gain_sigma > 0.0:
            wb = wb * rng.lognormal(mean=0.0, sigma=gain_sigma, size=3).astype(
                np.float32
            )
        img = img * exposure * wb.reshape(1, 1, 3)

        white_clip = _sample_float(config, "clip", 1.0, rng)
        img = np.clip(img, 0.0, max(white_clip, 1e-6)) / max(white_clip, 1e-6)

        pattern = str(_sample_any(config.get("bayer_pattern", "RGGB"), rng)).upper()
        raw_signal = _clip01(_bayer_mosaic_np(img, pattern))
        identity_maps = self._sensor_identity_maps_np(raw_signal.shape, pattern, config)
        if identity_maps is not None:
            raw_signal = _clip01(raw_signal * identity_maps["prnu"])

        black_levels = _sample_sensor_levels(config, "black_level", rng)
        black_levels += rng.normal(
            0.0,
            _sample_float(config, "black_level_jitter", 0.0, rng),
            size=3,
        ).astype(np.float32)
        black_levels = np.clip(black_levels, 0.0, 0.95)

        white_levels = _sample_sensor_levels(config, "white_level", rng)
        white_levels += rng.normal(
            0.0,
            _sample_float(config, "white_level_jitter", 0.0, rng),
            size=3,
        ).astype(np.float32)
        white_levels = np.clip(white_levels, 0.05, 1.0)

        black_map = _mosaic_channel_values(raw_signal.shape, pattern, black_levels)
        white_map = _mosaic_channel_values(raw_signal.shape, pattern, white_levels)
        white_map = np.maximum(white_map, black_map + 1e-4)
        raw_range = white_map - black_map

        electron_capacity = _resolve_electron_capacity(
            raw_signal.shape,
            pattern,
            config,
            rng,
        )
        noisy_signal = raw_signal
        if np.any(electron_capacity > 0.0):
            noisy_signal = rng.poisson(
                np.clip(raw_signal, 0.0, None) * electron_capacity
            ).astype(np.float32) / np.maximum(electron_capacity, 1e-6)
        noise_modulation = _sensor_noise_modulation(
            raw_signal,
            context,
            config,
            rng,
        )

        read_sigma_cfg = config.get("read_noise_sigma")
        if read_sigma_cfg is not None:
            read_sigma = _sample_float(config, "read_noise_sigma", 0.0, rng)
        else:
            read_electrons = _sample_float(
                config,
                "read_noise_electrons",
                0.0,
                rng,
            )
            read_sigma = np.where(
                electron_capacity > 0.0,
                read_electrons / np.maximum(electron_capacity, 1e-6),
                0.0,
            )
        if np.any(np.asarray(read_sigma) > 0.0):
            noisy_signal = (
                noisy_signal
                + rng.normal(
                    0.0,
                    1.0,
                    raw_signal.shape,
                ).astype(np.float32)
                * read_sigma
                * noise_modulation
            )

        raw = black_map + noisy_signal * raw_range
        if identity_maps is not None:
            raw = raw + identity_maps["dsnu"]
            raw = raw + identity_maps["row_bias"][:, None]
            raw = raw + identity_maps["column_bias"][None, :]

        fixed_sigma = _sample_float(config, "fixed_pattern_sigma", 0.0, rng)
        if fixed_sigma > 0.0:
            raw = raw + (
                rng.normal(0.0, fixed_sigma, raw.shape).astype(np.float32)
                * noise_modulation
            )

        banding_modulation = float(
            np.clip(_sample_float(config, "banding_modulation", 0.35, rng), 0, 1)
        )
        banding_gain = 1.0 + banding_modulation * (noise_modulation - 1.0)
        row_sigma = _sample_float(config, "row_noise_sigma", 0.0, rng)
        if row_sigma > 0.0:
            row_corr = _sample_float(
                config,
                "row_banding_correlation_px",
                48.0,
                rng,
            )
            row_bias = _smooth_random_bias(raw.shape[0], row_sigma, row_corr, rng)
            raw = raw + row_bias[:, None] * banding_gain

        column_sigma = _sample_float(config, "column_noise_sigma", 0.0, rng)
        if column_sigma > 0.0:
            column_corr = _sample_float(
                config,
                "column_banding_correlation_px",
                48.0,
                rng,
            )
            column_bias = _smooth_random_bias(
                raw.shape[1],
                column_sigma,
                column_corr,
                rng,
            )
            raw = raw + column_bias[None, :] * banding_gain

        if identity_maps is not None:
            raw = _apply_persistent_bad_pixels_np(
                raw,
                identity_maps,
                hot_value=white_map,
                dead_value=black_map,
            )
        raw = _apply_bad_pixels_np(
            raw,
            config,
            rng,
            hot_value=white_map,
            dead_value=black_map,
        )
        raw = np.clip(raw, black_map, white_map)

        bit_depth_key = (
            "raw_bit_depth"
            if config.get("raw_bit_depth") is not None
            else "adc_bit_depth"
        )
        adc_bit_depth = int(round(_sample_float(config, bit_depth_key, 12.0, rng)))
        if adc_bit_depth > 0:
            raw = _quantize_np(raw, adc_bit_depth)

        raw = (raw - black_map) / raw_range
        raw = _clip01(raw)

        if _bool_value(config.get("demosaic", True)):
            demosaiced = _clip01(_demosaic_bilinear_np(raw, pattern))
            post_bits = int(
                round(
                    _sample_float(
                        config,
                        "post_demosaic_bit_depth",
                        0.0,
                        rng,
                    )
                )
            )
            if post_bits > 0:
                demosaiced = _quantize_np(demosaiced, post_bits)
            return _apply_shadow_recovery_noise_np(
                demosaiced,
                pre_exposure_luminance,
                context,
                config,
                rng,
            )

        raw_rgb = np.repeat(raw[..., None], 3, axis=-1).astype(np.float32, copy=False)
        return _apply_shadow_recovery_noise_np(
            raw_rgb,
            pre_exposure_luminance,
            context,
            config,
            rng,
        )

    def _apply_torch(
        self,
        image,
        context: CaptureContext,
        rng: np.random.Generator,
        config: Mapping[str, Any],
    ):
        img = image
        if str(config.get("input_space", "linear")).lower() == "srgb":
            img = _srgb_to_linear_torch(img)

        matrix = _sample_matrix(config.get("camera_matrix"), rng)
        img = _apply_color_matrix_torch(img, matrix)
        pre_exposure_luminance = _linear_luminance_torch(img)

        exposure = _sample_float(config, "exposure_gain", 1.0, rng)
        exposure, config = _resolve_auto_exposure_torch(
            img,
            context,
            config,
            rng,
            manual_exposure_gain=exposure,
        )
        config = _apply_sensor_noise_adjustment(config, rng)
        wb = _sample_triplet(config.get("white_balance", [1.0, 1.0, 1.0]), rng)
        wb_jitter = _sample_float(config, "white_balance_jitter", 0.0, rng)
        if wb_jitter > 0.0:
            wb = wb * rng.lognormal(mean=0.0, sigma=wb_jitter, size=3).astype(
                np.float32
            )
        gain_sigma = _sample_float(config, "channel_gain_sigma", 0.0, rng)
        if gain_sigma > 0.0:
            wb = wb * rng.lognormal(mean=0.0, sigma=gain_sigma, size=3).astype(
                np.float32
            )
        wb_t = torch.as_tensor(wb, device=img.device, dtype=img.dtype)
        img = img * float(exposure) * wb_t.view(1, 1, 3)

        white_clip = _sample_float(config, "clip", 1.0, rng)
        white_clip = max(float(white_clip), 1e-6)
        img = torch.clamp(img, 0.0, white_clip) / white_clip

        pattern = str(_sample_any(config.get("bayer_pattern", "RGGB"), rng)).upper()
        raw_signal = _clip01_torch(_bayer_mosaic_torch(img, pattern))
        identity_maps = self._sensor_identity_maps_torch(
            tuple(raw_signal.shape),
            pattern,
            config,
            raw_signal.device,
            raw_signal.dtype,
        )
        if identity_maps is not None:
            raw_signal = _clip01_torch(raw_signal * identity_maps["prnu"])

        black_levels = _sample_sensor_levels(config, "black_level", rng)
        black_levels += rng.normal(
            0.0,
            _sample_float(config, "black_level_jitter", 0.0, rng),
            size=3,
        ).astype(np.float32)
        black_levels = np.clip(black_levels, 0.0, 0.95)

        white_levels = _sample_sensor_levels(config, "white_level", rng)
        white_levels += rng.normal(
            0.0,
            _sample_float(config, "white_level_jitter", 0.0, rng),
            size=3,
        ).astype(np.float32)
        white_levels = np.clip(white_levels, 0.05, 1.0)

        black_map = _mosaic_channel_values_torch(
            tuple(raw_signal.shape),
            pattern,
            black_levels,
            raw_signal.device,
            raw_signal.dtype,
        )
        white_map = _mosaic_channel_values_torch(
            tuple(raw_signal.shape),
            pattern,
            white_levels,
            raw_signal.device,
            raw_signal.dtype,
        )
        white_map = torch.maximum(white_map, black_map + 1e-4)
        raw_range = white_map - black_map

        electron_capacity = _resolve_electron_capacity_torch(
            tuple(raw_signal.shape),
            pattern,
            config,
            rng,
            raw_signal.device,
            raw_signal.dtype,
        )
        noisy_signal = raw_signal
        if bool(torch.any(electron_capacity > 0.0).detach().cpu()):
            poisson_input = torch.clamp(raw_signal, min=0.0) * electron_capacity
            noisy_signal = _poisson_torch(poisson_input, rng) / torch.clamp(
                electron_capacity,
                min=1e-6,
            )

        noise_modulation = _sensor_noise_modulation_torch(
            raw_signal,
            context,
            config,
            rng,
        )

        read_sigma_cfg = config.get("read_noise_sigma")
        if read_sigma_cfg is not None:
            read_sigma = _sample_float(config, "read_noise_sigma", 0.0, rng)
        else:
            read_electrons = _sample_float(
                config,
                "read_noise_electrons",
                0.0,
                rng,
            )
            read_sigma = torch.where(
                electron_capacity > 0.0,
                float(read_electrons) / torch.clamp(electron_capacity, min=1e-6),
                torch.zeros_like(electron_capacity),
            )
        if _positive_torch_or_float(read_sigma):
            noisy_signal = noisy_signal + (
                _randn_torch(raw_signal.shape, rng, raw_signal.device, raw_signal.dtype)
                * read_sigma
                * noise_modulation
            )

        raw = black_map + noisy_signal * raw_range
        if identity_maps is not None:
            raw = raw + identity_maps["dsnu"]
            raw = raw + identity_maps["row_bias"][:, None]
            raw = raw + identity_maps["column_bias"][None, :]

        fixed_sigma = _sample_float(config, "fixed_pattern_sigma", 0.0, rng)
        if fixed_sigma > 0.0:
            raw = raw + (
                _randn_torch(raw.shape, rng, raw.device, raw.dtype)
                * float(fixed_sigma)
                * noise_modulation
            )

        banding_modulation = float(
            np.clip(_sample_float(config, "banding_modulation", 0.35, rng), 0, 1)
        )
        banding_gain = 1.0 + banding_modulation * (noise_modulation - 1.0)
        row_sigma = _sample_float(config, "row_noise_sigma", 0.0, rng)
        if row_sigma > 0.0:
            row_corr = _sample_float(
                config,
                "row_banding_correlation_px",
                48.0,
                rng,
            )
            row_bias = _smooth_random_bias_torch(
                int(raw.shape[0]),
                row_sigma,
                row_corr,
                rng,
                raw.device,
                raw.dtype,
            )
            raw = raw + row_bias[:, None] * banding_gain

        column_sigma = _sample_float(config, "column_noise_sigma", 0.0, rng)
        if column_sigma > 0.0:
            column_corr = _sample_float(
                config,
                "column_banding_correlation_px",
                48.0,
                rng,
            )
            column_bias = _smooth_random_bias_torch(
                int(raw.shape[1]),
                column_sigma,
                column_corr,
                rng,
                raw.device,
                raw.dtype,
            )
            raw = raw + column_bias[None, :] * banding_gain

        if identity_maps is not None:
            raw = _apply_persistent_bad_pixels_torch(
                raw,
                identity_maps,
                hot_value=white_map,
                dead_value=black_map,
            )
        raw = _apply_bad_pixels_torch(
            raw,
            config,
            rng,
            hot_value=white_map,
            dead_value=black_map,
        )
        raw = torch.maximum(torch.minimum(raw, white_map), black_map)

        bit_depth_key = (
            "raw_bit_depth"
            if config.get("raw_bit_depth") is not None
            else "adc_bit_depth"
        )
        adc_bit_depth = int(round(_sample_float(config, bit_depth_key, 12.0, rng)))
        if adc_bit_depth > 0:
            raw = _quantize_torch(raw, adc_bit_depth)

        raw = (raw - black_map) / torch.clamp(raw_range, min=1e-6)
        raw = _clip01_torch(raw)

        if _bool_value(config.get("demosaic", True)):
            demosaiced = _clip01_torch(_demosaic_bilinear_torch(raw, pattern))
            post_bits = int(
                round(
                    _sample_float(
                        config,
                        "post_demosaic_bit_depth",
                        0.0,
                        rng,
                    )
                )
            )
            if post_bits > 0:
                demosaiced = _quantize_torch(demosaiced, post_bits)
            return _apply_shadow_recovery_noise_torch(
                demosaiced,
                pre_exposure_luminance,
                context,
                config,
                rng,
            )

        raw_rgb = raw[..., None].repeat(1, 1, 3)
        return _apply_shadow_recovery_noise_torch(
            raw_rgb,
            pre_exposure_luminance,
            context,
            config,
            rng,
        )

    def _sensor_identity_maps_np(
        self,
        shape: tuple[int, int],
        pattern: str,
        config: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        cfg = _config_block(config, "sensor_identity")
        if not _bool_value(cfg.get("enabled", False)):
            return None
        key, resolved_cfg = _sensor_identity_cache_key(cfg, shape, pattern)
        if key not in self._sensor_identity_cache:
            self._sensor_identity_cache[key] = _build_sensor_identity_maps_np(
                shape,
                resolved_cfg,
            )
        return self._sensor_identity_cache[key]

    def _sensor_identity_maps_torch(
        self,
        shape: tuple[int, int],
        pattern: str,
        config: Mapping[str, Any],
        device,
        dtype,
    ) -> dict[str, Any] | None:
        maps = self._sensor_identity_maps_np(shape, pattern, config)
        if maps is None:
            return None
        return {
            "prnu": torch.as_tensor(maps["prnu"], device=device, dtype=dtype),
            "dsnu": torch.as_tensor(maps["dsnu"], device=device, dtype=dtype),
            "row_bias": torch.as_tensor(maps["row_bias"], device=device, dtype=dtype),
            "column_bias": torch.as_tensor(
                maps["column_bias"],
                device=device,
                dtype=dtype,
            ),
            "hot_mask": torch.as_tensor(maps["hot_mask"], device=device),
            "dead_mask": torch.as_tensor(maps["dead_mask"], device=device),
        }


class ISPStage(ConfiguredCaptureStage):
    """Demosaiced camera-space RGB to display RGB with ISP artifacts."""

    name = "isp"
    DEFAULTS = {
        "denoise_sigma": {"dist": "uniform", "min": 0.0, "max": 0.45},
        "color_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "tone_map": "reinhard",
        "tone_map_strength": {"dist": "uniform", "min": 0.05, "max": 0.25},
        "tone_map_lut": None,
        "tone_map_lut_domain": "linear",
        "gamma": "srgb",
        "local_contrast_strength": {"dist": "uniform", "min": 0.0, "max": 0.12},
        "local_contrast_sigma": {"dist": "uniform", "min": 8.0, "max": 18.0},
        "sharpen_amount": {"dist": "uniform", "min": 0.0, "max": 0.35},
        "sharpen_sigma": 0.8,
        "saturation": {"dist": "uniform", "min": 0.75, "max": 1.05},
    }

    def _apply_np(
        self,
        image: np.ndarray,
        context: CaptureContext,
        rng: np.random.Generator,
        config: Mapping[str, Any],
    ) -> np.ndarray:
        img = image

        denoise_sigma = _sample_float(config, "denoise_sigma", 0.0, rng)
        if denoise_sigma > 1e-4:
            img = _gaussian_blur_np(img, denoise_sigma)

        matrix = _sample_matrix(config.get("color_matrix"), rng)
        img = _apply_color_matrix(img, matrix)

        tone_map = str(config.get("tone_map", "reinhard")).lower()
        strength = _sample_float(config, "tone_map_strength", 1.0, rng)
        if tone_map not in {"none", "false", "off"}:
            img = _apply_tone_map_np(img, tone_map, strength, config)

        gamma = config.get("gamma", "srgb")
        img = _apply_gamma_np(img, gamma, rng)

        local_strength = _sample_float(
            config,
            "local_contrast_strength",
            0.0,
            rng,
        )
        if local_strength > 1e-5:
            sigma = _sample_float(config, "local_contrast_sigma", 12.0, rng)
            base = _gaussian_blur_np(img, sigma)
            img = img + local_strength * (img - base)

        sharpen = _sample_float(config, "sharpen_amount", 0.0, rng)
        if sharpen > 1e-5:
            sigma = _sample_float(config, "sharpen_sigma", 0.8, rng)
            blurred = _gaussian_blur_np(img, sigma)
            img = img + sharpen * (img - blurred)

        saturation = _sample_float(config, "saturation", 1.0, rng)
        if abs(saturation - 1.0) > 1e-5:
            luma_weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            luma = np.sum(img * luma_weights, axis=-1, keepdims=True)
            img = luma + saturation * (img - luma)

        return _clip01(img)

    def _apply_torch(
        self,
        image,
        context: CaptureContext,
        rng: np.random.Generator,
        config: Mapping[str, Any],
    ):
        img = image

        denoise_sigma = _sample_float(config, "denoise_sigma", 0.0, rng)
        if denoise_sigma > 1e-4:
            img = _gaussian_blur_torch(img, denoise_sigma)

        matrix = _sample_matrix(config.get("color_matrix"), rng)
        img = _apply_color_matrix_torch(img, matrix)

        tone_map = str(config.get("tone_map", "reinhard")).lower()
        strength = _sample_float(config, "tone_map_strength", 1.0, rng)
        if tone_map not in {"none", "false", "off"}:
            img = _apply_tone_map_torch(img, tone_map, strength, config)

        gamma = config.get("gamma", "srgb")
        img = _apply_gamma_torch(img, gamma, rng)

        local_strength = _sample_float(
            config,
            "local_contrast_strength",
            0.0,
            rng,
        )
        if local_strength > 1e-5:
            sigma = _sample_float(config, "local_contrast_sigma", 12.0, rng)
            base = _gaussian_blur_torch(img, sigma)
            img = img + float(local_strength) * (img - base)

        sharpen = _sample_float(config, "sharpen_amount", 0.0, rng)
        if sharpen > 1e-5:
            sigma = _sample_float(config, "sharpen_sigma", 0.8, rng)
            blurred = _gaussian_blur_torch(img, sigma)
            img = img + float(sharpen) * (img - blurred)

        saturation = _sample_float(config, "saturation", 1.0, rng)
        if abs(saturation - 1.0) > 1e-5:
            weights = torch.tensor(
                [0.2126, 0.7152, 0.0722],
                device=img.device,
                dtype=img.dtype,
            )
            luma = torch.sum(img * weights.view(1, 1, 3), dim=-1, keepdim=True)
            img = luma + float(saturation) * (img - luma)

        return _clip01_torch(img)


class TransportStage(ConfiguredCaptureStage):
    """Final resize, quantization, and JPEG-like transport artifacts."""

    name = "transport"
    DEFAULTS = {
        "crop": None,
        "resize": None,
        "resize_scale": None,
        "resample": "bilinear",
        "bit_depth": 8,
        "jpeg": {
            "enabled": True,
            "quality": {"dist": "uniform", "min": 55.0, "max": 95.0},
            "subsampling": 2,
        },
        "jpeg_quality": None,
    }

    def _apply_np(
        self,
        image: np.ndarray,
        context: CaptureContext,
        rng: np.random.Generator,
        config: Mapping[str, Any],
    ) -> np.ndarray:
        img = image

        crop = config.get("crop")
        if crop is not None:
            img = _crop_np(img, crop, rng)

        target_size = _resolve_resize(config, img.shape, rng)
        if target_size is not None:
            img = _resize_np(img, target_size, str(config.get("resample", "bilinear")))

        bit_depth = int(round(_sample_float(config, "bit_depth", 8.0, rng)))
        if bit_depth > 0:
            img = _quantize_np(img, bit_depth)

        jpeg_cfg = _config_block(config, "jpeg")
        jpeg_quality = config.get("jpeg_quality")
        if jpeg_quality is not None:
            jpeg_cfg = dict(jpeg_cfg)
            jpeg_cfg["enabled"] = True
            jpeg_cfg["quality"] = jpeg_quality
        if _bool_value(jpeg_cfg.get("enabled", True)):
            quality = int(round(_sample_float(jpeg_cfg, "quality", 90.0, rng)))
            quality = int(np.clip(quality, 1, 100))
            subsampling = int(round(_sample_float(jpeg_cfg, "subsampling", 2.0, rng)))
            img = _jpeg_roundtrip_np(img, quality=quality, subsampling=subsampling)

        return _clip01(img)

    def _apply_torch(
        self,
        image,
        context: CaptureContext,
        rng: np.random.Generator,
        config: Mapping[str, Any],
    ):
        img = image

        crop = config.get("crop")
        if crop is not None:
            img = _crop_torch(img, crop, rng)

        target_size = _resolve_resize(config, tuple(img.shape), rng)
        if target_size is not None:
            img = _resize_torch(
                img,
                target_size,
                str(config.get("resample", "bilinear")),
            )

        bit_depth = int(round(_sample_float(config, "bit_depth", 8.0, rng)))
        if bit_depth > 0:
            img = _quantize_torch(img, bit_depth)

        jpeg_cfg = _config_block(config, "jpeg")
        jpeg_quality = config.get("jpeg_quality")
        if jpeg_quality is not None:
            jpeg_cfg = dict(jpeg_cfg)
            jpeg_cfg["enabled"] = True
            jpeg_cfg["quality"] = jpeg_quality
        if _bool_value(jpeg_cfg.get("enabled", True)):
            quality = int(round(_sample_float(jpeg_cfg, "quality", 90.0, rng)))
            quality = int(np.clip(quality, 1, 100))
            subsampling = int(round(_sample_float(jpeg_cfg, "subsampling", 2.0, rng)))
            img_np = _jpeg_roundtrip_np(
                _torch_to_numpy_image(img),
                quality=quality,
                subsampling=subsampling,
            )
            img = _numpy_to_torch_like(img_np, img)

        return _clip01_torch(img)


class CaptureArtifactPipeline:
    """Ordered post-fog camera/capture artifact pipeline.

    With no configured stages this is intentionally zero-copy: images are
    returned unchanged. Enable the camera stack explicitly with either
    ``"capture": true``, ``{"capture": {"preset": "camera"}}``, or a custom
    ``capture.stages`` list.
    """

    def __init__(self, stages: tuple[CaptureArtifactStage, ...] = ()) -> None:
        self.stages = stages

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CaptureArtifactPipeline":
        raw = config.get("capture_artifacts", config.get("capture"))
        if raw is None or raw is False:
            return cls()
        if raw is True:
            raw = {"preset": "camera"}
        if isinstance(raw, (list, tuple)):
            raw = {"stages": list(raw)}
        if not isinstance(raw, dict):
            raise ValueError("capture must be a boolean, object, or stage list")
        if raw.get("enabled", True) is False:
            return cls()
        camera_profile = _resolve_camera_profile(config, raw)
        stage_overrides = _resolve_capture_stage_overrides(config, raw)

        stages = raw.get("stages")
        if stages is None:
            preset = raw.get("preset", raw.get("profile"))
            if preset is None:
                return cls()
            stages = _preset_stages(str(preset))
        if stages is None:
            stages = ()
        if not isinstance(stages, (list, tuple)):
            raise ValueError("capture.stages must be a list")

        parsed: list[CaptureArtifactStage] = []
        for entry in stages:
            parsed.extend(
                _build_stage(
                    entry,
                    camera_profile=camera_profile,
                    stage_overrides=stage_overrides,
                )
            )
        return cls(tuple(parsed))

    def apply_np(self, image, context: CaptureContext):
        for stage in self.stages:
            image = stage.apply_np(image, context)
        return image

    def apply_torch(self, image, context: CaptureContext):
        if not self.stages:
            return image
        image_t = image
        for stage in self.stages:
            image_t = stage.apply_torch(image_t, context)
        return image_t

    def apply_torch_batch(self, images, contexts: tuple[CaptureContext, ...]):
        if not self.stages:
            return images
        processed = [
            self.apply_torch(images[index], context)
            for index, context in enumerate(contexts)
        ]
        return _stack_like(images, processed)


_STAGE_TYPES = {
    "optics": OpticsStage,
    "lens": OpticsStage,
    "windshield": OpticsStage,
    "exposure": ExposureStage,
    "sensor": SensorStage,
    "raw_sensor": SensorStage,
    "sensor_raw": SensorStage,
    "bayer": SensorStage,
    "noise": SensorStage,
    "isp": ISPStage,
    "transport": TransportStage,
    "compression": TransportStage,
}

_CANONICAL_STAGE_TYPES = {
    "lens": "optics",
    "windshield": "optics",
    "raw_sensor": "sensor",
    "sensor_raw": "sensor",
    "bayer": "sensor",
    "noise": "sensor",
    "compression": "transport",
}

_BUILTIN_CAMERA_PROFILES: dict[str, dict[str, Any]] = {
    "default": {},
    "generic": {},
    "dashcam": {
        "optics": {
            "windshield_haze": {"enabled": True, "probability": 0.5},
            "vignetting_strength": {"dist": "uniform", "min": 0.08, "max": 0.24},
        },
        "sensor": {
            "iso": {"dist": "choice", "values": [200.0, 400.0, 800.0]},
            "full_well_electrons": {
                "dist": "uniform",
                "min": 6000.0,
                "max": 18000.0,
            },
        },
        "transport": {
            "jpeg": {
                "enabled": True,
                "quality": {"dist": "uniform", "min": 58.0, "max": 88.0},
            }
        },
    },
    "low_light_fog": {
        "optics": {
            "bloom": {"strength": {"dist": "uniform", "min": 0.08, "max": 0.22}},
            "veiling_glare_strength": {"dist": "uniform", "min": 0.02, "max": 0.08},
            "fog_coupled_glare": {
                "enabled": True,
                "fog_strength": {"dist": "uniform", "min": 0.015, "max": 0.055},
                "highlight_strength": {"dist": "uniform", "min": 0.02, "max": 0.08},
                "airlight_strength": {"dist": "uniform", "min": 0.0, "max": 0.035},
                "smooth_sigma": {"dist": "uniform", "min": 10.0, "max": 22.0},
            },
        },
        "sensor": {
            "iso": {"dist": "choice", "values": [800.0, 1600.0, 3200.0]},
            "auto_exposure": {
                "enabled": True,
                "metering": "fog_aware_center_weighted",
                "target_luminance": {"dist": "uniform", "min": 0.13, "max": 0.21},
                "highlight_protection": 0.78,
                "manual_gain_weight": 0.0,
                "sky_suppression": 0.85,
                "fog_meter_suppression": 0.65,
                "depth_meter_decay_m": 35.0,
                "min_meter_weight": 0.05,
                "resolve_iso": True,
                "fog_iso_boost": 0.25,
            },
            "read_noise_electrons": {"dist": "uniform", "min": 3.0, "max": 10.0},
            "row_noise_sigma": {"dist": "uniform", "min": 0.002, "max": 0.009},
            "sensor_identity": {
                "enabled": True,
                "sensor_id": "low_light_fog",
                "seed": 2207,
                "prnu_sigma": {"dist": "uniform", "min": 0.001, "max": 0.004},
                "dsnu_sigma": {"dist": "uniform", "min": 0.00003, "max": 0.00018},
                "persistent_hot_pixel_probability": 0.00001,
                "persistent_dead_pixel_probability": 0.000005,
                "persistent_row_sigma": 0.00025,
                "persistent_column_sigma": 0.00016,
            },
        },
        "isp": {
            "denoise_sigma": {"dist": "uniform", "min": 0.25, "max": 0.8},
            "sharpen_amount": {"dist": "uniform", "min": 0.08, "max": 0.28},
            "tone_map": "lut",
            "tone_map_strength": 1.0,
            "tone_map_lut": [
                0.0,
                0.006,
                0.014,
                0.028,
                0.052,
                0.090,
                0.145,
                0.220,
                0.320,
                0.450,
                0.610,
                0.780,
                0.900,
                0.965,
                0.995,
                1.0,
            ],
        },
        "transport": {
            "jpeg": {
                "enabled": True,
                "quality": {"dist": "uniform", "min": 58.0, "max": 86.0},
            },
            "bit_depth": 8,
        },
    },
}


_CAMERA_PRESET = (
    {"type": "optics"},
    {"type": "sensor", "input_space": "srgb"},
    {"type": "isp"},
    {"type": "transport"},
)


def _preset_stages(name: str) -> tuple[dict[str, Any], ...]:
    normalized = name.strip().lower().replace("-", "_")
    if normalized in {"camera", "camera_stack", "foggy_camera", "realistic_camera"}:
        return tuple(dict(stage) for stage in _CAMERA_PRESET)
    if normalized in {"none", "off", "false", "noop", "no_op"}:
        return ()
    raise ValueError(f"Unsupported capture preset: {name}")


def _build_stage(
    entry: Any,
    *,
    camera_profile: Mapping[str, Any] | None = None,
    stage_overrides: Mapping[str, Any] | None = None,
) -> list[CaptureArtifactStage]:
    if isinstance(entry, str):
        stage_type = entry
        cfg: dict[str, Any] = {}
    elif isinstance(entry, dict):
        stage_type = str(entry.get("type", entry.get("name", "")))
        cfg = dict(entry)
    else:
        raise ValueError("capture stage entries must be strings or objects")

    normalized = stage_type.strip().lower().replace("-", "_")
    if normalized in {"camera", "camera_stack", "foggy_camera", "realistic_camera"}:
        stages = []
        for preset_entry in _preset_stages(normalized):
            stages.extend(
                _build_stage(
                    preset_entry,
                    camera_profile=camera_profile,
                    stage_overrides=stage_overrides,
                )
            )
        return stages

    stage_cls = _STAGE_TYPES.get(normalized)
    if stage_cls is None:
        raise ValueError(f"Unsupported capture artifact stage: {stage_type}")
    profile_cfg = _profile_stage_config(camera_profile or {}, normalized)
    override_cfg = _profile_stage_config(stage_overrides or {}, normalized)
    cfg = deep_merge(profile_cfg, cfg)
    cfg = deep_merge(cfg, override_cfg)
    return [stage_cls(cfg)]


def _resolve_capture_stage_overrides(
    config: Mapping[str, Any],
    capture_config: Mapping[str, Any],
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for source, keys in (
        (config, ("capture_overrides", "capture_stage_overrides")),
        (capture_config, ("overrides", "stage_overrides", "capture_overrides")),
    ):
        for key in keys:
            raw = source.get(key)
            if raw is None:
                continue
            if not isinstance(raw, dict):
                raise ValueError(f"{key} must be an object")
            overrides = deep_merge(overrides, raw)
    return overrides


def _resolve_camera_profile(
    config: Mapping[str, Any],
    capture_config: Mapping[str, Any],
) -> dict[str, Any]:
    profiles = {
        name: dict(profile) for name, profile in _BUILTIN_CAMERA_PROFILES.items()
    }
    configured_profiles = config.get("camera_profiles", {})
    if configured_profiles is not None:
        if not isinstance(configured_profiles, dict):
            raise ValueError("camera_profiles must be an object")
        profiles = deep_merge(profiles, configured_profiles)

    profile: dict[str, Any] = {}
    root_profile = config.get("camera_profile")
    if root_profile is not None:
        profile = deep_merge(profile, _resolve_profile_ref(root_profile, profiles))

    capture_profile = capture_config.get("camera_profile")
    if capture_profile is not None:
        profile = deep_merge(profile, _resolve_profile_ref(capture_profile, profiles))

    return profile


def _resolve_profile_ref(
    value: Any,
    profiles: Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(value, str):
        if value not in profiles:
            known = ", ".join(sorted(profiles))
            raise ValueError(f"Unknown camera_profile '{value}'. Known: {known}")
        profile = profiles[value]
        if not isinstance(profile, dict):
            raise ValueError(f"camera_profiles.{value} must be an object")
        return dict(profile)

    if isinstance(value, dict):
        inline = dict(value)
        name = inline.pop("name", None)
        base = {}
        if name is not None:
            base = _resolve_profile_ref(str(name), profiles)
        return deep_merge(base, inline)

    raise ValueError("camera_profile must be a string or object")


def _profile_stage_config(
    camera_profile: Mapping[str, Any],
    stage_type: str,
) -> dict[str, Any]:
    canonical = _CANONICAL_STAGE_TYPES.get(stage_type, stage_type)
    raw = camera_profile.get(canonical, {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"camera_profile.{canonical} must be an object")
    return dict(raw)


def _stack_like(reference, images: list[Any]):
    if not images:
        return reference
    if hasattr(reference, "new_empty") and hasattr(reference, "dim"):
        if torch is None:  # pragma: no cover - defensive
            raise RuntimeError("Torch input received but torch is unavailable")
        return torch.stack(images, dim=0)
    raise TypeError("Unsupported batch image type for capture pipeline")


def _rng(context: CaptureContext) -> np.random.Generator:
    if isinstance(context.rng, np.random.Generator):
        return context.rng
    return np.random.default_rng()


_CONDITION_PROFILE_KEYS = (
    "condition_profiles",
    "profile_choices",
    "exposure_profiles",
)
_CONDITION_PROFILE_SELECTOR_KEYS = (
    "condition_profile",
    "selected_condition_profile",
)
_CONDITION_PROFILE_METADATA = {
    "name",
    "id",
    "description",
    "weight",
    "profile_weight",
    "profile_probability",
}


def _conditioned_stage_config(
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> Mapping[str, Any]:
    profile_key = next(
        (key for key in _CONDITION_PROFILE_KEYS if config.get(key) is not None),
        None,
    )
    if profile_key is None:
        return config

    profiles = config.get(profile_key)
    if not isinstance(profiles, (list, tuple)):
        raise ValueError(f"{profile_key} must be a list")
    if not profiles:
        return config

    base = {
        key: value
        for key, value in dict(config).items()
        if key not in _CONDITION_PROFILE_KEYS
        and key not in _CONDITION_PROFILE_SELECTOR_KEYS
    }
    weights: list[float] = []
    choices: list[dict[str, Any]] = []
    names: list[str | None] = []
    for index, entry in enumerate(profiles):
        if not isinstance(entry, dict):
            raise ValueError(f"{profile_key}[{index}] must be an object")
        weight = float(
            entry.get(
                "weight",
                entry.get("profile_weight", entry.get("profile_probability", 1.0)),
            )
        )
        if weight < 0.0:
            raise ValueError(f"{profile_key}[{index}].weight must be non-negative")
        raw_profile = entry.get("config", entry)
        if not isinstance(raw_profile, dict):
            raise ValueError(f"{profile_key}[{index}].config must be an object")
        profile = {
            key: value
            for key, value in raw_profile.items()
            if key not in _CONDITION_PROFILE_METADATA
        }
        weights.append(weight)
        choices.append(profile)
        name = entry.get("name", entry.get("id"))
        names.append(None if name is None else str(name))

    selector = _condition_profile_selector(config)
    if selector is not None:
        for name, profile in zip(names, choices):
            if name == selector:
                return deep_merge(base, profile)
        available = ", ".join(name for name in names if name is not None)
        raise ValueError(
            f"Unknown condition_profile '{selector}' for {profile_key}. "
            f"Known: {available or '<none>'}"
        )

    weights_arr = np.asarray(weights, dtype=np.float64)
    total = float(weights_arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{profile_key} must contain at least one positive weight")

    selected = choices[int(rng.choice(len(choices), p=weights_arr / total))]
    return deep_merge(base, selected)


def _condition_profile_selector(config: Mapping[str, Any]) -> str | None:
    for key in _CONDITION_PROFILE_SELECTOR_KEYS:
        value = config.get(key)
        if value is not None:
            return str(value)
    return None


def _context_intrinsics(context: CaptureContext) -> np.ndarray | None:
    if context.intrinsics is None:
        return None
    intrinsics = np.asarray(context.intrinsics, dtype=np.float32)
    if intrinsics.shape != (3, 3):
        return None
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    if abs(fx) < 1e-6 or abs(fy) < 1e-6:
        return None
    return intrinsics


def _context_float_map(value: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if value is None:
        return None
    if torch is not None and torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim == 0:
        return np.full(shape, float(arr), dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        return None
    if arr.shape != shape:
        arr = _resize_map_np(arr, shape)
    return arr.astype(np.float32, copy=False)


def _context_depth_map(
    context: CaptureContext, shape: tuple[int, int]
) -> np.ndarray | None:
    return _context_float_map(context.depth_m, shape)


def _context_fog_opacity(
    context: CaptureContext,
    shape: tuple[int, int],
) -> np.ndarray | None:
    opacity = _context_float_map(context.fog_opacity, shape)
    if opacity is not None:
        return np.clip(opacity, 0.0, 1.0).astype(np.float32, copy=False)
    depth = _context_depth_map(context, shape)
    k_map = _context_float_map(context.k_map, shape)
    if depth is None or k_map is None:
        return None
    return np.clip(
        1.0 - np.exp(-np.maximum(depth, 0.0) * np.maximum(k_map, 0.0)), 0.0, 1.0
    ).astype(
        np.float32,
        copy=False,
    )


def _context_float_map_torch(
    value: Any,
    shape: tuple[int, int],
    device,
    dtype,
):
    if value is None or torch is None:
        return None
    if torch.is_tensor(value):
        arr = value.to(device=device, dtype=dtype)
    else:
        arr = torch.as_tensor(value, device=device, dtype=dtype)
    if arr.ndim == 0:
        return torch.full(shape, float(arr.detach().cpu()), device=device, dtype=dtype)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        return None
    arr = torch.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if tuple(arr.shape) != tuple(shape):
        arr = _resize_map_torch(arr, shape)
    return arr.to(device=device, dtype=dtype)


def _context_depth_map_torch(
    context: CaptureContext,
    shape: tuple[int, int],
    device,
    dtype,
):
    return _context_float_map_torch(context.depth_m, shape, device, dtype)


def _context_fog_opacity_torch(
    context: CaptureContext,
    shape: tuple[int, int],
    device,
    dtype,
):
    opacity = _context_float_map_torch(context.fog_opacity, shape, device, dtype)
    if opacity is not None:
        return torch.clamp(opacity, 0.0, 1.0)
    depth = _context_depth_map_torch(context, shape, device, dtype)
    k_map = _context_float_map_torch(context.k_map, shape, device, dtype)
    if depth is None or k_map is None:
        return None
    return torch.clamp(
        1.0 - torch.exp(-torch.clamp(depth, min=0.0) * torch.clamp(k_map, min=0.0)),
        0.0,
        1.0,
    )


def _context_attribute_map(
    context: CaptureContext,
    key: str,
    shape: tuple[int, int],
) -> np.ndarray | None:
    attrs = context.attributes or {}
    return _context_float_map(attrs.get(key), shape)


def _context_attribute_map_torch(
    context: CaptureContext,
    key: str,
    shape: tuple[int, int],
    device,
    dtype,
):
    attrs = context.attributes or {}
    return _context_float_map_torch(attrs.get(key), shape, device, dtype)


def _as_float_rgb(image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Capture stages expect RGB HWC images, got {arr.shape}")
    if np.issubdtype(arr.dtype, np.integer):
        max_value = np.iinfo(arr.dtype).max
        arr = arr.astype(np.float32) / float(max_value)
    else:
        arr = arr.astype(np.float32, copy=False)
    return _clip01(arr)


def _clip01(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)


def _clip01_torch(image):
    return torch.clamp(image, 0.0, 1.0)


def _as_float_rgb_torch(image):
    if torch is None or not torch.is_tensor(image):
        return _as_float_rgb(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"Capture stages expect RGB HWC images, got {tuple(image.shape)}"
        )
    if image.is_floating_point():
        arr = image.to(dtype=torch.float32)
    else:
        max_value = float(torch.iinfo(image.dtype).max)
        arr = image.to(dtype=torch.float32) / max_value
    return _clip01_torch(arr)


def _torch_to_numpy_image(image) -> np.ndarray:
    if torch is None or not torch.is_tensor(image):
        return _as_float_rgb(image)
    return _as_float_rgb(image.detach().cpu().numpy())


def _numpy_to_torch_like(image: np.ndarray, reference):
    if torch is None or not torch.is_tensor(reference):
        return image
    return torch.from_numpy(np.asarray(image, dtype=np.float32)).to(
        device=reference.device,
        dtype=reference.dtype,
    )


def _sample_any(spec: Any, rng: np.random.Generator) -> Any:
    if spec is None or isinstance(spec, bool):
        return spec
    return sample_value(spec, rng)


def _sample_float(
    config: Mapping[str, Any],
    key: str,
    default: float,
    rng: np.random.Generator,
) -> float:
    value = config.get(key, default)
    if value is None:
        return float(default)
    return float(sample_value(value, rng))


def _sample_triplet(spec: Any, rng: np.random.Generator) -> np.ndarray:
    if spec is None:
        return np.ones(3, dtype=np.float32)
    value = sample_value(spec, rng)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = np.repeat(arr, 3)
    if arr.shape != (3,):
        raise ValueError("Expected scalar or RGB triplet")
    return arr


def _sample_matrix(spec: Any, rng: np.random.Generator) -> np.ndarray:
    if spec is None:
        return np.eye(3, dtype=np.float32)
    value = sample_value(spec, rng)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (3, 3):
        raise ValueError("Color matrix must have shape 3x3")
    return arr


def _sample_sensor_levels(
    config: Mapping[str, Any],
    key: str,
    rng: np.random.Generator,
) -> np.ndarray:
    return _sample_triplet(config.get(key, [1.0, 1.0, 1.0]), rng)


def _resolve_auto_exposure_np(
    image: np.ndarray,
    context: CaptureContext,
    config: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    manual_exposure_gain: float,
) -> tuple[float, Mapping[str, Any]]:
    cfg = _config_block(config, "auto_exposure")
    if not _block_enabled(cfg, rng):
        return float(manual_exposure_gain), config

    metrics = _auto_exposure_metrics_np(image, context, cfg, rng)
    meter = max(float(metrics["meter_luminance"]), 1e-6)
    target = max(_sample_float(cfg, "target_luminance", 0.18, rng), 1e-6)
    auto_gain = target / meter

    highlight = max(float(metrics["highlight_luminance"]), 1e-6)
    highlight_target = max(_sample_float(cfg, "highlight_target", 0.92, rng), 1e-6)
    highlight_gain = highlight_target / highlight
    protection = float(
        np.clip(_sample_float(cfg, "highlight_protection", 0.65, rng), 0.0, 1.0)
    )
    if protection > 0.0 and auto_gain > highlight_gain > 0.0:
        log_auto = math.log(max(auto_gain, 1e-6))
        log_highlight = math.log(max(highlight_gain, 1e-6))
        auto_gain = math.exp((1.0 - protection) * log_auto + protection * log_highlight)

    compensation_ev = _sample_float(cfg, "exposure_compensation_ev", 0.0, rng)
    auto_gain *= 2.0**compensation_ev

    manual_weight = max(_sample_float(cfg, "manual_gain_weight", 1.0, rng), 0.0)
    manual_gain = max(float(manual_exposure_gain), 1e-6)
    exposure = auto_gain * (manual_gain**manual_weight)
    min_gain = max(_sample_float(cfg, "min_gain", 0.0, rng), 0.0)
    max_gain = max(_sample_float(cfg, "max_gain", 2.6, rng), max(min_gain, 1e-6))
    exposure = float(np.clip(exposure, min_gain, max_gain))

    if not _bool_value(cfg.get("resolve_iso", False)):
        return exposure, config

    resolved_config = dict(config)
    base_iso = max(_sample_float(config, "base_iso", 100.0, rng), 1e-6)
    configured_iso = max(_sample_float(config, "iso", base_iso, rng), 1e-6)
    min_iso_cfg = cfg.get("min_iso")
    min_iso = (
        max(float(sample_value(min_iso_cfg, rng)), 1e-6)
        if min_iso_cfg is not None
        else base_iso
    )
    max_iso = max(_sample_float(cfg, "max_iso", 1600.0, rng), min_iso)
    activation_gain = max(_sample_float(cfg, "iso_activation_gain", 1.2, rng), 1e-6)
    iso_power = max(_sample_float(cfg, "iso_gain_power", 0.85, rng), 0.0)
    iso_pressure = max(auto_gain / activation_gain, 1.0)
    resolved_iso = configured_iso * (iso_pressure**iso_power)

    dark_iso_boost = max(_sample_float(cfg, "dark_iso_boost", 0.0, rng), 0.0)
    if dark_iso_boost > 0.0:
        resolved_iso *= 1.0 + dark_iso_boost * float(metrics["dark_fraction"])

    fog_iso_boost = max(_sample_float(cfg, "fog_iso_boost", 0.0, rng), 0.0)
    if fog_iso_boost > 0.0:
        resolved_iso *= 1.0 + fog_iso_boost * float(metrics["mean_fog_opacity"])

    resolved_config["iso"] = float(np.clip(resolved_iso, min_iso, max_iso))
    resolved_config["base_iso"] = float(base_iso)
    return exposure, resolved_config


def _resolve_auto_exposure_torch(
    image,
    context: CaptureContext,
    config: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    manual_exposure_gain: float,
) -> tuple[float, Mapping[str, Any]]:
    cfg = _config_block(config, "auto_exposure")
    if not _block_enabled(cfg, rng):
        return float(manual_exposure_gain), config

    metrics = _auto_exposure_metrics_torch(image, context, cfg, rng)
    meter = max(float(metrics["meter_luminance"]), 1e-6)
    target = max(_sample_float(cfg, "target_luminance", 0.18, rng), 1e-6)
    auto_gain = target / meter

    highlight = max(float(metrics["highlight_luminance"]), 1e-6)
    highlight_target = max(_sample_float(cfg, "highlight_target", 0.92, rng), 1e-6)
    highlight_gain = highlight_target / highlight
    protection = float(
        np.clip(_sample_float(cfg, "highlight_protection", 0.65, rng), 0.0, 1.0)
    )
    if protection > 0.0 and auto_gain > highlight_gain > 0.0:
        log_auto = math.log(max(auto_gain, 1e-6))
        log_highlight = math.log(max(highlight_gain, 1e-6))
        auto_gain = math.exp((1.0 - protection) * log_auto + protection * log_highlight)

    compensation_ev = _sample_float(cfg, "exposure_compensation_ev", 0.0, rng)
    auto_gain *= 2.0**compensation_ev

    manual_weight = max(_sample_float(cfg, "manual_gain_weight", 1.0, rng), 0.0)
    manual_gain = max(float(manual_exposure_gain), 1e-6)
    exposure = auto_gain * (manual_gain**manual_weight)
    min_gain = max(_sample_float(cfg, "min_gain", 0.0, rng), 0.0)
    max_gain = max(_sample_float(cfg, "max_gain", 2.6, rng), max(min_gain, 1e-6))
    exposure = float(np.clip(exposure, min_gain, max_gain))

    if not _bool_value(cfg.get("resolve_iso", False)):
        return exposure, config

    resolved_config = dict(config)
    base_iso = max(_sample_float(config, "base_iso", 100.0, rng), 1e-6)
    configured_iso = max(_sample_float(config, "iso", base_iso, rng), 1e-6)
    min_iso_cfg = cfg.get("min_iso")
    min_iso = (
        max(float(sample_value(min_iso_cfg, rng)), 1e-6)
        if min_iso_cfg is not None
        else base_iso
    )
    max_iso = max(_sample_float(cfg, "max_iso", 1600.0, rng), min_iso)
    activation_gain = max(_sample_float(cfg, "iso_activation_gain", 1.2, rng), 1e-6)
    iso_power = max(_sample_float(cfg, "iso_gain_power", 0.85, rng), 0.0)
    iso_pressure = max(auto_gain / activation_gain, 1.0)
    resolved_iso = configured_iso * (iso_pressure**iso_power)

    dark_iso_boost = max(_sample_float(cfg, "dark_iso_boost", 0.0, rng), 0.0)
    if dark_iso_boost > 0.0:
        resolved_iso *= 1.0 + dark_iso_boost * float(metrics["dark_fraction"])

    fog_iso_boost = max(_sample_float(cfg, "fog_iso_boost", 0.0, rng), 0.0)
    if fog_iso_boost > 0.0:
        resolved_iso *= 1.0 + fog_iso_boost * float(metrics["mean_fog_opacity"])

    resolved_config["iso"] = float(np.clip(resolved_iso, min_iso, max_iso))
    resolved_config["base_iso"] = float(base_iso)
    return exposure, resolved_config


def _auto_exposure_metrics_np(
    image: np.ndarray,
    context: CaptureContext,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, float]:
    luminance = _linear_luminance_np(image)
    finite = np.isfinite(luminance)
    if not finite.any():
        return {
            "meter_luminance": 0.18,
            "mean_luminance": 0.18,
            "center_luminance": 0.18,
            "percentile_luminance": 0.18,
            "highlight_luminance": 0.18,
            "dark_fraction": 0.0,
            "contrast": 0.0,
            "mean_fog_opacity": 0.0,
        }
    luma = np.clip(luminance[finite], 0.0, None)
    mean_luma = float(luma.mean())
    center_luma = _center_weighted_mean_np(luminance, config, rng)
    meter_percentile = float(
        np.clip(_sample_float(config, "meter_percentile", 50.0, rng), 0.0, 100.0)
    )
    percentile_luma = float(np.percentile(luma, meter_percentile))
    highlight_percentile = float(
        np.clip(_sample_float(config, "highlight_percentile", 98.5, rng), 0.0, 100.0)
    )
    highlight_luma = float(np.percentile(luma, highlight_percentile))
    low_luma = float(np.percentile(luma, 5.0))
    high_luma = float(np.percentile(luma, 95.0))
    dark_threshold = max(
        _sample_float(config, "dark_fraction_threshold", 0.12, rng),
        0.0,
    )
    dark_fraction = float(np.mean(luma < dark_threshold))
    opacity = _context_fog_opacity(context, luminance.shape)
    mean_fog_opacity = float(np.mean(opacity)) if opacity is not None else 0.0

    metering = str(config.get("metering", "center_weighted")).strip().lower()
    center_weight = float(
        np.clip(_sample_float(config, "center_weight", 0.55, rng), 0.0, 1.0)
    )
    use_weighted = _auto_exposure_uses_weighted_metering(config)
    if use_weighted:
        center_weighted = metering in {
            "fog_aware_center_weighted",
            "fog-aware-center-weighted",
            "sky_aware_center_weighted",
            "sky-aware-center-weighted",
            "center_weighted",
            "center-weighted",
        }
        weights = _auto_exposure_weights_np(
            luminance,
            context,
            config,
            rng,
            center_weighted=center_weighted,
        )
    else:
        weights = None

    if weights is not None:
        weighted_mean = _weighted_mean_np(luminance, weights)
        weighted_percentile = _weighted_percentile_np(
            luminance,
            weights,
            meter_percentile,
        )
        weighted_highlight = _weighted_percentile_np(
            luminance,
            weights,
            highlight_percentile,
        )
        weighted_low = _weighted_percentile_np(luminance, weights, 5.0)
        weighted_high = _weighted_percentile_np(luminance, weights, 95.0)
        weighted_dark = _weighted_mean_np(
            (luminance < dark_threshold).astype(np.float32), weights
        )
        percentile_luma = weighted_percentile
        highlight_luma = weighted_highlight
        low_luma = weighted_low
        high_luma = weighted_high
        dark_fraction = weighted_dark

        if metering in {"mean", "average"}:
            meter_luma = weighted_mean
        elif metering in {"percentile", "median"}:
            meter_luma = weighted_percentile
        elif metering in {"center_percentile", "centered_percentile"}:
            meter_luma = (
                1.0 - center_weight
            ) * weighted_percentile + center_weight * weighted_mean
        elif metering in {"highlight", "highlight_protect", "highlight_protected"}:
            meter_luma = max(weighted_mean, weighted_percentile)
        elif metering in _FOG_AWARE_METERING_MODES:
            meter_luma = weighted_percentile
        else:
            meter_luma = weighted_mean
    else:
        if metering in {"mean", "average"}:
            meter_luma = mean_luma
        elif metering in {"percentile", "median"}:
            meter_luma = percentile_luma
        elif metering in {"center_percentile", "centered_percentile"}:
            meter_luma = (
                1.0 - center_weight
            ) * percentile_luma + center_weight * center_luma
        elif metering in {"highlight", "highlight_protect", "highlight_protected"}:
            meter_luma = max(center_luma, percentile_luma)
        else:
            meter_luma = (1.0 - center_weight) * mean_luma + center_weight * center_luma

    return {
        "meter_luminance": max(float(meter_luma), 1e-6),
        "mean_luminance": mean_luma,
        "center_luminance": center_luma,
        "percentile_luminance": percentile_luma,
        "highlight_luminance": highlight_luma,
        "dark_fraction": dark_fraction,
        "contrast": max(high_luma - low_luma, 0.0),
        "mean_fog_opacity": mean_fog_opacity,
    }


def _auto_exposure_metrics_torch(
    image,
    context: CaptureContext,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> dict[str, float]:
    luminance = _linear_luminance_torch(image)
    finite = torch.isfinite(luminance)
    if not bool(torch.any(finite).detach().cpu()):
        return {
            "meter_luminance": 0.18,
            "mean_luminance": 0.18,
            "center_luminance": 0.18,
            "percentile_luminance": 0.18,
            "highlight_luminance": 0.18,
            "dark_fraction": 0.0,
            "contrast": 0.0,
            "mean_fog_opacity": 0.0,
        }
    luma = torch.clamp(luminance[finite], min=0.0)
    mean_luma = float(torch.mean(luma).detach().cpu())
    center_luma = _center_weighted_mean_torch(luminance, config, rng)
    meter_percentile = float(
        np.clip(_sample_float(config, "meter_percentile", 50.0, rng), 0.0, 100.0)
    )
    percentile_luma = float(
        torch.quantile(luma, meter_percentile / 100.0).detach().cpu()
    )
    highlight_percentile = float(
        np.clip(_sample_float(config, "highlight_percentile", 98.5, rng), 0.0, 100.0)
    )
    highlight_luma = float(
        torch.quantile(luma, highlight_percentile / 100.0).detach().cpu()
    )
    low_luma = float(torch.quantile(luma, 0.05).detach().cpu())
    high_luma = float(torch.quantile(luma, 0.95).detach().cpu())
    dark_threshold = max(
        _sample_float(config, "dark_fraction_threshold", 0.12, rng),
        0.0,
    )
    dark_fraction = float(
        torch.mean((luma < dark_threshold).to(luma.dtype)).detach().cpu()
    )
    opacity = _context_fog_opacity_torch(
        context,
        tuple(luminance.shape),
        luminance.device,
        luminance.dtype,
    )
    mean_fog_opacity = (
        float(torch.mean(opacity).detach().cpu()) if opacity is not None else 0.0
    )

    metering = str(config.get("metering", "center_weighted")).strip().lower()
    center_weight = float(
        np.clip(_sample_float(config, "center_weight", 0.55, rng), 0.0, 1.0)
    )
    use_weighted = _auto_exposure_uses_weighted_metering(config)
    if use_weighted:
        center_weighted = metering in {
            "fog_aware_center_weighted",
            "fog-aware-center-weighted",
            "sky_aware_center_weighted",
            "sky-aware-center-weighted",
            "center_weighted",
            "center-weighted",
        }
        weights = _auto_exposure_weights_torch(
            luminance,
            context,
            config,
            rng,
            center_weighted=center_weighted,
        )
    else:
        weights = None

    if weights is not None:
        weighted_mean = _weighted_mean_torch(luminance, weights)
        weighted_percentile = _weighted_percentile_torch(
            luminance,
            weights,
            meter_percentile,
        )
        weighted_highlight = _weighted_percentile_torch(
            luminance,
            weights,
            highlight_percentile,
        )
        weighted_low = _weighted_percentile_torch(luminance, weights, 5.0)
        weighted_high = _weighted_percentile_torch(luminance, weights, 95.0)
        weighted_dark = _weighted_mean_torch(
            (luminance < dark_threshold).to(luminance.dtype),
            weights,
        )
        percentile_luma = weighted_percentile
        highlight_luma = weighted_highlight
        low_luma = weighted_low
        high_luma = weighted_high
        dark_fraction = weighted_dark

        if metering in {"mean", "average"}:
            meter_luma = weighted_mean
        elif metering in {"percentile", "median"}:
            meter_luma = weighted_percentile
        elif metering in {"center_percentile", "centered_percentile"}:
            meter_luma = (
                1.0 - center_weight
            ) * weighted_percentile + center_weight * weighted_mean
        elif metering in {"highlight", "highlight_protect", "highlight_protected"}:
            meter_luma = max(weighted_mean, weighted_percentile)
        elif metering in _FOG_AWARE_METERING_MODES:
            meter_luma = weighted_percentile
        else:
            meter_luma = weighted_mean
    else:
        if metering in {"mean", "average"}:
            meter_luma = mean_luma
        elif metering in {"percentile", "median"}:
            meter_luma = percentile_luma
        elif metering in {"center_percentile", "centered_percentile"}:
            meter_luma = (
                1.0 - center_weight
            ) * percentile_luma + center_weight * center_luma
        elif metering in {"highlight", "highlight_protect", "highlight_protected"}:
            meter_luma = max(center_luma, percentile_luma)
        else:
            meter_luma = (1.0 - center_weight) * mean_luma + center_weight * center_luma

    return {
        "meter_luminance": max(float(meter_luma), 1e-6),
        "mean_luminance": mean_luma,
        "center_luminance": center_luma,
        "percentile_luminance": percentile_luma,
        "highlight_luminance": highlight_luma,
        "dark_fraction": dark_fraction,
        "contrast": max(high_luma - low_luma, 0.0),
        "mean_fog_opacity": mean_fog_opacity,
    }


def _linear_luminance_np(image: np.ndarray) -> np.ndarray:
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.sum(np.clip(image, 0.0, None) * weights, axis=-1).astype(
        np.float32,
        copy=False,
    )


def _linear_luminance_torch(image):
    weights = torch.tensor(
        [0.2126, 0.7152, 0.0722],
        device=image.device,
        dtype=image.dtype,
    )
    return torch.sum(torch.clamp(image, min=0.0) * weights.view(1, 1, 3), dim=-1)


def _center_weighted_mean_np(
    luminance: np.ndarray,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> float:
    height, width = luminance.shape
    if height <= 0 or width <= 0:
        return 0.18
    sigma = max(_sample_float(config, "center_sigma", 0.35, rng), 1e-3)
    yy, xx = _coordinate_grid(height, width)
    x = (xx - (width - 1) / 2.0) / max(width - 1, 1)
    y = (yy - (height - 1) / 2.0) / max(height - 1, 1)
    weights = np.exp(-0.5 * (x * x + y * y) / (sigma * sigma)).astype(np.float32)
    finite = np.isfinite(luminance)
    if not finite.any():
        return 0.18
    weighted = np.where(finite, np.clip(luminance, 0.0, None), 0.0) * weights
    return float(weighted.sum() / max(float(weights[finite].sum()), 1e-6))


def _center_weighted_mean_torch(
    luminance,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> float:
    height, width = int(luminance.shape[0]), int(luminance.shape[1])
    if height <= 0 or width <= 0:
        return 0.18
    sigma = max(_sample_float(config, "center_sigma", 0.35, rng), 1e-3)
    yy, xx = _coordinate_grid_torch(height, width, luminance.device, luminance.dtype)
    x = (xx - (width - 1) / 2.0) / max(width - 1, 1)
    y = (yy - (height - 1) / 2.0) / max(height - 1, 1)
    weights = torch.exp(-0.5 * (x * x + y * y) / (sigma * sigma))
    finite = torch.isfinite(luminance)
    if not bool(torch.any(finite).detach().cpu()):
        return 0.18
    weighted = (
        torch.where(
            finite,
            torch.clamp(luminance, min=0.0),
            torch.zeros_like(luminance),
        )
        * weights
    )
    denom = torch.clamp(torch.sum(weights[finite]), min=1e-6)
    return float((torch.sum(weighted) / denom).detach().cpu())


_FOG_AWARE_METERING_MODES = {
    "fog_aware",
    "fog-aware",
    "fog_aware_center_weighted",
    "fog-aware-center-weighted",
    "sky_aware_center_weighted",
    "sky-aware-center-weighted",
}

_AUTO_EXPOSURE_WEIGHT_KEYS = {
    "sky_suppression",
    "fog_meter_suppression",
    "depth_meter_decay_m",
    "min_meter_weight",
}


def _auto_exposure_uses_weighted_metering(config: Mapping[str, Any]) -> bool:
    metering = str(config.get("metering", "center_weighted")).strip().lower()
    return metering in _FOG_AWARE_METERING_MODES or any(
        key in config for key in _AUTO_EXPOSURE_WEIGHT_KEYS
    )


def _auto_exposure_center_weight_map_np(
    shape: tuple[int, int],
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = shape
    sigma = max(_sample_float(config, "center_sigma", 0.35, rng), 1e-3)
    yy, xx = _coordinate_grid(height, width)
    x = (xx - (width - 1) / 2.0) / max(width - 1, 1)
    y = (yy - (height - 1) / 2.0) / max(height - 1, 1)
    return np.exp(-0.5 * (x * x + y * y) / (sigma * sigma)).astype(np.float32)


def _auto_exposure_center_weight_map_torch(
    shape: tuple[int, int],
    config: Mapping[str, Any],
    rng: np.random.Generator,
    device,
    dtype,
):
    height, width = shape
    sigma = max(_sample_float(config, "center_sigma", 0.35, rng), 1e-3)
    yy, xx = _coordinate_grid_torch(height, width, device, dtype)
    x = (xx - (width - 1) / 2.0) / max(width - 1, 1)
    y = (yy - (height - 1) / 2.0) / max(height - 1, 1)
    return torch.exp(-0.5 * (x * x + y * y) / (sigma * sigma))


def _auto_exposure_weights_np(
    luminance: np.ndarray,
    context: CaptureContext,
    config: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    center_weighted: bool,
) -> np.ndarray | None:
    shape = luminance.shape
    metering = str(config.get("metering", "center_weighted")).strip().lower()
    weights = (
        _auto_exposure_center_weight_map_np(shape, config, rng)
        if center_weighted
        else np.ones(shape, dtype=np.float32)
    )

    default_depth_decay = (
        35.0 if "fog_aware" in metering or "fog-aware" in metering else None
    )
    depth_decay_raw = config.get("depth_meter_decay_m", default_depth_decay)
    if depth_decay_raw is not None:
        depth_decay = max(float(sample_value(depth_decay_raw, rng)), 1e-6)
        min_weight = float(
            np.clip(
                _sample_float(config, "min_meter_weight", 0.05, rng),
                0.0,
                1.0,
            )
        )
        depth = _context_depth_map(context, shape)
        if depth is not None:
            depth_factor = np.exp(-np.maximum(depth, 0.0) / depth_decay)
            weights *= np.maximum(min_weight, depth_factor).astype(np.float32)

    default_fog_suppression = (
        0.60 if "fog_aware" in metering or "fog-aware" in metering else 0.0
    )
    fog_suppression = float(
        np.clip(
            _sample_float(
                config, "fog_meter_suppression", default_fog_suppression, rng
            ),
            0.0,
            1.0,
        )
    )
    if fog_suppression > 0.0:
        opacity = _context_fog_opacity(context, shape)
        if opacity is not None:
            weights *= 1.0 - fog_suppression * np.clip(opacity, 0.0, 1.0)

    default_sky_suppression = (
        0.85
        if "sky_aware" in metering
        or "sky-aware" in metering
        or "fog_aware" in metering
        or "fog-aware" in metering
        else 0.0
    )
    sky_suppression = float(
        np.clip(
            _sample_float(config, "sky_suppression", default_sky_suppression, rng),
            0.0,
            1.0,
        )
    )
    if sky_suppression > 0.0:
        sky = _context_attribute_map(context, "sky_mask", shape)
        if sky is not None:
            weights *= 1.0 - sky_suppression * np.clip(sky, 0.0, 1.0)

    finite = np.isfinite(luminance)
    weights = np.where(finite, np.clip(weights, 0.0, None), 0.0).astype(
        np.float32,
        copy=False,
    )
    if float(weights.sum()) <= 1e-8:
        return None
    return weights


def _auto_exposure_weights_torch(
    luminance,
    context: CaptureContext,
    config: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    center_weighted: bool,
):
    shape = tuple(luminance.shape)
    metering = str(config.get("metering", "center_weighted")).strip().lower()
    weights = (
        _auto_exposure_center_weight_map_torch(
            shape,
            config,
            rng,
            luminance.device,
            luminance.dtype,
        )
        if center_weighted
        else torch.ones(shape, device=luminance.device, dtype=luminance.dtype)
    )

    default_depth_decay = (
        35.0 if "fog_aware" in metering or "fog-aware" in metering else None
    )
    depth_decay_raw = config.get("depth_meter_decay_m", default_depth_decay)
    if depth_decay_raw is not None:
        depth_decay = max(float(sample_value(depth_decay_raw, rng)), 1e-6)
        min_weight = float(
            np.clip(
                _sample_float(config, "min_meter_weight", 0.05, rng),
                0.0,
                1.0,
            )
        )
        depth = _context_depth_map_torch(
            context,
            shape,
            luminance.device,
            luminance.dtype,
        )
        if depth is not None:
            depth_factor = torch.exp(-torch.clamp(depth, min=0.0) / depth_decay)
            weights = weights * torch.maximum(
                torch.full_like(depth_factor, min_weight),
                depth_factor,
            )

    default_fog_suppression = (
        0.60 if "fog_aware" in metering or "fog-aware" in metering else 0.0
    )
    fog_suppression = float(
        np.clip(
            _sample_float(
                config, "fog_meter_suppression", default_fog_suppression, rng
            ),
            0.0,
            1.0,
        )
    )
    if fog_suppression > 0.0:
        opacity = _context_fog_opacity_torch(
            context,
            shape,
            luminance.device,
            luminance.dtype,
        )
        if opacity is not None:
            weights = weights * (1.0 - fog_suppression * torch.clamp(opacity, 0.0, 1.0))

    default_sky_suppression = (
        0.85
        if "sky_aware" in metering
        or "sky-aware" in metering
        or "fog_aware" in metering
        or "fog-aware" in metering
        else 0.0
    )
    sky_suppression = float(
        np.clip(
            _sample_float(config, "sky_suppression", default_sky_suppression, rng),
            0.0,
            1.0,
        )
    )
    if sky_suppression > 0.0:
        sky = _context_attribute_map_torch(
            context,
            "sky_mask",
            shape,
            luminance.device,
            luminance.dtype,
        )
        if sky is not None:
            weights = weights * (1.0 - sky_suppression * torch.clamp(sky, 0.0, 1.0))

    finite = torch.isfinite(luminance)
    weights = torch.where(
        finite, torch.clamp(weights, min=0.0), torch.zeros_like(weights)
    )
    if float(torch.sum(weights).detach().cpu()) <= 1e-8:
        return None
    return weights


def _weighted_mean_np(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not finite.any():
        return 0.18
    vals = np.clip(values[finite], 0.0, None)
    w = weights[finite].astype(np.float64, copy=False)
    return float(np.sum(vals * w) / max(float(np.sum(w)), 1e-8))


def _weighted_percentile_np(
    values: np.ndarray,
    weights: np.ndarray,
    percentile: float,
) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not finite.any():
        return 0.18
    vals = np.clip(values[finite], 0.0, None).astype(np.float32, copy=False)
    w = weights[finite].astype(np.float64, copy=False)
    order = np.argsort(vals)
    vals = vals[order]
    w = w[order]
    cumulative = np.cumsum(w)
    total = float(cumulative[-1])
    if total <= 1e-8:
        return 0.18
    cutoff = np.clip(percentile, 0.0, 100.0) / 100.0 * total
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(vals[min(index, vals.shape[0] - 1)])


def _weighted_mean_torch(values, weights) -> float:
    finite = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0.0)
    if not bool(torch.any(finite).detach().cpu()):
        return 0.18
    vals = torch.clamp(values[finite], min=0.0)
    w = weights[finite]
    return float(
        (torch.sum(vals * w) / torch.clamp(torch.sum(w), min=1e-8)).detach().cpu()
    )


def _weighted_percentile_torch(values, weights, percentile: float) -> float:
    finite = torch.isfinite(values) & torch.isfinite(weights) & (weights > 0.0)
    if not bool(torch.any(finite).detach().cpu()):
        return 0.18
    vals = torch.clamp(values[finite], min=0.0)
    w = weights[finite]
    sorted_vals, order = torch.sort(vals)
    sorted_weights = w[order]
    cumulative = torch.cumsum(sorted_weights, dim=0)
    total = torch.clamp(cumulative[-1], min=1e-8)
    cutoff = float(np.clip(percentile, 0.0, 100.0) / 100.0) * total
    index = int(torch.searchsorted(cumulative, cutoff).detach().cpu())
    index = min(index, int(sorted_vals.numel()) - 1)
    return float(sorted_vals[index].detach().cpu())


def _resolve_electron_capacity(
    shape: tuple[int, int],
    pattern: str,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    legacy = config.get("shot_noise_electrons")
    if legacy is not None:
        electrons = float(sample_value(legacy, rng))
        if electrons <= 0.0:
            return np.zeros(shape, dtype=np.float32)
        return np.full(shape, electrons, dtype=np.float32)

    iso = max(_sample_float(config, "iso", 100.0, rng), 1e-6)
    base_iso = max(_sample_float(config, "base_iso", 100.0, rng), 1e-6)
    iso_gain = max(iso / base_iso, 1e-6)
    full_well = _sample_triplet(
        config.get("full_well_electrons", [12000.0, 12000.0, 12000.0]),
        rng,
    )
    full_well_map = _mosaic_channel_values(shape, pattern, full_well)
    capacity = full_well_map / iso_gain
    return np.where(capacity > 0.0, np.maximum(capacity, 1.0), 0.0).astype(
        np.float32,
        copy=False,
    )


def _resolve_electron_capacity_torch(
    shape: tuple[int, int],
    pattern: str,
    config: Mapping[str, Any],
    rng: np.random.Generator,
    device,
    dtype,
):
    legacy = config.get("shot_noise_electrons")
    if legacy is not None:
        electrons = float(sample_value(legacy, rng))
        if electrons <= 0.0:
            return torch.zeros(shape, device=device, dtype=dtype)
        return torch.full(shape, electrons, device=device, dtype=dtype)

    iso = max(_sample_float(config, "iso", 100.0, rng), 1e-6)
    base_iso = max(_sample_float(config, "base_iso", 100.0, rng), 1e-6)
    iso_gain = max(iso / base_iso, 1e-6)
    full_well = _sample_triplet(
        config.get("full_well_electrons", [12000.0, 12000.0, 12000.0]),
        rng,
    )
    full_well_map = _mosaic_channel_values_torch(
        shape,
        pattern,
        full_well,
        device,
        dtype,
    )
    capacity = full_well_map / iso_gain
    return torch.where(
        capacity > 0.0,
        torch.maximum(capacity, torch.ones_like(capacity)),
        torch.zeros_like(capacity),
    )


def _apply_sensor_noise_adjustment(
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> Mapping[str, Any]:
    cfg = _config_block(config, "noise_adjustment")
    if not cfg or not _bool_value(cfg.get("enabled", True)):
        return config

    level = max(
        _sample_float_from_keys(
            cfg,
            ("level", "amount", "factor", "noise_level"),
            1.0,
            rng,
        ),
        0.0,
    )
    bias = float(
        np.clip(
            _sample_float_from_keys(
                cfg,
                ("static_chroma_bias", "character_bias", "chroma_bias"),
                0.0,
                rng,
            ),
            -1.0,
            1.0,
        )
    )
    groups = _config_block(cfg, "groups")
    limits = _config_block(cfg, "limits")

    min_factor = max(_sample_float(limits, "min_factor", 0.0, rng), 0.0)
    max_factor = max(_sample_float(limits, "max_factor", 8.0, rng), min_factor)

    def group_factor(name: str, base: float) -> float:
        factor = base * max(_sample_float(groups, name, 1.0, rng), 0.0)
        return float(np.clip(factor, min_factor, max_factor))

    read_factor = group_factor("read", level * math.pow(2.0, 0.35 * bias))
    static_factor = group_factor("static", level * math.pow(2.0, -bias))
    banding_factor = group_factor("banding", static_factor)
    bad_pixel_factor = group_factor("bad_pixels", static_factor)
    chroma_factor = group_factor("chroma", level * math.pow(2.0, bias))
    shadow_luma_factor = group_factor("shadow_luma", level)
    modulation_factor = group_factor("modulation", 1.0 + (level - 1.0) * 0.75)

    if (
        abs(read_factor - 1.0) < 1e-9
        and abs(static_factor - 1.0) < 1e-9
        and abs(banding_factor - 1.0) < 1e-9
        and abs(bad_pixel_factor - 1.0) < 1e-9
        and abs(chroma_factor - 1.0) < 1e-9
        and abs(shadow_luma_factor - 1.0) < 1e-9
        and abs(modulation_factor - 1.0) < 1e-9
    ):
        return config

    adjusted = deepcopy(dict(config))

    _scale_config_path(adjusted, ("read_noise_electrons",), read_factor, min_value=0.0)
    _scale_config_path(adjusted, ("read_noise_sigma",), read_factor, min_value=0.0)

    _scale_config_path(adjusted, ("fixed_pattern_sigma",), static_factor, min_value=0.0)
    _scale_config_path(adjusted, ("row_noise_sigma",), banding_factor, min_value=0.0)
    _scale_config_path(adjusted, ("column_noise_sigma",), banding_factor, min_value=0.0)
    _scale_config_path(
        adjusted,
        ("banding_modulation",),
        banding_factor,
        min_value=0.0,
        max_value=1.0,
    )

    max_bad_pixel_probability = _sample_float(
        limits,
        "max_bad_pixel_probability",
        0.05,
        rng,
    )
    _scale_config_path(
        adjusted,
        ("hot_pixel_probability",),
        bad_pixel_factor,
        min_value=0.0,
        max_value=max_bad_pixel_probability,
    )
    _scale_config_path(
        adjusted,
        ("dead_pixel_probability",),
        bad_pixel_factor,
        min_value=0.0,
        max_value=max_bad_pixel_probability,
    )

    for key in ("dark_gain", "depth_gain", "fog_gain"):
        _scale_config_path(
            adjusted,
            ("noise_modulation", key),
            modulation_factor,
            min_value=0.0,
        )
    _scale_config_path(
        adjusted,
        ("noise_modulation", "max_gain"),
        modulation_factor,
        anchor=1.0,
        min_value=1.0,
    )

    _scale_config_path(
        adjusted,
        ("shadow_recovery_noise", "luma_sigma"),
        shadow_luma_factor,
        min_value=0.0,
    )
    _scale_config_path(
        adjusted,
        ("shadow_recovery_noise", "chroma_sigma"),
        chroma_factor,
        min_value=0.0,
    )
    _scale_config_path(
        adjusted,
        ("shadow_recovery_noise", "blotch_sigma"),
        chroma_factor,
        min_value=0.0,
    )
    _scale_config_path(
        adjusted,
        ("shadow_recovery_noise", "red_chroma_gain"),
        chroma_factor,
        anchor=1.0,
        min_value=0.0,
    )
    _scale_config_path(
        adjusted,
        ("shadow_recovery_noise", "blue_chroma_gain"),
        chroma_factor,
        anchor=1.0,
        min_value=0.0,
    )
    _scale_config_path(
        adjusted,
        ("shadow_recovery_noise", "chroma_axis_correlation"),
        chroma_factor,
        min_value=-0.95,
        max_value=0.95,
    )
    return adjusted


def _sample_float_from_keys(
    config: Mapping[str, Any],
    keys: tuple[str, ...],
    default: float,
    rng: np.random.Generator,
) -> float:
    for key in keys:
        if config.get(key) is not None:
            return _sample_float(config, key, default, rng)
    return float(default)


def _scale_config_path(
    config: dict[str, Any],
    path: tuple[str, ...],
    factor: float,
    *,
    anchor: float = 0.0,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    parent: dict[str, Any] = config
    for key in path[:-1]:
        value = parent.get(key)
        if not isinstance(value, dict):
            return
        parent = value
    key = path[-1]
    if key not in parent or parent[key] is None:
        return
    parent[key] = _scale_numeric_spec(
        parent[key],
        factor,
        anchor=anchor,
        min_value=min_value,
        max_value=max_value,
    )


def _scale_numeric_spec(
    spec: Any,
    factor: float,
    *,
    anchor: float = 0.0,
    min_value: float | None = None,
    max_value: float | None = None,
) -> Any:
    if isinstance(spec, bool) or spec is None:
        return spec
    if isinstance(spec, (int, float)):
        return _clamp_scaled_number(
            anchor + (float(spec) - anchor) * factor,
            min_value,
            max_value,
        )
    if isinstance(spec, list):
        return [
            _scale_numeric_spec(
                value,
                factor,
                anchor=anchor,
                min_value=min_value,
                max_value=max_value,
            )
            for value in spec
        ]
    if isinstance(spec, tuple):
        return tuple(
            _scale_numeric_spec(
                value,
                factor,
                anchor=anchor,
                min_value=min_value,
                max_value=max_value,
            )
            for value in spec
        )
    if not isinstance(spec, dict):
        return spec

    scaled = dict(spec)
    dist = scaled.get("dist")
    if dist is None:
        if "value" in scaled:
            scaled["value"] = _scale_numeric_spec(
                scaled["value"],
                factor,
                anchor=anchor,
                min_value=min_value,
                max_value=max_value,
            )
        return scaled

    if dist == "constant":
        scaled["value"] = _scale_numeric_spec(
            scaled.get("value", 0.0),
            factor,
            anchor=anchor,
            min_value=min_value,
            max_value=max_value,
        )
    elif dist == "uniform":
        scaled["min"] = _scale_numeric_spec(
            scaled["min"],
            factor,
            anchor=anchor,
            min_value=min_value,
            max_value=max_value,
        )
        scaled["max"] = _scale_numeric_spec(
            scaled["max"],
            factor,
            anchor=anchor,
            min_value=min_value,
            max_value=max_value,
        )
        if scaled["min"] > scaled["max"]:
            scaled["min"], scaled["max"] = scaled["max"], scaled["min"]
    elif dist == "normal":
        scaled["mean"] = _scale_numeric_spec(
            scaled["mean"],
            factor,
            anchor=anchor,
            min_value=min_value,
            max_value=max_value,
        )
        scaled["std"] = max(float(scaled.get("std", 0.0)) * abs(factor), 0.0)
        for key in ("min", "max"):
            if key in scaled and scaled[key] is not None:
                scaled[key] = _scale_numeric_spec(
                    scaled[key],
                    factor,
                    anchor=anchor,
                    min_value=min_value,
                    max_value=max_value,
                )
        if scaled.get("min") is not None and scaled.get("max") is not None:
            if scaled["min"] > scaled["max"]:
                scaled["min"], scaled["max"] = scaled["max"], scaled["min"]
    elif dist == "lognormal":
        if anchor == 0.0 and factor > 0.0:
            scaled["mean"] = float(scaled.get("mean", 0.0)) + math.log(factor)
        for key in ("min", "max"):
            if key in scaled and scaled[key] is not None:
                scaled[key] = _scale_numeric_spec(
                    scaled[key],
                    factor,
                    anchor=anchor,
                    min_value=min_value,
                    max_value=max_value,
                )
        if scaled.get("min") is not None and scaled.get("max") is not None:
            if scaled["min"] > scaled["max"]:
                scaled["min"], scaled["max"] = scaled["max"], scaled["min"]
    elif dist == "choice":
        scaled["values"] = [
            _scale_numeric_spec(
                value,
                factor,
                anchor=anchor,
                min_value=min_value,
                max_value=max_value,
            )
            for value in scaled.get("values", [])
        ]
    return scaled


def _clamp_scaled_number(
    value: float,
    min_value: float | None,
    max_value: float | None,
) -> float:
    if min_value is not None:
        value = max(float(min_value), value)
    if max_value is not None:
        value = min(float(max_value), value)
    return float(value)


def _sensor_noise_modulation(
    signal: np.ndarray,
    context: CaptureContext,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    cfg = _config_block(config, "noise_modulation")
    if not _bool_value(cfg.get("enabled", False)):
        return np.ones(signal.shape, dtype=np.float32)

    gamma = max(_sample_float(cfg, "gamma", 1.0, rng), 1e-6)
    modulation = np.ones(signal.shape, dtype=np.float32)

    dark_gain = _sample_float(cfg, "dark_gain", 0.0, rng)
    if dark_gain != 0.0:
        dark = np.power(1.0 - np.clip(signal, 0.0, 1.0), gamma)
        modulation += dark_gain * dark.astype(np.float32, copy=False)

    depth_gain = _sample_float(cfg, "depth_gain", 0.0, rng)
    if depth_gain != 0.0:
        depth = _context_depth_map(context, signal.shape)
        if depth is not None:
            modulation += depth_gain * _normalize_context_map(depth)

    fog_gain = _sample_float(cfg, "fog_gain", 0.0, rng)
    if fog_gain != 0.0:
        opacity = _context_fog_opacity(context, signal.shape)
        if opacity is not None:
            modulation += fog_gain * np.power(np.clip(opacity, 0.0, 1.0), gamma)

    black_gate = _black_suppression_gate_np(signal, cfg, rng)
    if black_gate is not None:
        modulation = 1.0 + (modulation - 1.0) * black_gate

    smooth_sigma = _sample_float(cfg, "smooth_sigma", 0.0, rng)
    if smooth_sigma > 1e-4:
        modulation = _gaussian_blur_np(modulation, smooth_sigma)

    max_gain = max(_sample_float(cfg, "max_gain", 3.0, rng), 1.0)
    return np.clip(modulation, 1.0, max_gain).astype(np.float32, copy=False)


def _sensor_noise_modulation_torch(
    signal,
    context: CaptureContext,
    config: Mapping[str, Any],
    rng: np.random.Generator,
):
    cfg = _config_block(config, "noise_modulation")
    if not _bool_value(cfg.get("enabled", False)):
        return torch.ones_like(signal)

    gamma = max(_sample_float(cfg, "gamma", 1.0, rng), 1e-6)
    modulation = torch.ones_like(signal)

    dark_gain = _sample_float(cfg, "dark_gain", 0.0, rng)
    if dark_gain != 0.0:
        dark = torch.pow(1.0 - torch.clamp(signal, 0.0, 1.0), gamma)
        modulation = modulation + float(dark_gain) * dark

    depth_gain = _sample_float(cfg, "depth_gain", 0.0, rng)
    if depth_gain != 0.0:
        depth = _context_depth_map_torch(
            context,
            tuple(signal.shape),
            signal.device,
            signal.dtype,
        )
        if depth is not None:
            modulation = modulation + float(depth_gain) * _normalize_context_map_torch(
                depth
            )

    fog_gain = _sample_float(cfg, "fog_gain", 0.0, rng)
    if fog_gain != 0.0:
        opacity = _context_fog_opacity_torch(
            context,
            tuple(signal.shape),
            signal.device,
            signal.dtype,
        )
        if opacity is not None:
            modulation = modulation + float(fog_gain) * torch.pow(
                torch.clamp(opacity, 0.0, 1.0),
                gamma,
            )

    black_gate = _black_suppression_gate_torch(signal, cfg, rng)
    if black_gate is not None:
        modulation = 1.0 + (modulation - 1.0) * black_gate

    smooth_sigma = _sample_float(cfg, "smooth_sigma", 0.0, rng)
    if smooth_sigma > 1e-4:
        modulation = _gaussian_blur_torch(modulation, smooth_sigma)

    max_gain = max(_sample_float(cfg, "max_gain", 3.0, rng), 1.0)
    return torch.clamp(modulation, 1.0, max_gain)


def _apply_shadow_recovery_noise_np(
    image: np.ndarray,
    pre_exposure_luminance: np.ndarray,
    context: CaptureContext,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    cfg = _config_block(config, "shadow_recovery_noise")
    if not _block_enabled(cfg, rng):
        return image

    weight = _shadow_recovery_weight_np(pre_exposure_luminance, context, cfg, rng)
    if not np.any(weight > 1e-6):
        return image

    strength = max(_sample_float(cfg, "strength", 1.0, rng), 0.0)
    if strength <= 0.0:
        return image

    out = image.astype(np.float32, copy=True)
    luma_sigma = max(_sample_float(cfg, "luma_sigma", 0.0, rng), 0.0) * strength
    if luma_sigma > 0.0:
        luma_noise = rng.normal(0.0, luma_sigma, image.shape[:2]).astype(np.float32)
        out += luma_noise[..., None] * weight[..., None]

    chroma_sigma = max(_sample_float(cfg, "chroma_sigma", 0.0, rng), 0.0) * strength
    if chroma_sigma > 0.0:
        chroma_preservation = float(
            np.clip(
                _sample_float(cfg, "chroma_luminance_preservation", 1.0, rng),
                0.0,
                1.0,
            )
        )
        chroma_noise = _shadow_chroma_noise_np(
            image.shape,
            chroma_sigma,
            cfg,
            rng,
            chroma_preservation,
        )
        out += chroma_noise * weight[..., None]

    blotch_sigma = max(_sample_float(cfg, "blotch_sigma", 0.0, rng), 0.0) * strength
    if blotch_sigma > 0.0:
        blotch = _low_frequency_field(image.shape[0], image.shape[1], rng) - 0.5
        out += blotch[..., None] * (2.0 * blotch_sigma) * weight[..., None]

    return _clip01(out)


def _apply_shadow_recovery_noise_torch(
    image,
    pre_exposure_luminance,
    context: CaptureContext,
    config: Mapping[str, Any],
    rng: np.random.Generator,
):
    cfg = _config_block(config, "shadow_recovery_noise")
    if not _block_enabled(cfg, rng):
        return image

    weight = _shadow_recovery_weight_torch(
        pre_exposure_luminance,
        context,
        cfg,
        rng,
    )
    if not bool(torch.any(weight > 1e-6).detach().cpu()):
        return image

    strength = max(_sample_float(cfg, "strength", 1.0, rng), 0.0)
    if strength <= 0.0:
        return image

    out = image.clone()
    luma_sigma = max(_sample_float(cfg, "luma_sigma", 0.0, rng), 0.0) * strength
    if luma_sigma > 0.0:
        luma_noise = _randn_torch(
            image.shape[:2],
            rng,
            image.device,
            image.dtype,
        ) * float(luma_sigma)
        out = out + luma_noise[..., None] * weight[..., None]

    chroma_sigma = max(_sample_float(cfg, "chroma_sigma", 0.0, rng), 0.0) * strength
    if chroma_sigma > 0.0:
        chroma_preservation = float(
            np.clip(
                _sample_float(cfg, "chroma_luminance_preservation", 1.0, rng),
                0.0,
                1.0,
            )
        )
        chroma_noise = _shadow_chroma_noise_torch(
            tuple(image.shape),
            chroma_sigma,
            cfg,
            rng,
            chroma_preservation,
            image.device,
            image.dtype,
        )
        out = out + chroma_noise * weight[..., None]

    blotch_sigma = max(_sample_float(cfg, "blotch_sigma", 0.0, rng), 0.0) * strength
    if blotch_sigma > 0.0:
        blotch = (
            _low_frequency_field_torch(
                int(image.shape[0]),
                int(image.shape[1]),
                rng,
                image.device,
                image.dtype,
            )
            - 0.5
        )
        out = out + blotch[..., None] * (2.0 * blotch_sigma) * weight[..., None]

    return _clip01_torch(out)


def _shadow_recovery_weight_np(
    pre_exposure_luminance: np.ndarray,
    context: CaptureContext,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    luma = np.clip(pre_exposure_luminance, 0.0, None).astype(np.float32, copy=False)
    threshold = max(_sample_float(config, "luminance_threshold", 0.18, rng), 1e-6)
    softness = max(_sample_float(config, "luminance_softness", 0.08, rng), 1e-6)
    shadow = np.clip((threshold + softness - luma) / softness, 0.0, 1.0)
    gamma = max(_sample_float(config, "gamma", 1.4, rng), 1e-6)
    weight = np.power(shadow, gamma).astype(np.float32, copy=False)

    fog_weight = _sample_float(config, "fog_weight", 0.0, rng)
    if fog_weight != 0.0:
        opacity = _context_fog_opacity(context, luma.shape)
        if opacity is not None:
            weight += fog_weight * np.clip(opacity, 0.0, 1.0)

    depth_weight = _sample_float(config, "depth_weight", 0.0, rng)
    if depth_weight != 0.0:
        depth = _context_depth_map(context, luma.shape)
        if depth is not None:
            weight += depth_weight * _normalize_context_map(depth)

    black_gate = _black_suppression_gate_np(luma, config, rng)
    if black_gate is not None:
        weight *= black_gate

    smooth_sigma = _sample_float(config, "smooth_sigma", 1.0, rng)
    if smooth_sigma > 1e-4:
        weight = _gaussian_blur_np(weight, smooth_sigma)

    max_weight = max(_sample_float(config, "max_weight", 1.0, rng), 0.0)
    return np.clip(weight, 0.0, max_weight).astype(np.float32, copy=False)


def _shadow_recovery_weight_torch(
    pre_exposure_luminance,
    context: CaptureContext,
    config: Mapping[str, Any],
    rng: np.random.Generator,
):
    luma = torch.clamp(pre_exposure_luminance, min=0.0)
    threshold = max(_sample_float(config, "luminance_threshold", 0.18, rng), 1e-6)
    softness = max(_sample_float(config, "luminance_softness", 0.08, rng), 1e-6)
    shadow = torch.clamp((threshold + softness - luma) / softness, 0.0, 1.0)
    gamma = max(_sample_float(config, "gamma", 1.4, rng), 1e-6)
    weight = torch.pow(shadow, gamma)

    fog_weight = _sample_float(config, "fog_weight", 0.0, rng)
    if fog_weight != 0.0:
        opacity = _context_fog_opacity_torch(
            context,
            tuple(luma.shape),
            luma.device,
            luma.dtype,
        )
        if opacity is not None:
            weight = weight + float(fog_weight) * torch.clamp(opacity, 0.0, 1.0)

    depth_weight = _sample_float(config, "depth_weight", 0.0, rng)
    if depth_weight != 0.0:
        depth = _context_depth_map_torch(
            context,
            tuple(luma.shape),
            luma.device,
            luma.dtype,
        )
        if depth is not None:
            weight = weight + float(depth_weight) * _normalize_context_map_torch(depth)

    black_gate = _black_suppression_gate_torch(luma, config, rng)
    if black_gate is not None:
        weight = weight * black_gate

    smooth_sigma = _sample_float(config, "smooth_sigma", 1.0, rng)
    if smooth_sigma > 1e-4:
        weight = _gaussian_blur_torch(weight, smooth_sigma)

    max_weight = max(_sample_float(config, "max_weight", 1.0, rng), 0.0)
    return torch.clamp(weight, 0.0, max_weight)


def _shadow_chroma_noise_np(
    shape: tuple[int, int, int],
    sigma: float,
    config: Mapping[str, Any],
    rng: np.random.Generator,
    luminance_preservation: float,
) -> np.ndarray:
    raw = rng.normal(0.0, sigma, shape).astype(np.float32)
    if luminance_preservation <= 0.0:
        return raw

    mode = str(config.get("chroma_mode", "balanced")).strip().lower().replace("-", "_")
    if mode in {"balanced", "balanced_luminance", "balanced_luminance_preserving"}:
        preserved = _balanced_luminance_preserving_chroma_noise_np(
            shape,
            sigma,
            config,
            rng,
        )
    else:
        weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        chroma_luma = np.sum(raw * weights.reshape(1, 1, 3), axis=-1)
        preserved = raw - chroma_luma[..., None]

    if luminance_preservation >= 1.0:
        return preserved
    return (
        (1.0 - luminance_preservation) * raw + luminance_preservation * preserved
    ).astype(np.float32, copy=False)


def _balanced_luminance_preserving_chroma_noise_np(
    shape: tuple[int, int, int],
    sigma: float,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    if len(shape) != 3 or shape[2] != 3:
        return rng.normal(0.0, sigma, shape).astype(np.float32)

    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    red_axis = np.array([1.0, -weights[0] / weights[1], 0.0], dtype=np.float32)
    blue_axis = np.array([0.0, -weights[2] / weights[1], 1.0], dtype=np.float32)

    red_gain = max(_sample_float(config, "red_chroma_gain", 1.0, rng), 0.0)
    blue_gain = max(_sample_float(config, "blue_chroma_gain", 1.0, rng), 0.0)
    correlation = float(
        np.clip(
            _sample_float(config, "chroma_axis_correlation", 0.0, rng),
            -0.95,
            0.95,
        )
    )
    red_unit = rng.normal(0.0, 1.0, shape[:2]).astype(np.float32)
    blue_unit = rng.normal(0.0, 1.0, shape[:2]).astype(np.float32)
    if abs(correlation) > 1e-6:
        blue_unit = (
            correlation * red_unit
            + np.sqrt(max(1.0 - correlation * correlation, 0.0)) * blue_unit
        ).astype(np.float32, copy=False)

    red_noise = red_unit * (sigma * red_gain)
    blue_noise = blue_unit * (sigma * blue_gain)
    noise = (
        red_noise[..., None] * red_axis.reshape(1, 1, 3)
        + blue_noise[..., None] * blue_axis.reshape(1, 1, 3)
    ).astype(np.float32, copy=False)

    spatial_sigma = _sample_float(config, "chroma_spatial_sigma", 0.0, rng)
    if spatial_sigma > 1e-4:
        fine_fraction = float(
            np.clip(
                _sample_float(config, "chroma_fine_fraction", 1.0, rng),
                0.0,
                1.0,
            )
        )
        blurred = _gaussian_blur_np(noise, spatial_sigma)
        noise = fine_fraction * noise + (1.0 - fine_fraction) * blurred

    return noise.astype(np.float32, copy=False)


def _shadow_chroma_noise_torch(
    shape: tuple[int, int, int],
    sigma: float,
    config: Mapping[str, Any],
    rng: np.random.Generator,
    luminance_preservation: float,
    device,
    dtype,
):
    raw = _randn_torch(shape, rng, device, dtype) * float(sigma)
    if luminance_preservation <= 0.0:
        return raw

    mode = str(config.get("chroma_mode", "balanced")).strip().lower().replace("-", "_")
    if mode in {"balanced", "balanced_luminance", "balanced_luminance_preserving"}:
        preserved = _balanced_luminance_preserving_chroma_noise_torch(
            shape,
            sigma,
            config,
            rng,
            device,
            dtype,
        )
    else:
        weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device, dtype=dtype)
        chroma_luma = torch.sum(raw * weights.view(1, 1, 3), dim=-1)
        preserved = raw - chroma_luma[..., None]

    if luminance_preservation >= 1.0:
        return preserved
    return (1.0 - luminance_preservation) * raw + luminance_preservation * preserved


def _balanced_luminance_preserving_chroma_noise_torch(
    shape: tuple[int, int, int],
    sigma: float,
    config: Mapping[str, Any],
    rng: np.random.Generator,
    device,
    dtype,
):
    if len(shape) != 3 or shape[2] != 3:
        return _randn_torch(shape, rng, device, dtype) * float(sigma)

    red_axis = torch.tensor(
        [1.0, -(0.2126 / 0.7152), 0.0],
        device=device,
        dtype=dtype,
    )
    blue_axis = torch.tensor(
        [0.0, -(0.0722 / 0.7152), 1.0],
        device=device,
        dtype=dtype,
    )

    red_gain = max(_sample_float(config, "red_chroma_gain", 1.0, rng), 0.0)
    blue_gain = max(_sample_float(config, "blue_chroma_gain", 1.0, rng), 0.0)
    correlation = float(
        np.clip(
            _sample_float(config, "chroma_axis_correlation", 0.0, rng),
            -0.95,
            0.95,
        )
    )
    red_unit = _randn_torch(shape[:2], rng, device, dtype)
    blue_unit = _randn_torch(shape[:2], rng, device, dtype)
    if abs(correlation) > 1e-6:
        blue_unit = (
            correlation * red_unit
            + math.sqrt(max(1.0 - correlation * correlation, 0.0)) * blue_unit
        )

    red_noise = red_unit * (float(sigma) * red_gain)
    blue_noise = blue_unit * (float(sigma) * blue_gain)
    noise = red_noise[..., None] * red_axis.view(1, 1, 3)
    noise = noise + blue_noise[..., None] * blue_axis.view(1, 1, 3)

    spatial_sigma = _sample_float(config, "chroma_spatial_sigma", 0.0, rng)
    if spatial_sigma > 1e-4:
        fine_fraction = float(
            np.clip(
                _sample_float(config, "chroma_fine_fraction", 1.0, rng),
                0.0,
                1.0,
            )
        )
        blurred = _gaussian_blur_torch(noise, spatial_sigma)
        noise = fine_fraction * noise + (1.0 - fine_fraction) * blurred

    return noise


def _black_suppression_gate_np(
    luminance_or_signal: np.ndarray,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> np.ndarray | None:
    floor = float(
        np.clip(_sample_float(config, "black_noise_floor", 1.0, rng), 0.0, 1.0)
    )
    if floor >= 1.0:
        return None

    threshold = max(
        _sample_float(config, "black_suppression_luminance", 0.0, rng),
        0.0,
    )
    softness = max(
        _sample_float(config, "black_suppression_softness", 0.05, rng),
        1e-6,
    )
    x = np.clip((luminance_or_signal - threshold) / softness, 0.0, 1.0)
    smooth = x * x * (3.0 - 2.0 * x)
    return (floor + (1.0 - floor) * smooth).astype(np.float32, copy=False)


def _black_suppression_gate_torch(
    luminance_or_signal,
    config: Mapping[str, Any],
    rng: np.random.Generator,
):
    floor = float(
        np.clip(_sample_float(config, "black_noise_floor", 1.0, rng), 0.0, 1.0)
    )
    if floor >= 1.0:
        return None

    threshold = max(
        _sample_float(config, "black_suppression_luminance", 0.0, rng),
        0.0,
    )
    softness = max(
        _sample_float(config, "black_suppression_softness", 0.05, rng),
        1e-6,
    )
    x = torch.clamp((luminance_or_signal - threshold) / softness, 0.0, 1.0)
    smooth = x * x * (3.0 - 2.0 * x)
    return floor + (1.0 - floor) * smooth


def _normalize_context_map(value: np.ndarray) -> np.ndarray:
    finite = np.isfinite(value)
    if not finite.any():
        return np.zeros(value.shape, dtype=np.float32)
    clipped = value.astype(np.float32, copy=True)
    clipped[~finite] = 0.0
    lo = float(np.percentile(clipped[finite], 2.0))
    hi = float(np.percentile(clipped[finite], 98.0))
    if hi <= lo:
        return np.zeros(value.shape, dtype=np.float32)
    return np.clip((clipped - lo) / (hi - lo), 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )


def _normalize_context_map_torch(value):
    finite = torch.isfinite(value)
    if not bool(torch.any(finite).detach().cpu()):
        return torch.zeros_like(value)
    clipped = torch.where(finite, value, torch.zeros_like(value))
    valid = clipped[finite]
    lo = torch.quantile(valid, 0.02)
    hi = torch.quantile(valid, 0.98)
    if not bool((hi > lo).detach().cpu()):
        return torch.zeros_like(value)
    return torch.clamp((clipped - lo) / (hi - lo), 0.0, 1.0)


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _config_block(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    raw = config.get(key, {})
    if raw is None:
        return {"enabled": False}
    if isinstance(raw, bool):
        return {"enabled": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"{key} must be a boolean or object")
    return dict(raw)


def _block_enabled(config: Mapping[str, Any], rng: np.random.Generator) -> bool:
    if not _bool_value(config.get("enabled", True)):
        return False
    probability = _sample_float(config, "probability", 1.0, rng)
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    return bool(rng.random() < probability)


def _srgb_to_linear(image: np.ndarray) -> np.ndarray:
    return srgb_to_linear(image)


def _srgb_to_linear_torch(image):
    return srgb_to_linear_torch(image)


def _linear_to_srgb(image: np.ndarray) -> np.ndarray:
    return linear_to_srgb(image)


def _linear_to_srgb_torch(image):
    return linear_to_srgb_torch(image)


def _apply_color_matrix(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.tensordot(image, matrix.T, axes=([-1], [0])).astype(
        np.float32,
        copy=False,
    )


def _apply_color_matrix_torch(image, matrix: np.ndarray):
    matrix_t = torch.as_tensor(matrix, device=image.device, dtype=image.dtype)
    return image @ matrix_t.transpose(0, 1)


def _gaussian_blur_np(image: np.ndarray, sigma: float) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 1e-4:
        return image
    kernel = _gaussian_kernel1d(sigma)
    out = _convolve1d_reflect(image, kernel, axis=1)
    out = _convolve1d_reflect(out, kernel, axis=0)
    return out.astype(np.float32, copy=False)


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= float(kernel.sum())
    return kernel.astype(np.float32)


def _gaussian_kernel1d_torch(sigma: float, *, device, dtype):
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / float(sigma)) ** 2)
    return kernel / torch.clamp(kernel.sum(), min=torch.finfo(dtype).eps)


def _gaussian_blur_torch(image, sigma: float):
    sigma = float(sigma)
    if sigma <= 1e-4:
        return image
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2-D or HWC image tensor, got {tuple(image.shape)}")

    is_2d = image.ndim == 2
    if is_2d:
        x = image.to(dtype=torch.float32).view(1, 1, image.shape[0], image.shape[1])
    else:
        if image.shape[-1] <= 0:
            return image
        x = image.to(dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    channels = int(x.shape[1])
    dtype = x.dtype
    device = x.device
    kernel = _gaussian_kernel1d_torch(sigma, device=device, dtype=dtype)
    radius = int(kernel.numel() // 2)
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    horizontal_pad = "reflect" if radius < int(x.shape[-1]) else "replicate"
    x = torch.nn.functional.pad(x, (radius, radius, 0, 0), mode=horizontal_pad)
    x = torch.nn.functional.conv2d(x, horizontal, groups=channels)
    vertical_pad = "reflect" if radius < int(x.shape[-2]) else "replicate"
    x = torch.nn.functional.pad(x, (0, 0, radius, radius), mode=vertical_pad)
    x = torch.nn.functional.conv2d(x, vertical, groups=channels)

    if is_2d:
        return x[0, 0].to(dtype=image.dtype)
    return x[0].permute(1, 2, 0).to(dtype=image.dtype)


def _smooth_random_bias(
    length: int,
    sigma: float,
    correlation_px: float,
    rng: np.random.Generator,
) -> np.ndarray:
    values = rng.normal(0.0, sigma, length).astype(np.float32)
    if correlation_px <= 1.0 or length <= 2:
        return values
    kernel = _gaussian_kernel1d(max(float(correlation_px) / 3.0, 1e-4))
    radius = len(kernel) // 2
    padded = np.pad(values, (radius, radius), mode="reflect")
    out = np.zeros_like(values, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        out += float(weight) * padded[offset : offset + length]
    original_std = float(values.std())
    smoothed_std = float(out.std())
    if smoothed_std > 1e-8 and original_std > 0.0:
        out = out * (original_std / smoothed_std)
    return out.astype(np.float32, copy=False)


def _smooth_random_bias_torch(
    length: int,
    sigma: float,
    correlation_px: float,
    rng: np.random.Generator,
    device,
    dtype,
):
    values = _randn_torch((length,), rng, device, dtype) * float(sigma)
    if correlation_px <= 1.0 or length <= 2:
        return values
    kernel = _gaussian_kernel1d_torch(
        max(float(correlation_px) / 3.0, 1e-4),
        device=device,
        dtype=dtype,
    )
    radius = int(kernel.numel() // 2)
    pad_mode = "reflect" if radius < length else "replicate"
    x = values.to(dtype=torch.float32).view(1, 1, length)
    padded = torch.nn.functional.pad(x, (radius, radius), mode=pad_mode)
    out = torch.nn.functional.conv1d(
        padded,
        kernel.to(dtype=torch.float32).view(1, 1, -1),
    )[0, 0].to(dtype=dtype)
    original_std = torch.std(values, unbiased=False)
    smoothed_std = torch.std(out, unbiased=False)
    if bool((smoothed_std > 1e-8).detach().cpu()) and bool(
        (original_std > 0.0).detach().cpu()
    ):
        out = out * (original_std / smoothed_std)
    return out


def _convolve1d_reflect(
    image: np.ndarray,
    kernel: np.ndarray,
    *,
    axis: int,
) -> np.ndarray:
    radius = len(kernel) // 2
    if radius == 0:
        return image
    pad_width = [(0, 0)] * image.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(image, pad_width, mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        slices = [slice(None)] * image.ndim
        slices[axis] = slice(offset, offset + image.shape[axis])
        out += float(weight) * padded[tuple(slices)]
    return out


def _convolve2d_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float32)
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            out += (
                float(kernel[y, x])
                * padded[
                    y : y + image.shape[0],
                    x : x + image.shape[1],
                ]
            )
    return out


def _coordinate_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    return yy, xx


def _camera_geometry(
    height: int,
    width: int,
    intrinsics: np.ndarray | None,
) -> tuple[float, float, float, float]:
    if intrinsics is not None:
        fx = max(abs(float(intrinsics[0, 0])), 1e-6)
        fy = max(abs(float(intrinsics[1, 1])), 1e-6)
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])
        return fx, fy, cx, cy
    fx = max((width - 1) / 2.0, 1e-6)
    fy = max((height - 1) / 2.0, 1e-6)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return fx, fy, cx, cy


def _context_intrinsics_torch(context: CaptureContext, device, dtype):
    if context.intrinsics is None or torch is None:
        return None
    if torch.is_tensor(context.intrinsics):
        intrinsics = context.intrinsics.to(device=device, dtype=dtype)
        if tuple(intrinsics.shape) != (3, 3):
            return None
        fx = float(intrinsics[0, 0].detach().cpu())
        fy = float(intrinsics[1, 1].detach().cpu())
        if abs(fx) < 1e-6 or abs(fy) < 1e-6:
            return None
        return intrinsics

    intrinsics_np = _context_intrinsics(context)
    if intrinsics_np is None:
        return None
    return torch.as_tensor(intrinsics_np, device=device, dtype=dtype)


def _camera_geometry_torch(
    height: int,
    width: int,
    intrinsics,
    device,
    dtype,
) -> tuple[float, float, float, float]:
    if intrinsics is not None:
        fx = max(abs(float(intrinsics[0, 0].detach().cpu())), 1e-6)
        fy = max(abs(float(intrinsics[1, 1].detach().cpu())), 1e-6)
        cx = float(intrinsics[0, 2].detach().cpu())
        cy = float(intrinsics[1, 2].detach().cpu())
        return fx, fy, cx, cy
    fx = max((width - 1) / 2.0, 1e-6)
    fy = max((height - 1) / 2.0, 1e-6)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return fx, fy, cx, cy


def _coordinate_grid_torch(height: int, width: int, device, dtype):
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return yy, xx


def _sample_bilinear_np(
    image: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.clip(x, 0.0, max(width - 1, 0))
    y = np.clip(y, 0.0, max(height - 1, 0))
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    if image.ndim == 2:
        top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
        bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
        return (top * (1.0 - wy) + bottom * wy).astype(np.float32, copy=False)

    wx3 = wx[..., None]
    wy3 = wy[..., None]
    top = image[y0, x0] * (1.0 - wx3) + image[y0, x1] * wx3
    bottom = image[y1, x0] * (1.0 - wx3) + image[y1, x1] * wx3
    return (top * (1.0 - wy3) + bottom * wy3).astype(np.float32, copy=False)


def _sample_bilinear_torch(image, y, x):
    height = int(image.shape[0])
    width = int(image.shape[1])
    x = torch.clamp(x, 0.0, float(max(width - 1, 0)))
    y = torch.clamp(y, 0.0, float(max(height - 1, 0)))
    if width > 1:
        grid_x = 2.0 * x / float(width - 1) - 1.0
    else:
        grid_x = torch.zeros_like(x)
    if height > 1:
        grid_y = 2.0 * y / float(height - 1) - 1.0
    else:
        grid_y = torch.zeros_like(y)
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

    if image.ndim == 2:
        input_t = image.to(dtype=torch.float32).view(1, 1, height, width)
        sampled = torch.nn.functional.grid_sample(
            input_t,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled[0, 0].to(dtype=image.dtype)

    input_t = image.to(dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    sampled = torch.nn.functional.grid_sample(
        input_t,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[0].permute(1, 2, 0).to(dtype=image.dtype)


def _lens_distort_np(
    image: np.ndarray,
    k1: float,
    k2: float,
    intrinsics: np.ndarray | None = None,
) -> np.ndarray:
    height, width = image.shape[:2]
    yy, xx = _coordinate_grid(height, width)
    fx, fy, cx, cy = _camera_geometry(height, width, intrinsics)
    nx = (xx - cx) / fx
    ny = (yy - cy) / fy
    r2 = nx * nx + ny * ny
    scale = 1.0 + float(k1) * r2 + float(k2) * r2 * r2
    src_x = cx + nx * scale * fx
    src_y = cy + ny * scale * fy
    return _sample_bilinear_np(image, src_y, src_x)


def _lens_distort_torch(image, k1: float, k2: float, intrinsics=None):
    height, width = int(image.shape[0]), int(image.shape[1])
    yy, xx = _coordinate_grid_torch(height, width, image.device, image.dtype)
    fx, fy, cx, cy = _camera_geometry_torch(
        height,
        width,
        intrinsics,
        image.device,
        image.dtype,
    )
    nx = (xx - cx) / fx
    ny = (yy - cy) / fy
    r2 = nx * nx + ny * ny
    scale = 1.0 + float(k1) * r2 + float(k2) * r2 * r2
    src_x = cx + nx * scale * fx
    src_y = cy + ny * scale * fy
    return _sample_bilinear_torch(image, src_y, src_x)


def _chromatic_aberration_np(
    image: np.ndarray,
    amount_px: float,
    intrinsics: np.ndarray | None = None,
) -> np.ndarray:
    height, width = image.shape[:2]
    yy, xx = _coordinate_grid(height, width)
    _, _, cx, cy = _camera_geometry(height, width, intrinsics)
    dx = xx - cx
    dy = yy - cy
    radius = np.sqrt(dx * dx + dy * dy)
    max_radius = max(float(radius.max()), 1e-6)
    ux = dx / np.maximum(radius, 1e-6)
    uy = dy / np.maximum(radius, 1e-6)
    offset = float(amount_px) * (radius / max_radius)
    out = image.copy()
    out[..., 0] = _sample_bilinear_np(image[..., 0], yy - uy * offset, xx - ux * offset)
    out[..., 2] = _sample_bilinear_np(image[..., 2], yy + uy * offset, xx + ux * offset)
    return out.astype(np.float32, copy=False)


def _chromatic_aberration_torch(image, amount_px: float, intrinsics=None):
    height, width = int(image.shape[0]), int(image.shape[1])
    yy, xx = _coordinate_grid_torch(height, width, image.device, image.dtype)
    _, _, cx, cy = _camera_geometry_torch(
        height,
        width,
        intrinsics,
        image.device,
        image.dtype,
    )
    dx = xx - cx
    dy = yy - cy
    radius = torch.sqrt(dx * dx + dy * dy)
    max_radius = torch.clamp(radius.max(), min=1e-6)
    safe_radius = torch.clamp(radius, min=1e-6)
    ux = dx / safe_radius
    uy = dy / safe_radius
    offset = float(amount_px) * (radius / max_radius)
    red = _sample_bilinear_torch(image[..., 0], yy - uy * offset, xx - ux * offset)
    blue = _sample_bilinear_torch(image[..., 2], yy + uy * offset, xx + ux * offset)
    return torch.stack((red, image[..., 1], blue), dim=-1).to(dtype=image.dtype)


def _apply_depth_chromatic_fringing_np(
    image: np.ndarray,
    config: Mapping[str, Any],
    context: CaptureContext,
    intrinsics: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    strength = _sample_float(config, "strength_px", 0.0, rng)
    if strength <= 1e-5:
        return image

    height, width = image.shape[:2]
    weight = _depth_fog_dark_weight_map(image, context, config, rng)
    if not np.any(weight > 1e-5):
        return image

    yy, xx = _coordinate_grid(height, width)
    _, _, cx, cy = _camera_geometry(height, width, intrinsics)
    dx = xx - cx
    dy = yy - cy
    radius = np.sqrt(dx * dx + dy * dy)
    max_radius = max(float(radius.max()), 1e-6)
    ux = dx / np.maximum(radius, 1e-6)
    uy = dy / np.maximum(radius, 1e-6)
    offset = strength * (radius / max_radius) * weight

    shifted = image.copy()
    shifted[..., 0] = _sample_bilinear_np(
        image[..., 0],
        yy - uy * offset,
        xx - ux * offset,
    )
    shifted[..., 2] = _sample_bilinear_np(
        image[..., 2],
        yy + uy * offset,
        xx + ux * offset,
    )
    alpha = np.clip(
        weight * _sample_float(config, "max_alpha", 0.8, rng),
        0.0,
        1.0,
    )
    return (image * (1.0 - alpha[..., None]) + shifted * alpha[..., None]).astype(
        np.float32, copy=False
    )


def _apply_depth_chromatic_fringing_torch(
    image,
    config: Mapping[str, Any],
    context: CaptureContext,
    intrinsics,
    rng: np.random.Generator,
):
    strength = _sample_float(config, "strength_px", 0.0, rng)
    if strength <= 1e-5:
        return image

    height, width = int(image.shape[0]), int(image.shape[1])
    weight = _depth_fog_dark_weight_map_torch(image, context, config, rng)
    if not bool(torch.any(weight > 1e-5).detach().cpu()):
        return image

    yy, xx = _coordinate_grid_torch(height, width, image.device, image.dtype)
    _, _, cx, cy = _camera_geometry_torch(
        height,
        width,
        intrinsics,
        image.device,
        image.dtype,
    )
    dx = xx - cx
    dy = yy - cy
    radius = torch.sqrt(dx * dx + dy * dy)
    max_radius = torch.clamp(radius.max(), min=1e-6)
    safe_radius = torch.clamp(radius, min=1e-6)
    ux = dx / safe_radius
    uy = dy / safe_radius
    offset = float(strength) * (radius / max_radius) * weight

    red = _sample_bilinear_torch(image[..., 0], yy - uy * offset, xx - ux * offset)
    blue = _sample_bilinear_torch(image[..., 2], yy + uy * offset, xx + ux * offset)
    shifted = torch.stack((red, image[..., 1], blue), dim=-1)
    alpha = torch.clamp(
        weight * _sample_float(config, "max_alpha", 0.8, rng),
        0.0,
        1.0,
    )
    return image * (1.0 - alpha[..., None]) + shifted * alpha[..., None]


def _depth_fog_dark_weight_map(
    image: np.ndarray,
    context: CaptureContext,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    shape = image.shape[:2]
    weight = np.zeros(shape, dtype=np.float32)

    depth_weight = _sample_float(config, "depth_weight", 0.0, rng)
    if depth_weight != 0.0:
        depth = _context_depth_map(context, shape)
        if depth is not None:
            weight += depth_weight * _normalize_context_map(depth)

    fog_weight = _sample_float(config, "fog_weight", 0.0, rng)
    if fog_weight != 0.0:
        opacity = _context_fog_opacity(context, shape)
        if opacity is not None:
            weight += fog_weight * np.clip(opacity, 0.0, 1.0)

    dark_weight = _sample_float(config, "dark_weight", 0.0, rng)
    if dark_weight != 0.0:
        luma = np.sum(
            image * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32),
            axis=-1,
        )
        weight += dark_weight * (1.0 - np.clip(luma, 0.0, 1.0))

    gamma = max(_sample_float(config, "gamma", 1.0, rng), 1e-6)
    weight = np.power(np.clip(weight, 0.0, 1.0), gamma)
    blur_sigma = _sample_float(config, "blur_sigma", 0.0, rng)
    if blur_sigma > 1e-4:
        weight = _gaussian_blur_np(weight, blur_sigma)
    return np.clip(weight, 0.0, 1.0).astype(np.float32, copy=False)


def _depth_fog_dark_weight_map_torch(
    image,
    context: CaptureContext,
    config: Mapping[str, Any],
    rng: np.random.Generator,
):
    shape = (int(image.shape[0]), int(image.shape[1]))
    weight = torch.zeros(shape, device=image.device, dtype=image.dtype)

    depth_weight = _sample_float(config, "depth_weight", 0.0, rng)
    if depth_weight != 0.0:
        depth = _context_depth_map_torch(context, shape, image.device, image.dtype)
        if depth is not None:
            weight = weight + float(depth_weight) * _normalize_context_map_torch(depth)

    fog_weight = _sample_float(config, "fog_weight", 0.0, rng)
    if fog_weight != 0.0:
        opacity = _context_fog_opacity_torch(context, shape, image.device, image.dtype)
        if opacity is not None:
            weight = weight + float(fog_weight) * torch.clamp(opacity, 0.0, 1.0)

    dark_weight = _sample_float(config, "dark_weight", 0.0, rng)
    if dark_weight != 0.0:
        weights = torch.tensor(
            [0.2126, 0.7152, 0.0722],
            device=image.device,
            dtype=image.dtype,
        )
        luma = torch.sum(image * weights.view(1, 1, 3), dim=-1)
        weight = weight + float(dark_weight) * (1.0 - torch.clamp(luma, 0.0, 1.0))

    gamma = max(_sample_float(config, "gamma", 1.0, rng), 1e-6)
    weight = torch.pow(torch.clamp(weight, 0.0, 1.0), gamma)
    blur_sigma = _sample_float(config, "blur_sigma", 0.0, rng)
    if blur_sigma > 1e-4:
        weight = _gaussian_blur_torch(weight, blur_sigma)
    return torch.clamp(weight, 0.0, 1.0)


def _motion_blur_np(
    image: np.ndarray, length_px: float, angle_deg: float
) -> np.ndarray:
    length = max(2, int(round(length_px)))
    yy, xx = _coordinate_grid(image.shape[0], image.shape[1])
    angle = math.radians(float(angle_deg))
    offsets = np.linspace(-(length - 1) / 2.0, (length - 1) / 2.0, length)
    out = np.zeros_like(image, dtype=np.float32)
    for offset in offsets:
        out += _sample_bilinear_np(
            image,
            yy + math.sin(angle) * offset,
            xx + math.cos(angle) * offset,
        )
    return out / float(length)


def _motion_blur_torch(image, length_px: float, angle_deg: float):
    length = max(2, int(round(length_px)))
    height, width = int(image.shape[0]), int(image.shape[1])
    yy, xx = _coordinate_grid_torch(height, width, image.device, image.dtype)
    angle = math.radians(float(angle_deg))
    offsets = torch.linspace(
        -(length - 1) / 2.0,
        (length - 1) / 2.0,
        length,
        device=image.device,
        dtype=image.dtype,
    )
    out = torch.zeros_like(image)
    sin_angle = math.sin(angle)
    cos_angle = math.cos(angle)
    for offset in offsets:
        out = out + _sample_bilinear_torch(
            image,
            yy + sin_angle * offset,
            xx + cos_angle * offset,
        )
    return out / float(length)


def _apply_bloom_np(
    image: np.ndarray,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    threshold = _sample_float(config, "threshold", 0.85, rng)
    strength = _sample_float(config, "strength", 0.0, rng)
    sigma = _sample_float(config, "sigma", 3.0, rng)
    if strength <= 1e-5:
        return image
    luminance = image.max(axis=-1, keepdims=True)
    mask = np.clip((luminance - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    glow = _gaussian_blur_np(image * mask, sigma)
    return image + strength * glow


def _apply_bloom_torch(image, config: Mapping[str, Any], rng: np.random.Generator):
    threshold = _sample_float(config, "threshold", 0.85, rng)
    strength = _sample_float(config, "strength", 0.0, rng)
    sigma = _sample_float(config, "sigma", 3.0, rng)
    if strength <= 1e-5:
        return image
    luminance = image.amax(dim=-1, keepdim=True)
    mask = torch.clamp(
        (luminance - float(threshold)) / max(1.0 - float(threshold), 1e-6),
        0.0,
        1.0,
    )
    glow = _gaussian_blur_torch(image * mask, sigma)
    return image + float(strength) * glow


def _apply_fog_coupled_glare_np(
    image: np.ndarray,
    config: Mapping[str, Any],
    context: CaptureContext,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = image.shape[:2]
    alpha = np.full(
        (height, width),
        max(_sample_float(config, "base_strength", 0.0, rng), 0.0),
        dtype=np.float32,
    )

    fog_strength = max(_sample_float(config, "fog_strength", 0.0, rng), 0.0)
    if fog_strength > 0.0:
        opacity = _context_fog_opacity(context, (height, width))
        if opacity is not None:
            alpha += fog_strength * np.clip(opacity, 0.0, 1.0)

    highlight_strength = max(
        _sample_float(config, "highlight_strength", 0.0, rng),
        0.0,
    )
    if highlight_strength > 0.0:
        threshold = _sample_float(config, "highlight_threshold", 0.72, rng)
        luma = _linear_luminance_np(image)
        highlights = np.clip((luma - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
        alpha += highlight_strength * highlights

    airlight_strength = max(
        _sample_float(config, "airlight_strength", 0.0, rng),
        0.0,
    )
    if airlight_strength > 0.0:
        airlight = (context.attributes or {}).get("airlight")
        if airlight is not None:
            alpha += airlight_strength * float(np.mean(np.clip(airlight, 0.0, 1.0)))

    smooth_sigma = _sample_float(config, "smooth_sigma", 12.0, rng)
    if smooth_sigma > 1e-4:
        alpha = _gaussian_blur_np(alpha, smooth_sigma)
    alpha = np.clip(alpha, 0.0, 0.75).astype(np.float32, copy=False)
    color = _sample_triplet(config.get("color", [0.92, 0.95, 1.0]), rng)
    return image * (1.0 - alpha[..., None]) + color.reshape(1, 1, 3) * alpha[..., None]


def _apply_fog_coupled_glare_torch(
    image,
    config: Mapping[str, Any],
    context: CaptureContext,
    rng: np.random.Generator,
):
    height, width = int(image.shape[0]), int(image.shape[1])
    alpha = torch.full(
        (height, width),
        max(_sample_float(config, "base_strength", 0.0, rng), 0.0),
        device=image.device,
        dtype=image.dtype,
    )

    fog_strength = max(_sample_float(config, "fog_strength", 0.0, rng), 0.0)
    if fog_strength > 0.0:
        opacity = _context_fog_opacity_torch(
            context,
            (height, width),
            image.device,
            image.dtype,
        )
        if opacity is not None:
            alpha = alpha + fog_strength * torch.clamp(opacity, 0.0, 1.0)

    highlight_strength = max(
        _sample_float(config, "highlight_strength", 0.0, rng),
        0.0,
    )
    if highlight_strength > 0.0:
        threshold = _sample_float(config, "highlight_threshold", 0.72, rng)
        luma = _linear_luminance_torch(image)
        highlights = torch.clamp(
            (luma - threshold) / max(1.0 - threshold, 1e-6),
            0.0,
            1.0,
        )
        alpha = alpha + highlight_strength * highlights

    airlight_strength = max(
        _sample_float(config, "airlight_strength", 0.0, rng),
        0.0,
    )
    if airlight_strength > 0.0:
        airlight = (context.attributes or {}).get("airlight")
        if airlight is not None:
            airlight_t = torch.as_tensor(
                airlight, device=image.device, dtype=image.dtype
            )
            alpha = alpha + airlight_strength * torch.mean(
                torch.clamp(airlight_t, 0.0, 1.0)
            )

    smooth_sigma = _sample_float(config, "smooth_sigma", 12.0, rng)
    if smooth_sigma > 1e-4:
        alpha = _gaussian_blur_torch(alpha, smooth_sigma)
    alpha = torch.clamp(alpha, 0.0, 0.75)
    color = torch.as_tensor(
        _sample_triplet(config.get("color", [0.92, 0.95, 1.0]), rng),
        device=image.device,
        dtype=image.dtype,
    )
    return image * (1.0 - alpha[..., None]) + color.view(1, 1, 3) * alpha[..., None]


def _low_frequency_field(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    largest = max(height, width)
    scales = [max(4, int(largest * fraction)) for fraction in (0.75, 0.38, 0.19)]
    return perlin_fbm(height, width, scales, rng)


def _torch_generator_from_rng(rng: np.random.Generator, device):
    seed = int(rng.integers(0, np.iinfo(np.int64).max))
    try:
        generator = torch.Generator(device=device)
    except Exception:  # pragma: no cover - backend-specific fallback
        generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _rand_torch(shape, rng: np.random.Generator, device, dtype):
    return torch.rand(
        tuple(shape),
        generator=_torch_generator_from_rng(rng, device),
        device=device,
        dtype=dtype,
    )


def _randn_torch(shape, rng: np.random.Generator, device, dtype):
    return torch.randn(
        tuple(shape),
        generator=_torch_generator_from_rng(rng, device),
        device=device,
        dtype=dtype,
    )


def _poisson_torch(values, rng: np.random.Generator):
    return torch.poisson(
        values,
        generator=_torch_generator_from_rng(rng, values.device),
    ).to(dtype=values.dtype)


def _positive_torch_or_float(value: Any) -> bool:
    if torch is not None and torch.is_tensor(value):
        return bool(torch.any(value > 0.0).detach().cpu())
    return float(value) > 0.0


def _low_frequency_field_torch(
    height: int,
    width: int,
    rng: np.random.Generator,
    device,
    dtype,
):
    largest = max(height, width)
    scales = [max(4, int(largest * fraction)) for fraction in (0.75, 0.38, 0.19)]
    generator = _torch_generator_from_rng(rng, device)
    field = perlin_fbm_torch(height, width, scales, generator, device)
    return field.to(device=device, dtype=dtype)


def _vignette_mask(
    height: int,
    width: int,
    strength: float,
    radius: float,
    intrinsics: np.ndarray | None = None,
) -> np.ndarray:
    yy, xx = _coordinate_grid(height, width)
    fx, fy, cx, cy = _camera_geometry(height, width, intrinsics)
    nx = (xx - cx) / fx
    ny = (yy - cy) / fy
    r = np.sqrt(nx * nx + ny * ny) / max(float(radius), 1e-6)
    mask = 1.0 - float(strength) * np.clip(r * r, 0.0, 1.0)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def _vignette_mask_torch(
    height: int,
    width: int,
    strength: float,
    radius: float,
    intrinsics,
    device,
    dtype,
):
    yy, xx = _coordinate_grid_torch(height, width, device, dtype)
    fx, fy, cx, cy = _camera_geometry_torch(height, width, intrinsics, device, dtype)
    nx = (xx - cx) / fx
    ny = (yy - cy) / fy
    r = torch.sqrt(nx * nx + ny * ny) / max(float(radius), 1e-6)
    mask = 1.0 - float(strength) * torch.clamp(r * r, 0.0, 1.0)
    return torch.clamp(mask, 0.0, 1.0)


def _apply_windshield_haze_np(
    image: np.ndarray,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    strength = _sample_float(config, "strength", 0.0, rng)
    if strength <= 1e-5:
        return image
    blur_sigma = _sample_float(config, "blur_sigma", 8.0, rng)
    color = _sample_triplet(config.get("color", [0.82, 0.86, 0.88]), rng)
    field = _low_frequency_field(image.shape[0], image.shape[1], rng)
    alpha = strength * (0.45 + 0.55 * field)
    blurred = _gaussian_blur_np(image, blur_sigma)
    veil = 0.7 * blurred + 0.3 * color.reshape(1, 1, 3)
    return image * (1.0 - alpha[..., None]) + veil * alpha[..., None]


def _apply_windshield_haze_torch(image, config: Mapping[str, Any], rng):
    strength = _sample_float(config, "strength", 0.0, rng)
    if strength <= 1e-5:
        return image
    blur_sigma = _sample_float(config, "blur_sigma", 8.0, rng)
    color = _sample_triplet(config.get("color", [0.82, 0.86, 0.88]), rng)
    field = _low_frequency_field_torch(
        int(image.shape[0]),
        int(image.shape[1]),
        rng,
        image.device,
        image.dtype,
    )
    alpha = float(strength) * (0.45 + 0.55 * field)
    blurred = _gaussian_blur_torch(image, blur_sigma)
    color_t = torch.as_tensor(color, device=image.device, dtype=image.dtype)
    veil = 0.7 * blurred + 0.3 * color_t.view(1, 1, 3)
    return image * (1.0 - alpha[..., None]) + veil * alpha[..., None]


def _apply_droplets_np(
    image: np.ndarray,
    config: Mapping[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = image.shape[:2]
    count = max(0, int(round(_sample_float(config, "count", 0.0, rng))))
    if count <= 0:
        return image
    radius_fraction = _sample_float(config, "radius_fraction", 0.025, rng)
    opacity = _sample_float(config, "opacity", 0.4, rng)
    refraction_px = _sample_float(config, "refraction_px", 3.0, rng)
    blur_sigma = _sample_float(config, "blur_sigma", 2.0, rng)

    yy, xx = _coordinate_grid(height, width)
    alpha = np.zeros((height, width), dtype=np.float32)
    offset_x = np.zeros((height, width), dtype=np.float32)
    offset_y = np.zeros((height, width), dtype=np.float32)
    ring = np.zeros((height, width), dtype=np.float32)
    base_radius = radius_fraction * max(height, width)
    for _ in range(count):
        cx = rng.uniform(0.0, max(width - 1, 1))
        cy = rng.uniform(0.0, max(height - 1, 1))
        radius = max(1.0, base_radius * rng.uniform(0.55, 1.45))
        dx = xx - cx
        dy = yy - cy
        dist = np.sqrt(dx * dx + dy * dy) / radius
        inside = dist < 1.0
        profile = np.zeros_like(alpha)
        profile[inside] = (1.0 - dist[inside]) ** 0.45
        alpha = np.maximum(alpha, profile * opacity)
        direction = profile * refraction_px
        offset_x += np.where(inside, dx / radius * direction, 0.0)
        offset_y += np.where(inside, dy / radius * direction, 0.0)
        ring = np.maximum(ring, np.exp(-((dist - 0.72) ** 2) / 0.012) * inside)

    blurred = _gaussian_blur_np(image, blur_sigma)
    refracted = _sample_bilinear_np(blurred, yy + offset_y, xx + offset_x)
    out = image * (1.0 - alpha[..., None]) + refracted * alpha[..., None]
    out += 0.08 * ring[..., None]
    return out


def _bayer_masks(height: int, width: int, pattern: str) -> np.ndarray:
    pattern = pattern.upper()
    tiles = {
        "RGGB": ((0, 1), (1, 2)),
        "BGGR": ((2, 1), (1, 0)),
        "GRBG": ((1, 0), (2, 1)),
        "GBRG": ((1, 2), (0, 1)),
    }
    if pattern not in tiles:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")
    tile = np.asarray(tiles[pattern], dtype=np.int8)
    yy, xx = np.indices((height, width))
    channel_index = tile[yy % 2, xx % 2]
    masks = np.zeros((height, width, 3), dtype=bool)
    for channel in range(3):
        masks[..., channel] = channel_index == channel
    return masks


def _bayer_masks_torch(height: int, width: int, pattern: str, device):
    pattern = pattern.upper()
    tiles = {
        "RGGB": ((0, 1), (1, 2)),
        "BGGR": ((2, 1), (1, 0)),
        "GRBG": ((1, 0), (2, 1)),
        "GBRG": ((1, 2), (0, 1)),
    }
    if pattern not in tiles:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")
    tile = torch.tensor(tiles[pattern], device=device, dtype=torch.int64)
    yy = torch.arange(height, device=device, dtype=torch.int64).view(-1, 1)
    xx = torch.arange(width, device=device, dtype=torch.int64).view(1, -1)
    channel_index = tile[yy % 2, xx % 2]
    channels = torch.arange(3, device=device, dtype=torch.int64).view(1, 1, 3)
    return channel_index[..., None] == channels


def _bayer_mosaic_np(image: np.ndarray, pattern: str) -> np.ndarray:
    masks = _bayer_masks(image.shape[0], image.shape[1], pattern)
    return np.sum(image * masks.astype(np.float32), axis=-1).astype(np.float32)


def _bayer_mosaic_torch(image, pattern: str):
    masks = _bayer_masks_torch(
        int(image.shape[0]),
        int(image.shape[1]),
        pattern,
        image.device,
    )
    return torch.sum(image * masks.to(dtype=image.dtype), dim=-1)


def _mosaic_channel_values(
    shape: tuple[int, int],
    pattern: str,
    values: np.ndarray,
) -> np.ndarray:
    height, width = shape
    masks = _bayer_masks(height, width, pattern)
    return np.sum(masks.astype(np.float32) * values.reshape(1, 1, 3), axis=-1)


def _mosaic_channel_values_torch(
    shape: tuple[int, int],
    pattern: str,
    values: np.ndarray,
    device,
    dtype,
):
    height, width = shape
    masks = _bayer_masks_torch(height, width, pattern, device).to(dtype=dtype)
    values_t = torch.as_tensor(values, device=device, dtype=dtype)
    return torch.sum(masks * values_t.view(1, 1, 3), dim=-1)


def _sensor_identity_cache_key(
    config: Mapping[str, Any],
    shape: tuple[int, int],
    pattern: str,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    sensor_id = str(config.get("sensor_id", "default"))
    seed_rng = np.random.default_rng(_stable_seed("sensor_identity_seed", sensor_id))
    base_seed = int(round(float(sample_value(config.get("seed", 0), seed_rng))))
    height, width = int(shape[0]), int(shape[1])
    resolved = dict(config)
    resolved["sensor_id"] = sensor_id
    resolved["seed"] = base_seed
    resolved["pattern"] = str(pattern).upper()
    resolved["shape"] = (height, width)
    key = (sensor_id, resolved["pattern"], height, width, base_seed)
    return key, resolved


def _build_sensor_identity_maps_np(
    shape: tuple[int, int],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    height, width = int(shape[0]), int(shape[1])
    rng = np.random.default_rng(
        _stable_seed(
            "sensor_identity",
            config.get("sensor_id", "default"),
            config.get("seed", 0),
            config.get("pattern", ""),
            height,
            width,
        )
    )

    prnu_sigma = max(_identity_sample_float(config, "prnu_sigma", 0.0, rng), 0.0)
    if prnu_sigma > 0.0:
        prnu = rng.normal(1.0, prnu_sigma, size=(height, width)).astype(np.float32)
        prnu = np.clip(prnu, 0.25, 4.0)
    else:
        prnu = np.ones((height, width), dtype=np.float32)

    dsnu_sigma = max(_identity_sample_float(config, "dsnu_sigma", 0.0, rng), 0.0)
    dsnu = (
        rng.normal(0.0, dsnu_sigma, size=(height, width)).astype(np.float32)
        if dsnu_sigma > 0.0
        else np.zeros((height, width), dtype=np.float32)
    )

    row_sigma = max(
        _identity_sample_float(config, "persistent_row_sigma", 0.0, rng),
        0.0,
    )
    row_bias = (
        rng.normal(0.0, row_sigma, size=height).astype(np.float32)
        if row_sigma > 0.0
        else np.zeros(height, dtype=np.float32)
    )

    column_sigma = max(
        _identity_sample_float(config, "persistent_column_sigma", 0.0, rng),
        0.0,
    )
    column_bias = (
        rng.normal(0.0, column_sigma, size=width).astype(np.float32)
        if column_sigma > 0.0
        else np.zeros(width, dtype=np.float32)
    )

    hot_probability = float(
        np.clip(
            _identity_sample_float(
                config,
                "persistent_hot_pixel_probability",
                0.0,
                rng,
            ),
            0.0,
            1.0,
        )
    )
    dead_probability = float(
        np.clip(
            _identity_sample_float(
                config,
                "persistent_dead_pixel_probability",
                0.0,
                rng,
            ),
            0.0,
            1.0,
        )
    )
    hot_mask = rng.random((height, width)) < hot_probability
    dead_mask = rng.random((height, width)) < dead_probability
    dead_mask &= ~hot_mask

    return {
        "prnu": prnu.astype(np.float32, copy=False),
        "dsnu": dsnu.astype(np.float32, copy=False),
        "row_bias": row_bias.astype(np.float32, copy=False),
        "column_bias": column_bias.astype(np.float32, copy=False),
        "hot_mask": hot_mask,
        "dead_mask": dead_mask,
    }


def _identity_sample_float(
    config: Mapping[str, Any],
    key: str,
    default: float,
    rng: np.random.Generator,
) -> float:
    value = config.get(key, default)
    if value is None:
        return float(default)
    return float(sample_value(value, rng))


def _stable_seed(*parts: Any) -> int:
    joined = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _demosaic_bilinear_np(raw: np.ndarray, pattern: str) -> np.ndarray:
    masks = _bayer_masks(raw.shape[0], raw.shape[1], pattern).astype(np.float32)
    kernel = np.array(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        dtype=np.float32,
    )
    rgb = np.empty(raw.shape + (3,), dtype=np.float32)
    for channel in range(3):
        mask = masks[..., channel]
        numerator = _convolve2d_same(raw * mask, kernel)
        denominator = _convolve2d_same(mask, kernel)
        rgb[..., channel] = numerator / np.maximum(denominator, 1e-6)
    return rgb


def _convolve2d_same_torch(image, kernel):
    pad_y = int(kernel.shape[0] // 2)
    pad_x = int(kernel.shape[1] // 2)
    x = image.to(dtype=torch.float32).view(
        1,
        1,
        int(image.shape[0]),
        int(image.shape[1]),
    )
    pad_mode = (
        "reflect"
        if pad_y < int(image.shape[0]) and pad_x < int(image.shape[1])
        else "replicate"
    )
    x = torch.nn.functional.pad(x, (pad_x, pad_x, pad_y, pad_y), mode=pad_mode)
    k = kernel.to(device=image.device, dtype=torch.float32).view(
        1,
        1,
        int(kernel.shape[0]),
        int(kernel.shape[1]),
    )
    return torch.nn.functional.conv2d(x, k)[0, 0].to(dtype=image.dtype)


def _demosaic_bilinear_torch(raw, pattern: str):
    masks = _bayer_masks_torch(
        int(raw.shape[0]),
        int(raw.shape[1]),
        pattern,
        raw.device,
    ).to(dtype=raw.dtype)
    kernel = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
        device=raw.device,
        dtype=raw.dtype,
    )
    channels = []
    for channel in range(3):
        mask = masks[..., channel]
        numerator = _convolve2d_same_torch(raw * mask, kernel)
        denominator = _convolve2d_same_torch(mask, kernel)
        channels.append(numerator / torch.clamp(denominator, min=1e-6))
    return torch.stack(channels, dim=-1)


def _apply_persistent_bad_pixels_np(
    raw: np.ndarray,
    identity_maps: Mapping[str, Any],
    *,
    hot_value: float | np.ndarray = 1.0,
    dead_value: float | np.ndarray = 0.0,
) -> np.ndarray:
    out = raw
    hot_mask = np.asarray(identity_maps.get("hot_mask", False), dtype=bool)
    if hot_mask.shape == raw.shape and hot_mask.any():
        out = out.copy()
        hot_arr = np.asarray(hot_value, dtype=np.float32)
        out[hot_mask] = hot_arr[hot_mask] if hot_arr.shape else float(hot_arr)
    dead_mask = np.asarray(identity_maps.get("dead_mask", False), dtype=bool)
    if dead_mask.shape == raw.shape and dead_mask.any():
        if out is raw:
            out = out.copy()
        dead_arr = np.asarray(dead_value, dtype=np.float32)
        out[dead_mask] = dead_arr[dead_mask] if dead_arr.shape else float(dead_arr)
    return out


def _apply_persistent_bad_pixels_torch(
    raw,
    identity_maps: Mapping[str, Any],
    *,
    hot_value: float | Any = 1.0,
    dead_value: float | Any = 0.0,
):
    out = raw
    hot_mask = identity_maps.get("hot_mask")
    if hot_mask is not None and bool(torch.any(hot_mask).detach().cpu()):
        hot_t = torch.as_tensor(hot_value, device=raw.device, dtype=raw.dtype)
        out = torch.where(hot_mask, hot_t.expand_as(raw), out)
    dead_mask = identity_maps.get("dead_mask")
    if dead_mask is not None and bool(torch.any(dead_mask).detach().cpu()):
        dead_t = torch.as_tensor(dead_value, device=raw.device, dtype=raw.dtype)
        out = torch.where(dead_mask, dead_t.expand_as(raw), out)
    return out


def _apply_bad_pixels_np(
    raw: np.ndarray,
    config: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    hot_value: float | np.ndarray = 1.0,
    dead_value: float | np.ndarray = 0.0,
) -> np.ndarray:
    out = raw
    hot_prob = _sample_float(config, "hot_pixel_probability", 0.0, rng)
    if hot_prob > 0.0:
        mask = rng.random(raw.shape) < hot_prob
        if mask.any():
            out = out.copy()
            hot_arr = np.asarray(hot_value, dtype=np.float32)
            out[mask] = hot_arr[mask] if hot_arr.shape else float(hot_arr)
    dead_prob = _sample_float(config, "dead_pixel_probability", 0.0, rng)
    if dead_prob > 0.0:
        mask = rng.random(raw.shape) < dead_prob
        if mask.any():
            out = out.copy()
            dead_arr = np.asarray(dead_value, dtype=np.float32)
            out[mask] = dead_arr[mask] if dead_arr.shape else float(dead_arr)
    return out


def _apply_bad_pixels_torch(
    raw,
    config: Mapping[str, Any],
    rng: np.random.Generator,
    *,
    hot_value: float | Any = 1.0,
    dead_value: float | Any = 0.0,
):
    out = raw
    hot_prob = _sample_float(config, "hot_pixel_probability", 0.0, rng)
    if hot_prob > 0.0:
        mask = _rand_torch(raw.shape, rng, raw.device, raw.dtype) < float(hot_prob)
        if bool(torch.any(mask).detach().cpu()):
            hot_t = torch.as_tensor(hot_value, device=raw.device, dtype=raw.dtype)
            out = torch.where(mask, hot_t.expand_as(raw), out)

    dead_prob = _sample_float(config, "dead_pixel_probability", 0.0, rng)
    if dead_prob > 0.0:
        mask = _rand_torch(raw.shape, rng, raw.device, raw.dtype) < float(dead_prob)
        if bool(torch.any(mask).detach().cpu()):
            dead_t = torch.as_tensor(dead_value, device=raw.device, dtype=raw.dtype)
            out = torch.where(mask, dead_t.expand_as(raw), out)
    return out


# Radiance floor used when blending tone curves in log space, and the bounds
# applied to the log-space result so extreme strengths cannot overflow ``exp``.
_TONE_CURVE_FLOOR = 1e-8
_TONE_CURVE_LOG_MIN = -80.0
_TONE_CURVE_LOG_MAX = 20.0


def _blend_tone_curve_np(
    image: np.ndarray,
    mapped: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Apply a tone curve with ``strength`` acting as an exponent on its response.

    The blend is geometric rather than linear: ``strength`` exponentiates the
    per-pixel response ratio ``mapped / image``, so ``0`` returns the input
    unchanged, ``1`` reproduces the curve exactly, and values above ``1`` apply
    the curve more strongly. A linear blend extrapolates through zero once
    ``strength`` exceeds ``1 / (1 - toe_slope)``, which drives every shadow
    pixel negative and clips whole frames to black; the geometric form stays
    strictly non-negative for any strength.
    """
    if strength == 1.0:
        return np.asarray(mapped, dtype=np.float32)
    if strength == 0.0:
        return np.asarray(image, dtype=np.float32)

    image = np.asarray(image, dtype=np.float32)
    mapped = np.asarray(mapped, dtype=np.float32)
    log_image = np.log(np.maximum(image, _TONE_CURVE_FLOOR))
    log_mapped = np.log(np.maximum(mapped, _TONE_CURVE_FLOOR))
    log_blend = np.clip(
        log_image + strength * (log_mapped - log_image),
        _TONE_CURVE_LOG_MIN,
        _TONE_CURVE_LOG_MAX,
    )
    return np.where(image > 0.0, np.exp(log_blend), 0.0).astype(
        np.float32,
        copy=False,
    )


def _apply_tone_map_np(
    image: np.ndarray,
    mode: str,
    strength: float,
    config: Mapping[str, Any] | None = None,
) -> np.ndarray:
    img = np.clip(image, 0.0, None).astype(np.float32, copy=False)
    strength = max(float(strength), 0.0)
    if mode == "reinhard":
        return img / (1.0 + strength * img)
    if mode == "aces":
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        mapped = (img * (a * img + b)) / (img * (c * img + d) + e)
        return _blend_tone_curve_np(img, mapped, strength)
    if mode == "clip":
        return _clip01(img)
    if mode == "lut":
        mapped = _apply_tone_map_lut_np(img, config or {})
        return _blend_tone_curve_np(img, mapped, strength)
    raise ValueError(f"Unsupported tone_map: {mode}")


def _blend_tone_curve_torch(image, mapped, strength: float):
    """Torch counterpart of :func:`_blend_tone_curve_np`."""
    if strength == 1.0:
        return mapped
    if strength == 0.0:
        return image

    log_image = torch.log(torch.clamp(image, min=_TONE_CURVE_FLOOR))
    log_mapped = torch.log(torch.clamp(mapped, min=_TONE_CURVE_FLOOR))
    log_blend = torch.clamp(
        log_image + strength * (log_mapped - log_image),
        min=_TONE_CURVE_LOG_MIN,
        max=_TONE_CURVE_LOG_MAX,
    )
    blended = torch.exp(log_blend)
    return torch.where(image > 0.0, blended, torch.zeros_like(blended))


def _apply_tone_map_torch(
    image,
    mode: str,
    strength: float,
    config: Mapping[str, Any] | None = None,
):
    img = torch.clamp(image, min=0.0)
    strength = max(float(strength), 0.0)
    if mode == "reinhard":
        return img / (1.0 + strength * img)
    if mode == "aces":
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        mapped = (img * (a * img + b)) / (img * (c * img + d) + e)
        return _blend_tone_curve_torch(img, mapped, strength)
    if mode == "clip":
        return _clip01_torch(img)
    if mode == "lut":
        mapped = _apply_tone_map_lut_torch(img, config or {})
        return _blend_tone_curve_torch(img, mapped, strength)
    raise ValueError(f"Unsupported tone_map: {mode}")


def _apply_tone_map_lut_np(
    image: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    domain = str(config.get("tone_map_lut_domain", "linear")).lower()
    if domain not in {"linear", "scene_linear", "scene-linear"}:
        raise ValueError("tone_map_lut_domain must be 'linear'")
    raw_lut = config.get("tone_map_lut")
    if raw_lut is None:
        raise ValueError("tone_map_lut must be provided when tone_map='lut'")
    lut = np.asarray(raw_lut, dtype=np.float32)
    if lut.ndim != 1 or lut.shape[0] < 2:
        raise ValueError("tone_map_lut must contain at least two values")
    x = np.linspace(0.0, 1.0, lut.shape[0], dtype=np.float32)
    return np.interp(np.clip(image, 0.0, 1.0), x, np.clip(lut, 0.0, 1.0)).astype(
        np.float32,
        copy=False,
    )


def _apply_tone_map_lut_torch(image, config: Mapping[str, Any]):
    domain = str(config.get("tone_map_lut_domain", "linear")).lower()
    if domain not in {"linear", "scene_linear", "scene-linear"}:
        raise ValueError("tone_map_lut_domain must be 'linear'")
    raw_lut = config.get("tone_map_lut")
    if raw_lut is None:
        raise ValueError("tone_map_lut must be provided when tone_map='lut'")
    lut = torch.as_tensor(
        raw_lut,
        device=image.device,
        dtype=image.dtype,
    )
    if lut.ndim != 1 or int(lut.shape[0]) < 2:
        raise ValueError("tone_map_lut must contain at least two values")
    lut = torch.clamp(lut, 0.0, 1.0)
    scaled = torch.clamp(image, 0.0, 1.0) * float(int(lut.shape[0]) - 1)
    lower = torch.floor(scaled).to(dtype=torch.long)
    upper = torch.clamp(lower + 1, max=int(lut.shape[0]) - 1)
    alpha = scaled - lower.to(dtype=image.dtype)
    return lut[lower] * (1.0 - alpha) + lut[upper] * alpha


def _apply_gamma_np(
    image: np.ndarray,
    gamma_spec: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    if isinstance(gamma_spec, str):
        mode = gamma_spec.lower()
        if mode == "srgb":
            return _linear_to_srgb(image)
        if mode in {"none", "false", "off", "linear"}:
            return _clip01(image)
    gamma = float(sample_value(gamma_spec, rng))
    if gamma <= 0.0:
        raise ValueError("gamma must be > 0")
    return np.power(_clip01(image), 1.0 / gamma).astype(np.float32, copy=False)


def _apply_gamma_torch(
    image,
    gamma_spec: Any,
    rng: np.random.Generator,
):
    if isinstance(gamma_spec, str):
        mode = gamma_spec.lower()
        if mode == "srgb":
            return _linear_to_srgb_torch(image)
        if mode in {"none", "false", "off", "linear"}:
            return _clip01_torch(image)
    gamma = float(sample_value(gamma_spec, rng))
    if gamma <= 0.0:
        raise ValueError("gamma must be > 0")
    return torch.pow(_clip01_torch(image), 1.0 / gamma)


def _crop_np(
    image: np.ndarray,
    crop: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = image.shape[:2]
    if isinstance(crop, dict):
        if "center_fraction" in crop:
            fraction = float(sample_value(crop["center_fraction"], rng))
            fraction = float(np.clip(fraction, 1e-3, 1.0))
            crop_w = max(1, int(round(width * fraction)))
            crop_h = max(1, int(round(height * fraction)))
            x0 = (width - crop_w) // 2
            y0 = (height - crop_h) // 2
        else:
            x0 = int(round(_sample_float(crop, "x", 0.0, rng)))
            y0 = int(round(_sample_float(crop, "y", 0.0, rng)))
            crop_w = int(round(_sample_float(crop, "width", width, rng)))
            crop_h = int(round(_sample_float(crop, "height", height, rng)))
    elif isinstance(crop, (list, tuple)) and len(crop) == 4:
        values = [int(round(float(sample_value(value, rng)))) for value in crop]
        x0, y0, crop_w, crop_h = values
    else:
        raise ValueError("crop must be an object or [x, y, width, height]")

    x0 = int(np.clip(x0, 0, max(width - 1, 0)))
    y0 = int(np.clip(y0, 0, max(height - 1, 0)))
    x1 = int(np.clip(x0 + max(crop_w, 1), x0 + 1, width))
    y1 = int(np.clip(y0 + max(crop_h, 1), y0 + 1, height))
    return image[y0:y1, x0:x1]


def _crop_torch(
    image,
    crop: Any,
    rng: np.random.Generator,
):
    height, width = int(image.shape[0]), int(image.shape[1])
    if isinstance(crop, dict):
        if "center_fraction" in crop:
            fraction = float(sample_value(crop["center_fraction"], rng))
            fraction = float(np.clip(fraction, 1e-3, 1.0))
            crop_w = max(1, int(round(width * fraction)))
            crop_h = max(1, int(round(height * fraction)))
            x0 = (width - crop_w) // 2
            y0 = (height - crop_h) // 2
        else:
            x0 = int(round(_sample_float(crop, "x", 0.0, rng)))
            y0 = int(round(_sample_float(crop, "y", 0.0, rng)))
            crop_w = int(round(_sample_float(crop, "width", width, rng)))
            crop_h = int(round(_sample_float(crop, "height", height, rng)))
    elif isinstance(crop, (list, tuple)) and len(crop) == 4:
        values = [int(round(float(sample_value(value, rng)))) for value in crop]
        x0, y0, crop_w, crop_h = values
    else:
        raise ValueError("crop must be an object or [x, y, width, height]")

    x0 = int(np.clip(x0, 0, max(width - 1, 0)))
    y0 = int(np.clip(y0, 0, max(height - 1, 0)))
    x1 = int(np.clip(x0 + max(crop_w, 1), x0 + 1, width))
    y1 = int(np.clip(y0 + max(crop_h, 1), y0 + 1, height))
    return image[y0:y1, x0:x1]


def _resolve_resize(
    config: Mapping[str, Any],
    shape: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    height, width = shape[:2]
    resize = config.get("resize")
    if resize is not None:
        if isinstance(resize, dict):
            target_w = int(round(_sample_float(resize, "width", width, rng)))
            target_h = int(round(_sample_float(resize, "height", height, rng)))
        elif isinstance(resize, (list, tuple)) and len(resize) == 2:
            target_w = int(round(float(sample_value(resize[0], rng))))
            target_h = int(round(float(sample_value(resize[1], rng))))
        else:
            raise ValueError("resize must be an object or [width, height]")
        return max(1, target_w), max(1, target_h)

    scale = config.get("resize_scale")
    if scale is None:
        return None
    factor = float(sample_value(scale, rng))
    if factor <= 0.0:
        raise ValueError("resize_scale must be > 0")
    return max(1, int(round(width * factor))), max(1, int(round(height * factor)))


def _resize_np(
    image: np.ndarray,
    size: tuple[int, int],
    resample: str,
) -> np.ndarray:
    pil = Image.fromarray(_to_uint8(image), mode="RGB")
    resized = pil.resize(size, resample=_pil_resampling(resample))
    return np.asarray(resized, dtype=np.float32) / 255.0


def _resize_torch(
    image,
    size: tuple[int, int],
    resample: str,
):
    target_width, target_height = size
    mode = {
        "nearest": "nearest",
        "bilinear": "bilinear",
        "bicubic": "bicubic",
        "lanczos": "bicubic",
        "area": "area",
        "box": "area",
    }.get(resample.lower(), "bilinear")
    x = image.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    kwargs: dict[str, Any] = {"size": (target_height, target_width), "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    resized = torch.nn.functional.interpolate(x, **kwargs)
    return resized[0].permute(1, 2, 0).to(dtype=image.dtype)


def _resize_map_np(map_2d: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy = np.linspace(0.0, max(map_2d.shape[0] - 1, 0), height, dtype=np.float32)
    xx = np.linspace(0.0, max(map_2d.shape[1] - 1, 0), width, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    return _sample_bilinear_np(map_2d.astype(np.float32, copy=False), grid_y, grid_x)


def _resize_map_torch(map_2d, shape: tuple[int, int]):
    height, width = shape
    x = map_2d.to(dtype=torch.float32).view(
        1,
        1,
        int(map_2d.shape[0]),
        int(map_2d.shape[1]),
    )
    resized = torch.nn.functional.interpolate(
        x,
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    )
    return resized[0, 0].to(dtype=map_2d.dtype)


def _pil_resampling(name: str):
    mapping = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
        "area": Image.Resampling.BOX,
        "box": Image.Resampling.BOX,
    }
    return mapping.get(name.lower(), Image.Resampling.BILINEAR)


def _quantize_np(image: np.ndarray, bit_depth: int) -> np.ndarray:
    if bit_depth >= 16:
        return _clip01(image)
    levels = max(2, 2**bit_depth - 1)
    return np.round(_clip01(image) * levels) / float(levels)


def _quantize_torch(image, bit_depth: int):
    if bit_depth >= 16:
        return _clip01_torch(image)
    levels = max(2, 2**bit_depth - 1)
    return torch.round(_clip01_torch(image) * float(levels)) / float(levels)


def _jpeg_roundtrip_np(
    image: np.ndarray,
    *,
    quality: int,
    subsampling: int,
) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(_to_uint8(image), mode="RGB").save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=int(np.clip(subsampling, 0, 2)),
    )
    buffer.seek(0)
    decoded = Image.open(buffer).convert("RGB")
    return np.asarray(decoded, dtype=np.float32) / 255.0


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.round(_clip01(image) * 255.0).astype(np.uint8)
