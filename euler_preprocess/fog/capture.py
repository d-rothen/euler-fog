from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from PIL import Image

from euler_preprocess.common.noise import perlin_fbm
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

    def _apply_np(
        self,
        image: np.ndarray,
        context: CaptureContext,
        rng: np.random.Generator,
        config: Mapping[str, Any],
    ) -> np.ndarray:
        return image

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
            "exposure_gain"
            if config.get("exposure_gain") is not None
            else "gain"
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
        "demosaic": True,
    }

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
            noisy_signal = noisy_signal + rng.normal(
                0.0,
                1.0,
                raw_signal.shape,
            ).astype(np.float32) * read_sigma * noise_modulation

        raw = black_map + noisy_signal * raw_range

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


class ISPStage(ConfiguredCaptureStage):
    """Demosaiced camera-space RGB to display RGB with ISP artifacts."""

    name = "isp"
    DEFAULTS = {
        "denoise_sigma": {"dist": "uniform", "min": 0.0, "max": 0.45},
        "color_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "tone_map": "reinhard",
        "tone_map_strength": {"dist": "uniform", "min": 0.05, "max": 0.25},
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
            img = _apply_tone_map_np(img, tone_map, strength)

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
        np_image = _torch_to_numpy_image(image)
        for stage in self.stages:
            np_image = stage.apply_np(np_image, context)
        return _numpy_to_torch_like(np_image, image)

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
        },
        "sensor": {
            "iso": {"dist": "choice", "values": [800.0, 1600.0, 3200.0]},
            "read_noise_electrons": {"dist": "uniform", "min": 3.0, "max": 10.0},
            "row_noise_sigma": {"dist": "uniform", "min": 0.002, "max": 0.009},
        },
        "isp": {
            "denoise_sigma": {"dist": "uniform", "min": 0.25, "max": 0.8},
            "sharpen_amount": {"dist": "uniform", "min": 0.08, "max": 0.28},
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
        name: dict(profile)
        for name, profile in _BUILTIN_CAMERA_PROFILES.items()
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


def _context_depth_map(context: CaptureContext, shape: tuple[int, int]) -> np.ndarray | None:
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
    return np.clip(1.0 - np.exp(-np.maximum(depth, 0.0) * np.maximum(k_map, 0.0)), 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )


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
        auto_gain = math.exp(
            (1.0 - protection) * log_auto + protection * log_highlight
        )

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
    if metering in {"mean", "average"}:
        meter_luma = mean_luma
    elif metering in {"percentile", "median"}:
        meter_luma = percentile_luma
    elif metering in {"center_percentile", "centered_percentile"}:
        meter_luma = (
            (1.0 - center_weight) * percentile_luma
            + center_weight * center_luma
        )
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
        blotch = (_low_frequency_field(image.shape[0], image.shape[1], rng) - 0.5)
        out += blotch[..., None] * (2.0 * blotch_sigma) * weight[..., None]

    return _clip01(out)


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
        (1.0 - luminance_preservation) * raw
        + luminance_preservation * preserved
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
    img = _clip01(image)
    return np.where(
        img <= 0.04045,
        img / 12.92,
        ((img + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32, copy=False)


def _linear_to_srgb(image: np.ndarray) -> np.ndarray:
    img = _clip01(image)
    return np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * np.power(img, 1.0 / 2.4) - 0.055,
    ).astype(np.float32, copy=False)


def _apply_color_matrix(image: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.tensordot(image, matrix.T, axes=([-1], [0])).astype(
        np.float32,
        copy=False,
    )


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
            out += float(kernel[y, x]) * padded[
                y : y + image.shape[0],
                x : x + image.shape[1],
            ]
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
    return (
        image * (1.0 - alpha[..., None]) + shifted * alpha[..., None]
    ).astype(np.float32, copy=False)


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


def _motion_blur_np(image: np.ndarray, length_px: float, angle_deg: float) -> np.ndarray:
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


def _low_frequency_field(
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    largest = max(height, width)
    scales = [max(4, int(largest * fraction)) for fraction in (0.75, 0.38, 0.19)]
    return perlin_fbm(height, width, scales, rng)


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


def _bayer_mosaic_np(image: np.ndarray, pattern: str) -> np.ndarray:
    masks = _bayer_masks(image.shape[0], image.shape[1], pattern)
    return np.sum(image * masks.astype(np.float32), axis=-1).astype(np.float32)


def _mosaic_channel_values(
    shape: tuple[int, int],
    pattern: str,
    values: np.ndarray,
) -> np.ndarray:
    height, width = shape
    masks = _bayer_masks(height, width, pattern)
    return np.sum(masks.astype(np.float32) * values.reshape(1, 1, 3), axis=-1)


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


def _apply_tone_map_np(image: np.ndarray, mode: str, strength: float) -> np.ndarray:
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
        return img * (1.0 - strength) + mapped * strength
    if mode == "clip":
        return _clip01(img)
    raise ValueError(f"Unsupported tone_map: {mode}")


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


def _resize_map_np(map_2d: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy = np.linspace(0.0, max(map_2d.shape[0] - 1, 0), height, dtype=np.float32)
    xx = np.linspace(0.0, max(map_2d.shape[1] - 1, 0), width, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(yy, xx, indexing="ij")
    return _sample_bilinear_np(map_2d.astype(np.float32, copy=False), grid_y, grid_x)


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
