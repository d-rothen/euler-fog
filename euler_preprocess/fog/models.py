from __future__ import annotations

import logging
import math

import numpy as np

from euler_preprocess.common.noise import perlin_fbm
from euler_preprocess.common.sampling import deep_merge, sample_value

try:
    import torch
except ImportError:
    torch = None

_logger = logging.getLogger("foggify")

AIRLIGHT_METHODS = ("from_sky", "dcp", "dcp_heuristic")

DEFAULT_CONTRAST_THRESHOLD = 0.05

DEFAULT_AIRLIGHT_DAMPENING_CONFIG = {
    "enabled": True,
    "apply_to": "estimated",
    "reference_visibility_m": 80.0,
    "min_factor": 0.45,
    "max_factor": 1.0,
    "strength": 1.0,
}

DEFAULT_SCENE_ILLUMINATION_CONFIG = {
    "enabled": False,
    "global_ev": 0.0,
    "near_ev": 0.0,
    "near_decay_depth_m": 15.0,
    "fog_coupled_ev": 0.0,
    "airlight_ev_ratio": 0.0,
    "sky_weight": 0.0,
    "min_radiance_scale": 0.08,
}

_AIRLIGHT_DAMPENING_KEYS = (
    "airlight_dampening",
    "airlight_damping",
    "airlight_intensity_dampening",
    "airlight_intensity_damping",
)

DEFAULT_MODEL_CONFIGS = {
    "uniform": {
        "visibility_m": {"dist": "constant", "value": 80.0},
        "atmospheric_light": "from_sky",
        "airlight_dampening": dict(DEFAULT_AIRLIGHT_DAMPENING_CONFIG),
    },
    "heterogeneous_k": {
        "visibility_m": {"dist": "constant", "value": 80.0},
        "atmospheric_light": "from_sky",
        "airlight_dampening": dict(DEFAULT_AIRLIGHT_DAMPENING_CONFIG),
        "k_hetero": {
            "scales": "smooth_auto",
            "correlation_length_fraction": 0.25,
            "octaves": 3,
            "max_scale": None,
            "min_factor": 0.65,
            "max_factor": 1.45,
            "contrast": 0.65,
            "normalize_to_mean": True,
        },
    },
    "heterogeneous_ls": {
        "visibility_m": {"dist": "constant", "value": 80.0},
        "atmospheric_light": "from_sky",
        "airlight_dampening": dict(DEFAULT_AIRLIGHT_DAMPENING_CONFIG),
        "ls_hetero": {
            "scales": "smooth_auto",
            "correlation_length_fraction": 0.35,
            "octaves": 3,
            "max_scale": None,
            "min_factor": 0.85,
            "max_factor": 1.08,
            "contrast": 0.55,
            "normalize_to_mean": False,
        },
    },
    "heterogeneous_k_ls": {
        "visibility_m": {"dist": "constant", "value": 80.0},
        "atmospheric_light": "from_sky",
        "airlight_dampening": dict(DEFAULT_AIRLIGHT_DAMPENING_CONFIG),
        "k_hetero": {
            "scales": "smooth_auto",
            "correlation_length_fraction": 0.25,
            "octaves": 3,
            "max_scale": None,
            "min_factor": 0.65,
            "max_factor": 1.45,
            "contrast": 0.65,
            "normalize_to_mean": True,
        },
        "ls_hetero": {
            "scales": "smooth_auto",
            "correlation_length_fraction": 0.35,
            "octaves": 3,
            "max_scale": None,
            "min_factor": 0.85,
            "max_factor": 1.08,
            "contrast": 0.55,
            "normalize_to_mean": False,
        },
    },
}


def visibility_to_k(visibility_m: float, contrast_threshold: float) -> float:
    if not math.isfinite(visibility_m) or visibility_m <= 0:
        raise ValueError(f"Visibility must be > 0, got {visibility_m}")
    if not math.isfinite(contrast_threshold) or not 0.0 < contrast_threshold < 1.0:
        raise ValueError(
            f"Contrast threshold must be in (0, 1), got {contrast_threshold}"
        )
    return -math.log(contrast_threshold) / visibility_m


def resolve_scattering_coefficient(
    model_cfg: dict,
    rng: np.random.Generator,
    contrast_threshold_default: float,
) -> tuple[float, float | None, float]:
    """Resolve the mean scattering coefficient for a fog model.

    The historic config expresses fog density as meteorological visibility
    (MOR) and converts it to beta.  Stepped augmentation configs may provide
    beta directly via ``scattering_coefficient`` or its alias ``beta``.

    For heterogeneous-k models this resolves the scalar base beta before the
    spatial factor field is generated.  If ``k_hetero.normalize_to_mean`` is
    enabled, the resulting beta map has this value as its arithmetic mean.

    Returns ``(beta, visibility_m, contrast_threshold)``.  ``visibility_m`` is
    ``None`` when beta was configured directly.
    """

    contrast_threshold = float(
        sample_value(
            model_cfg.get("contrast_threshold", contrast_threshold_default), rng
        )
    )
    beta_spec = model_cfg.get("scattering_coefficient", model_cfg.get("beta"))
    if beta_spec is not None:
        beta = float(sample_value(beta_spec, rng))
        if not math.isfinite(beta) or beta < 0:
            raise ValueError(f"Scattering coefficient must be >= 0, got {beta}")
        return beta, None, contrast_threshold

    visibility = float(sample_value(model_cfg.get("visibility_m"), rng))
    return (
        visibility_to_k(visibility, contrast_threshold),
        visibility,
        contrast_threshold,
    )


def normalize_atmospheric_light(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim == 0:
        value = np.array([value, value, value], dtype=np.float32)
    if value.ndim != 1 or value.shape[0] != 3:
        raise ValueError("atmospheric_light must be scalar or length-3 list")
    if value.max() > 1.0:
        value = value / 255.0
    return np.clip(value, 0.0, 1.0)


def normalize_atmospheric_light_torch(value: "torch.Tensor") -> "torch.Tensor":
    value_t = value.to(torch.float32)
    if value_t.ndim == 0:
        value_t = value_t.repeat(3)
    if value_t.ndim == 1:
        value_t = value_t.unsqueeze(0)
    if value_t.ndim != 2 or value_t.shape[-1] != 3:
        raise ValueError("atmospheric_light must be scalar or length-3 list")
    max_val = float(value_t.max().item()) if value_t.numel() else 0.0
    if max_val > 1.0:
        value_t = value_t / 255.0
    return torch.clamp(value_t, 0.0, 1.0)


def _raw_airlight_dampening_config(model_cfg: dict):
    raw_cfg = model_cfg.get("airlight_dampening", {})
    for key in _AIRLIGHT_DAMPENING_KEYS[1:]:
        if key not in model_cfg:
            continue
        override = model_cfg[key]
        if isinstance(raw_cfg, dict) and isinstance(override, dict):
            raw_cfg = deep_merge(raw_cfg, override)
        else:
            raw_cfg = override
    return raw_cfg


def resolve_airlight_dampening_config(
    model_cfg: dict,
    rng: np.random.Generator,
    contrast_threshold: float,
) -> dict:
    """Resolve model-level airlight intensity dampening controls.

    The default applies only to estimated airlight. Literal atmospheric-light
    colours remain exact unless ``apply_to`` is set to ``"all"``.
    """
    raw_cfg = _raw_airlight_dampening_config(model_cfg)
    if isinstance(raw_cfg, bool):
        raw_cfg = {"enabled": raw_cfg}
    if raw_cfg is None:
        raw_cfg = {}
    if not isinstance(raw_cfg, dict):
        raise ValueError("airlight_dampening must be a boolean or object")

    resolved = deep_merge(DEFAULT_AIRLIGHT_DAMPENING_CONFIG, raw_cfg)
    enabled = resolved.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("airlight_dampening.enabled must be a boolean")

    apply_to = str(resolved.get("apply_to", "estimated"))
    if apply_to not in ("estimated", "all", "none"):
        raise ValueError(
            "airlight_dampening.apply_to must be 'estimated', 'all', or 'none'"
        )

    min_factor = _sample_float(
        resolved.get("min_factor", 0.45),
        rng,
        "airlight_dampening.min_factor",
    )
    max_factor = _sample_float(
        resolved.get("max_factor", 1.0),
        rng,
        "airlight_dampening.max_factor",
    )
    strength = _sample_float(
        resolved.get("strength", 1.0),
        rng,
        "airlight_dampening.strength",
    )
    if min_factor < 0.0:
        raise ValueError(
            f"airlight_dampening.min_factor must be >= 0, got {min_factor}"
        )
    if max_factor < 0.0:
        raise ValueError(
            f"airlight_dampening.max_factor must be >= 0, got {max_factor}"
        )
    if min_factor > max_factor:
        raise ValueError("airlight_dampening.min_factor must be <= max_factor")
    if strength < 0.0:
        raise ValueError(f"airlight_dampening.strength must be >= 0, got {strength}")

    reference_beta_spec = resolved.get(
        "reference_scattering_coefficient",
        resolved.get("reference_beta"),
    )
    if reference_beta_spec is None:
        reference_visibility = _sample_float(
            resolved.get("reference_visibility_m", 80.0),
            rng,
            "airlight_dampening.reference_visibility_m",
        )
        reference_beta = visibility_to_k(reference_visibility, contrast_threshold)
    else:
        reference_beta = _sample_float(
            reference_beta_spec,
            rng,
            "airlight_dampening.reference_scattering_coefficient",
        )
        if reference_beta <= 0.0:
            raise ValueError(
                "airlight_dampening.reference_scattering_coefficient "
                f"must be > 0, got {reference_beta}"
            )

    return {
        "enabled": enabled,
        "apply_to": apply_to,
        "reference_beta": reference_beta,
        "min_factor": min_factor,
        "max_factor": max_factor,
        "strength": strength,
    }


def should_dampen_airlight(
    dampening_cfg: dict,
    *,
    estimated_airlight: bool,
) -> bool:
    if not dampening_cfg["enabled"]:
        return False
    apply_to = dampening_cfg["apply_to"]
    if apply_to == "none":
        return False
    if apply_to == "all":
        return True
    return estimated_airlight


def airlight_dampening_factor(beta, dampening_cfg: dict) -> np.ndarray:
    beta_arr = np.maximum(np.asarray(beta, dtype=np.float32), 0.0)
    reference_beta = max(
        float(dampening_cfg["reference_beta"]),
        np.finfo(np.float32).eps,
    )
    relative_strength = beta_arr / reference_beta
    factor = float(dampening_cfg["min_factor"]) + (
        float(dampening_cfg["max_factor"]) - float(dampening_cfg["min_factor"])
    ) / (1.0 + float(dampening_cfg["strength"]) * relative_strength)
    return np.asarray(factor, dtype=np.float32)


def dampen_airlight(
    airlight: np.ndarray,
    beta,
    model_cfg: dict,
    rng: np.random.Generator,
    contrast_threshold: float,
    *,
    estimated_airlight: bool,
) -> np.ndarray:
    light = np.asarray(airlight, dtype=np.float32)
    dampening_cfg = resolve_airlight_dampening_config(
        model_cfg,
        rng,
        contrast_threshold,
    )
    if not should_dampen_airlight(
        dampening_cfg,
        estimated_airlight=estimated_airlight,
    ):
        return light
    factor = airlight_dampening_factor(beta, dampening_cfg)
    while factor.ndim < light.ndim:
        factor = np.expand_dims(factor, axis=-1)
    return np.clip(light * factor, 0.0, 1.0).astype(np.float32, copy=False)


def dampen_airlight_torch(
    airlight: "torch.Tensor",
    beta,
    model_cfg: dict,
    rng: np.random.Generator,
    contrast_threshold: float,
    *,
    estimated_airlight: bool,
) -> "torch.Tensor":
    dampening_cfg = resolve_airlight_dampening_config(
        model_cfg,
        rng,
        contrast_threshold,
    )
    if not should_dampen_airlight(
        dampening_cfg,
        estimated_airlight=estimated_airlight,
    ):
        return airlight
    if torch.is_tensor(beta):
        beta_t = beta.to(device=airlight.device, dtype=airlight.dtype)
    else:
        beta_t = torch.tensor(beta, device=airlight.device, dtype=airlight.dtype)
    beta_t = torch.clamp(beta_t, min=0.0)
    reference_beta = max(
        float(dampening_cfg["reference_beta"]),
        float(torch.finfo(airlight.dtype).eps),
    )
    relative_strength = beta_t / reference_beta
    factor = float(dampening_cfg["min_factor"]) + (
        float(dampening_cfg["max_factor"]) - float(dampening_cfg["min_factor"])
    ) / (1.0 + float(dampening_cfg["strength"]) * relative_strength)
    while factor.ndim < airlight.ndim:
        factor = factor.unsqueeze(-1)
    return torch.clamp(airlight * factor, 0.0, 1.0)


def estimate_airlight_torch(
    image: "torch.Tensor",
    sky_mask: "torch.Tensor",
    sample_id: str | None = None,
) -> "torch.Tensor":
    if sky_mask.sum() == 0:
        id_str = f" (sample {sample_id})" if sample_id else ""
        _logger.warning(
            "No sky pixels in segmentation mask%s; "
            "using default airlight fallback [1.0, 1.0, 1.0]",
            id_str,
        )
        return torch.ones(3, device=image.device, dtype=image.dtype)
    airlight_pixels = image[sky_mask]
    airlight = airlight_pixels.mean(dim=0)
    if not torch.all(torch.isfinite(airlight)):
        id_str = f" (sample {sample_id})" if sample_id else ""
        _logger.warning(
            "Airlight estimated from sky pixels contains non-finite values "
            "(%s)%s; using default airlight fallback [1.0, 1.0, 1.0]",
            airlight.tolist(),
            id_str,
        )
        return torch.ones(3, device=image.device, dtype=image.dtype)
    return airlight


def resolve_scales(
    hetero_cfg: dict, height: int, width: int, rng: np.random.Generator
) -> list[int]:
    scales_spec = hetero_cfg.get("scales", "auto")
    scales_spec = sample_value(scales_spec, rng)
    if isinstance(scales_spec, str):
        if scales_spec == "smooth_auto":
            return _resolve_smooth_auto_scales(hetero_cfg, height, width, rng)
        if scales_spec != "auto":
            raise ValueError(f"Unsupported scales value: {scales_spec}")
        min_scale = int(sample_value(hetero_cfg.get("min_scale", 2), rng))
        max_scale = hetero_cfg.get("max_scale", None)
        if max_scale is None:
            max_scale = max(height, width)
        max_scale = int(sample_value(max_scale, rng))
        scales = []
        scale = max(1, min_scale)
        while scale <= max_scale:
            scales.append(scale)
            scale *= 2
        return scales or [max(height, width)]
    if isinstance(scales_spec, (int, float)):
        return [int(scales_spec)]
    if isinstance(scales_spec, list):
        return [int(s) for s in scales_spec if int(s) > 0]
    raise ValueError(f"Unsupported scales spec: {scales_spec}")


def _resolve_smooth_auto_scales(
    hetero_cfg: dict,
    height: int,
    width: int,
    rng: np.random.Generator,
) -> list[int]:
    """Resolve low-frequency Perlin scales for realistic fog gradients."""
    min_dimension = max(1, min(height, width))
    max_dimension = max(1, max(height, width))

    base_scale = _resolve_scale_alias(
        hetero_cfg,
        rng,
        absolute_keys=("correlation_length", "base_scale", "min_scale"),
        fraction_keys=(
            "correlation_length_fraction",
            "base_scale_fraction",
            "min_scale_fraction",
        ),
        fraction_basis=min_dimension,
        default=max(4, int(round(min_dimension * 0.25))),
    )
    max_scale = _resolve_scale_alias(
        hetero_cfg,
        rng,
        absolute_keys=("max_scale",),
        fraction_keys=("max_scale_fraction",),
        fraction_basis=max_dimension,
        default=max_dimension,
        allow_none=True,
    )
    max_scale = max(base_scale, max_scale)

    octaves = max(
        1,
        int(round(_sample_float(hetero_cfg.get("octaves", 3), rng, "octaves"))),
    )
    lacunarity = _sample_float(hetero_cfg.get("lacunarity", 2.0), rng, "lacunarity")
    if lacunarity <= 1.0:
        raise ValueError(f"lacunarity must be > 1.0, got {lacunarity}")

    scales: list[int] = []
    scale = float(base_scale)
    for _ in range(octaves):
        scales.append(max(1, int(round(scale))))
        if scale >= max_scale:
            break
        scale = min(scale * lacunarity, float(max_scale))
    return _unique_positive_scales(scales)


def _resolve_scale_alias(
    hetero_cfg: dict,
    rng: np.random.Generator,
    *,
    absolute_keys: tuple[str, ...],
    fraction_keys: tuple[str, ...],
    fraction_basis: int,
    default: int,
    allow_none: bool = False,
) -> int:
    for key in absolute_keys:
        if key not in hetero_cfg:
            continue
        raw_value = hetero_cfg[key]
        if raw_value is None and allow_none:
            break
        return _scale_pixels(raw_value, rng, key)
    for key in fraction_keys:
        if key not in hetero_cfg:
            continue
        fraction = _sample_float(hetero_cfg[key], rng, key)
        if fraction <= 0:
            raise ValueError(f"{key} must be > 0, got {fraction}")
        return max(1, int(round(float(fraction_basis) * fraction)))
    return max(1, int(default))


def _scale_pixels(value, rng: np.random.Generator, name: str) -> int:
    scale = _sample_float(value, rng, name)
    if scale <= 0:
        raise ValueError(f"{name} must be > 0, got {scale}")
    return max(1, int(round(scale)))


def _sample_float(value, rng: np.random.Generator, name: str) -> float:
    sampled = sample_value(value, rng)
    try:
        resolved = float(sampled)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must resolve to a number, got {sampled!r}") from exc
    if not math.isfinite(resolved):
        raise ValueError(f"{name} must be finite, got {resolved}")
    return resolved


def _unique_positive_scales(scales: list[int]) -> list[int]:
    unique: list[int] = []
    seen: set[int] = set()
    for scale in scales:
        scale = int(scale)
        if scale <= 0 or scale in seen:
            continue
        seen.add(scale)
        unique.append(scale)
    return unique or [1]


def prepare_noise_field(
    noise: np.ndarray,
    hetero_cfg: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply optional smoothing and contrast control to a Perlin noise field."""
    noise = np.asarray(noise, dtype=np.float32)
    sigma = resolve_smoothing_sigma(hetero_cfg, noise.shape[0], noise.shape[1], rng)
    if sigma > 0.0:
        noise = _gaussian_blur_np(noise, sigma)
    noise = _normalize_noise_np(noise)
    contrast = resolve_noise_contrast(hetero_cfg, rng)
    if contrast != 1.0:
        noise = 0.5 + (noise - 0.5) * contrast
    return np.clip(noise, 0.0, 1.0).astype(np.float32, copy=False)


def prepare_noise_field_torch(
    noise: "torch.Tensor",
    hetero_cfg: dict,
    rng: np.random.Generator,
) -> "torch.Tensor":
    """Torch equivalent of :func:`prepare_noise_field`."""
    height = int(noise.shape[-2])
    width = int(noise.shape[-1])
    sigma = resolve_smoothing_sigma(hetero_cfg, height, width, rng)
    if sigma > 0.0:
        noise = _gaussian_blur_torch(noise, sigma)
    noise = _normalize_noise_torch(noise)
    contrast = resolve_noise_contrast(hetero_cfg, rng)
    if contrast != 1.0:
        noise = 0.5 + (noise - 0.5) * contrast
    return torch.clamp(noise, 0.0, 1.0)


def resolve_smoothing_sigma(
    hetero_cfg: dict,
    height: int,
    width: int,
    rng: np.random.Generator,
) -> float:
    for key in ("smooth_sigma", "smoothing_sigma", "blur_sigma"):
        if key in hetero_cfg:
            sigma = _sample_float(hetero_cfg[key], rng, key)
            if sigma < 0:
                raise ValueError(f"{key} must be >= 0, got {sigma}")
            return sigma
    for key in (
        "smooth_sigma_fraction",
        "smoothing_sigma_fraction",
        "blur_sigma_fraction",
    ):
        if key in hetero_cfg:
            fraction = _sample_float(hetero_cfg[key], rng, key)
            if fraction < 0:
                raise ValueError(f"{key} must be >= 0, got {fraction}")
            return fraction * float(max(1, min(height, width)))
    return 0.0


def resolve_noise_contrast(hetero_cfg: dict, rng: np.random.Generator) -> float:
    raw = hetero_cfg.get("contrast", hetero_cfg.get("noise_contrast", 1.0))
    contrast = _sample_float(raw, rng, "contrast")
    if contrast < 0:
        raise ValueError(f"contrast must be >= 0, got {contrast}")
    return contrast


def _normalize_noise_np(noise: np.ndarray) -> np.ndarray:
    min_val = float(np.min(noise))
    max_val = float(np.max(noise))
    denom = max_val - min_val
    if denom <= 1e-8:
        return np.full_like(noise, 0.5, dtype=np.float32)
    return ((noise - min_val) / denom).astype(np.float32, copy=False)


def _normalize_noise_torch(noise: "torch.Tensor") -> "torch.Tensor":
    min_val = noise.amin()
    max_val = noise.amax()
    denom = max_val - min_val
    if float(denom.item()) <= 1e-8:
        return torch.full_like(noise, 0.5)
    return (noise - min_val) / denom


def _gaussian_kernel_np(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / float(sigma)) ** 2)
    kernel /= float(kernel.sum())
    return kernel.astype(np.float32)


def _convolve_axis_np(
    values: np.ndarray,
    kernel: np.ndarray,
    axis: int,
) -> np.ndarray:
    radius = kernel.shape[0] // 2
    padding = [(0, 0)] * values.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(values, padding, mode="edge")
    result = np.zeros_like(values, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        slices = [slice(None)] * values.ndim
        slices[axis] = slice(offset, offset + values.shape[axis])
        result += float(weight) * padded[tuple(slices)]
    return result


def _gaussian_blur_np(noise: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return noise
    kernel = _gaussian_kernel_np(sigma)
    blurred = _convolve_axis_np(noise, kernel, axis=1)
    return _convolve_axis_np(blurred, kernel, axis=0)


def _gaussian_blur_torch(noise: "torch.Tensor", sigma: float) -> "torch.Tensor":
    if sigma <= 0.0:
        return noise
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = torch.arange(
        -radius,
        radius + 1,
        device=noise.device,
        dtype=torch.float32,
    )
    kernel = torch.exp(-0.5 * (offsets / float(sigma)) ** 2)
    kernel = kernel / kernel.sum()
    x = noise.to(dtype=torch.float32).view(
        1,
        1,
        int(noise.shape[-2]),
        int(noise.shape[-1]),
    )
    x = torch.nn.functional.pad(x, (radius, radius, 0, 0), mode="replicate")
    x = torch.nn.functional.conv2d(x, kernel.view(1, 1, 1, -1))
    x = torch.nn.functional.pad(x, (0, 0, radius, radius), mode="replicate")
    x = torch.nn.functional.conv2d(x, kernel.view(1, 1, -1, 1))
    return x.view(noise.shape)


def modulate_with_noise(
    mean_value: np.ndarray,
    noise: np.ndarray,
    min_factor: float,
    max_factor: float,
    normalize_to_mean: bool,
) -> np.ndarray:
    factors = min_factor + (max_factor - min_factor) * noise
    if normalize_to_mean:
        mean_factor = float(np.mean(factors))
        if mean_factor > 0:
            factors = factors / mean_factor
    if mean_value.ndim == 1:
        mean_value = mean_value.reshape(1, 1, -1)
    return mean_value * factors[..., None]


def modulate_with_noise_torch(
    mean_value: "torch.Tensor",
    noise: "torch.Tensor",
    min_factor: float,
    max_factor: float,
    normalize_to_mean: bool,
) -> "torch.Tensor":
    factors = min_factor + (max_factor - min_factor) * noise
    if normalize_to_mean:
        mean_factor = float(factors.mean().item())
        if mean_factor > 0:
            factors = factors / mean_factor
    if mean_value.ndim == 1:
        mean_value = mean_value.view(1, 1, -1)
    return mean_value * factors[..., None]


def apply_ls_gradient(
    ls_field: np.ndarray,
    depth_m: np.ndarray,
    k_field,
    model_cfg: dict,
    ls_cfg: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    cfg = _resolve_ls_gradient_config(model_cfg, ls_cfg)
    if not _gradient_enabled(cfg, rng, "ls_gradient"):
        return ls_field.astype(np.float32, copy=False)

    height, width = depth_m.shape
    factors = _ls_gradient_factors_np(height, width, cfg, rng)
    factors = _weight_ls_gradient_by_opacity_np(
        factors,
        depth_m,
        k_field,
        cfg,
        rng,
    )
    if bool(cfg.get("normalize_to_mean", False)):
        factors = _normalize_gradient_factors_np(factors)
    if ls_field.shape == (3,):
        ls_field = ls_field.reshape(1, 1, 3)
    return (ls_field * factors[..., None]).astype(np.float32, copy=False)


def apply_ls_gradient_torch(
    ls_field: "torch.Tensor",
    depth_m: "torch.Tensor",
    k_field,
    model_cfg: dict,
    ls_cfg: dict,
    rng: np.random.Generator,
) -> "torch.Tensor":
    cfg = _resolve_ls_gradient_config(model_cfg, ls_cfg)
    if not _gradient_enabled(cfg, rng, "ls_gradient"):
        return ls_field

    height = int(depth_m.shape[-2])
    width = int(depth_m.shape[-1])
    factors = _ls_gradient_factors_torch(
        height,
        width,
        cfg,
        rng,
        device=depth_m.device,
        dtype=depth_m.dtype,
    )
    factors = _weight_ls_gradient_by_opacity_torch(
        factors,
        depth_m,
        k_field,
        cfg,
        rng,
    )
    if bool(cfg.get("normalize_to_mean", False)):
        factors = _normalize_gradient_factors_torch(factors)
    if ls_field.ndim == 1:
        ls_field = ls_field.view(1, 1, 3)
    return ls_field * factors.unsqueeze(-1)


def _resolve_ls_gradient_config(model_cfg: dict, ls_cfg: dict) -> dict:
    raw = None
    for source in (ls_cfg, model_cfg):
        for key in (
            "ls_gradient",
            "light_gradient",
            "airlight_gradient",
            "illumination_gradient",
        ):
            if key in source:
                raw = source[key]
                break
        if raw is not None:
            break
    if raw is None:
        return {"enabled": False}
    if isinstance(raw, bool):
        return {"enabled": raw}
    if not isinstance(raw, dict):
        raise ValueError("ls_gradient must be a boolean or object")
    return dict(raw)


def _gradient_enabled(
    cfg: dict,
    rng: np.random.Generator,
    name: str,
) -> bool:
    enabled = cfg.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{name}.enabled must be a boolean")
    if not enabled:
        return False
    probability = _sample_float(
        cfg.get("probability", 1.0),
        rng,
        f"{name}.probability",
    )
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    return bool(rng.random() < probability)


def _ls_gradient_factors_np(
    height: int,
    width: int,
    cfg: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    top = _sample_float(cfg.get("top_factor", 1.0), rng, "ls_gradient.top_factor")
    bottom = _sample_float(
        cfg.get("bottom_factor", 1.0),
        rng,
        "ls_gradient.bottom_factor",
    )
    if top < 0.0 or bottom < 0.0:
        raise ValueError("ls_gradient top/bottom factors must be non-negative")
    gamma = _sample_float(cfg.get("gamma", 1.0), rng, "ls_gradient.gamma")
    if gamma <= 0.0:
        raise ValueError(f"ls_gradient.gamma must be > 0, got {gamma}")

    axis = str(cfg.get("axis", "vertical")).lower()
    if axis in {"vertical", "y", "top_bottom", "top-to-bottom"}:
        coord = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        factors = top + (bottom - top) * np.power(coord, gamma)
        return np.broadcast_to(factors, (height, width)).astype(np.float32, copy=True)
    if axis in {"horizontal", "x", "left_right", "left-to-right"}:
        coord = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        factors = top + (bottom - top) * np.power(coord, gamma)
        return np.broadcast_to(factors, (height, width)).astype(np.float32, copy=True)
    raise ValueError("ls_gradient.axis must be 'vertical' or 'horizontal'")


def _ls_gradient_factors_torch(
    height: int,
    width: int,
    cfg: dict,
    rng: np.random.Generator,
    *,
    device,
    dtype,
) -> "torch.Tensor":
    top = _sample_float(cfg.get("top_factor", 1.0), rng, "ls_gradient.top_factor")
    bottom = _sample_float(
        cfg.get("bottom_factor", 1.0),
        rng,
        "ls_gradient.bottom_factor",
    )
    if top < 0.0 or bottom < 0.0:
        raise ValueError("ls_gradient top/bottom factors must be non-negative")
    gamma = _sample_float(cfg.get("gamma", 1.0), rng, "ls_gradient.gamma")
    if gamma <= 0.0:
        raise ValueError(f"ls_gradient.gamma must be > 0, got {gamma}")

    axis = str(cfg.get("axis", "vertical")).lower()
    if axis in {"vertical", "y", "top_bottom", "top-to-bottom"}:
        coord = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype).view(
            height,
            1,
        )
        return (top + (bottom - top) * torch.pow(coord, gamma)).expand(height, width)
    if axis in {"horizontal", "x", "left_right", "left-to-right"}:
        coord = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype).view(
            1,
            width,
        )
        return (top + (bottom - top) * torch.pow(coord, gamma)).expand(height, width)
    raise ValueError("ls_gradient.axis must be 'vertical' or 'horizontal'")


def _weight_ls_gradient_by_opacity_np(
    factors: np.ndarray,
    depth_m: np.ndarray,
    k_field,
    cfg: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    weight = _sample_float(
        cfg.get("fog_opacity_weight", 0.0),
        rng,
        "ls_gradient.fog_opacity_weight",
    )
    weight = float(np.clip(weight, 0.0, 1.0))
    if weight <= 0.0:
        return factors
    gamma = _sample_float(
        cfg.get("fog_opacity_gamma", 1.0),
        rng,
        "ls_gradient.fog_opacity_gamma",
    )
    if gamma <= 0.0:
        raise ValueError(f"ls_gradient.fog_opacity_gamma must be > 0, got {gamma}")

    k_map = broadcast_k_field(k_field, depth_m.shape[0], depth_m.shape[1])
    opacity = 1.0 - np.exp(-np.maximum(k_map, 0.0) * np.maximum(depth_m, 0.0))
    blend = (1.0 - weight) + weight * np.power(np.clip(opacity, 0.0, 1.0), gamma)
    return 1.0 + (factors - 1.0) * blend.astype(np.float32, copy=False)


def _weight_ls_gradient_by_opacity_torch(
    factors: "torch.Tensor",
    depth_m: "torch.Tensor",
    k_field,
    cfg: dict,
    rng: np.random.Generator,
) -> "torch.Tensor":
    weight = _sample_float(
        cfg.get("fog_opacity_weight", 0.0),
        rng,
        "ls_gradient.fog_opacity_weight",
    )
    weight = float(np.clip(weight, 0.0, 1.0))
    if weight <= 0.0:
        return factors
    gamma = _sample_float(
        cfg.get("fog_opacity_gamma", 1.0),
        rng,
        "ls_gradient.fog_opacity_gamma",
    )
    if gamma <= 0.0:
        raise ValueError(f"ls_gradient.fog_opacity_gamma must be > 0, got {gamma}")

    if torch.is_tensor(k_field):
        k_t = k_field.to(device=depth_m.device, dtype=depth_m.dtype)
    else:
        k_t = torch.tensor(k_field, device=depth_m.device, dtype=depth_m.dtype)
    opacity = 1.0 - torch.exp(
        -torch.clamp(k_t, min=0.0) * torch.clamp(depth_m, min=0.0)
    )
    blend = (1.0 - weight) + weight * torch.pow(
        torch.clamp(opacity, 0.0, 1.0),
        gamma,
    )
    return 1.0 + (factors - 1.0) * blend


def _normalize_gradient_factors_np(factors: np.ndarray) -> np.ndarray:
    mean_factor = float(np.mean(factors))
    if mean_factor <= 0.0:
        return factors.astype(np.float32, copy=False)
    return (factors / mean_factor).astype(np.float32, copy=False)


def _normalize_gradient_factors_torch(factors: "torch.Tensor") -> "torch.Tensor":
    mean_factor = factors.mean()
    if float(mean_factor.item()) <= 0.0:
        return factors
    return factors / mean_factor


def _sanitize_depth_np(depth_m: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(depth, 0.0).astype(np.float32, copy=False)


def _sanitize_depth_torch(depth_m: "torch.Tensor") -> "torch.Tensor":
    depth = torch.nan_to_num(
        depth_m.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    return torch.clamp(depth, min=0.0)


def _resolve_scene_illumination_config(model_cfg: dict) -> dict:
    raw = model_cfg.get("scene_illumination")
    if raw is None:
        raw = {"enabled": False}
    if isinstance(raw, bool):
        raw = {"enabled": raw}
    if not isinstance(raw, dict):
        raise ValueError("scene_illumination must be a boolean or object")
    return deep_merge(dict(DEFAULT_SCENE_ILLUMINATION_CONFIG), raw)


def apply_scene_illumination_np(
    rgb_lin: np.ndarray,
    depth_m: np.ndarray,
    k_field,
    model_cfg: dict,
    rng: np.random.Generator,
    sky_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Darken pre-fog scene radiance for gloomy illumination conditions."""
    height, width = depth_m.shape
    cfg = _resolve_scene_illumination_config(model_cfg)
    enabled = cfg.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("scene_illumination.enabled must be a boolean")
    if not enabled:
        return (
            np.asarray(rgb_lin, dtype=np.float32),
            np.zeros((height, width), dtype=np.float32),
        )

    global_ev = _sample_float(
        cfg.get("global_ev", 0.0),
        rng,
        "scene_illumination.global_ev",
    )
    near_ev = _sample_float(
        cfg.get("near_ev", 0.0),
        rng,
        "scene_illumination.near_ev",
    )
    decay_depth = max(
        _sample_float(
            cfg.get("near_decay_depth_m", 15.0),
            rng,
            "scene_illumination.near_decay_depth_m",
        ),
        1e-6,
    )
    fog_ev = _sample_float(
        cfg.get("fog_coupled_ev", 0.0),
        rng,
        "scene_illumination.fog_coupled_ev",
    )

    depth = _sanitize_depth_np(depth_m)
    k_map = np.maximum(broadcast_k_field(k_field, height, width), 0.0)
    near_weight = np.exp(-depth / decay_depth).astype(np.float32, copy=False)
    fog_opacity = 1.0 - np.exp(-k_map * depth)
    ev_map = (
        global_ev + near_ev * near_weight + fog_ev * np.clip(fog_opacity, 0.0, 1.0)
    ).astype(np.float32, copy=False)

    if sky_mask is not None:
        sky = np.asarray(sky_mask, dtype=np.float32)
        if sky.shape != (height, width):
            raise ValueError(
                f"sky_mask must have shape ({height}, {width}); got {sky.shape}"
            )
        sky_weight = float(
            np.clip(
                _sample_float(
                    cfg.get("sky_weight", 0.0),
                    rng,
                    "scene_illumination.sky_weight",
                ),
                0.0,
                1.0,
            )
        )
        ev_map *= 1.0 - np.clip(sky, 0.0, 1.0) * (1.0 - sky_weight)

    min_scale = max(
        _sample_float(
            cfg.get("min_radiance_scale", 0.08),
            rng,
            "scene_illumination.min_radiance_scale",
        ),
        0.0,
    )
    scale = np.maximum(np.exp2(-ev_map), min_scale).astype(np.float32, copy=False)
    radiance = np.nan_to_num(
        np.asarray(rgb_lin, dtype=np.float32),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    return (radiance * scale[..., None]).astype(np.float32, copy=False), ev_map


def _apply_scene_airlight_dampening_np(
    ls_field: np.ndarray,
    ev_map: np.ndarray,
    model_cfg: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    cfg = _resolve_scene_illumination_config(model_cfg)
    if not bool(cfg.get("enabled", False)):
        return ls_field.astype(np.float32, copy=False)
    ratio = _sample_float(
        cfg.get("airlight_ev_ratio", 0.0),
        rng,
        "scene_illumination.airlight_ev_ratio",
    )
    if ratio <= 0.0:
        return ls_field.astype(np.float32, copy=False)
    scale = np.exp2(-np.maximum(ev_map, 0.0) * ratio).astype(np.float32, copy=False)
    ls_map = broadcast_ls_field(ls_field, ev_map.shape[0], ev_map.shape[1])
    return np.clip(ls_map * scale[..., None], 0.0, 1.0).astype(np.float32, copy=False)


def apply_fog(
    rgb: np.ndarray, depth_m: np.ndarray, k_field: np.ndarray, ls_field: np.ndarray
) -> np.ndarray:
    depth = _sanitize_depth_np(depth_m)
    k_arr = np.maximum(np.asarray(k_field, dtype=np.float32), 0.0)
    t = np.exp(-k_arr * depth)
    t = t[..., None]
    return (
        np.asarray(rgb, dtype=np.float32) * t
        + np.asarray(ls_field, dtype=np.float32) * (1.0 - t)
    ).astype(np.float32, copy=False)


def apply_fog_torch(
    rgb: "torch.Tensor",
    depth_m: "torch.Tensor",
    k_field,
    ls_field,
) -> "torch.Tensor":
    if not torch.is_tensor(k_field):
        k_field = torch.tensor(k_field, device=rgb.device, dtype=rgb.dtype)
    if not torch.is_tensor(ls_field):
        ls_field = torch.tensor(ls_field, device=rgb.device, dtype=rgb.dtype)
    depth = _sanitize_depth_torch(depth_m).to(device=rgb.device, dtype=rgb.dtype)
    k_field = torch.clamp(k_field.to(device=rgb.device, dtype=rgb.dtype), min=0.0)
    t = torch.exp(-k_field * depth)
    if t.ndim in (2, 3):
        t = t.unsqueeze(-1)
    return rgb * t + ls_field * (1.0 - t)


def select_model(config: dict, rng: np.random.Generator) -> str:
    selection = config.get("selection")
    if selection is None:
        if "fog_model" in config:
            return config["fog_model"]
        return "uniform"
    mode = selection.get("mode", "fixed")
    if mode == "fixed":
        return selection.get("model", "uniform")
    if mode == "weighted":
        weights = selection.get("weights", {})
        if not weights:
            raise ValueError("selection.weights must be provided for weighted mode")
        names = list(weights.keys())
        probs = np.array([weights[name] for name in names], dtype=np.float32)
        probs = probs / probs.sum()
        return str(rng.choice(names, p=probs))
    raise ValueError(f"Unsupported selection mode: {mode}")


def resolve_model_config(model_name: str, models_cfg: dict) -> dict:
    base = DEFAULT_MODEL_CONFIGS.get(model_name, {})
    override = models_cfg.get(model_name, {})
    return deep_merge(base, override)


def uses_estimated_airlight(al_spec) -> bool:
    return al_spec is None or al_spec in AIRLIGHT_METHODS


def broadcast_k_field(k_field: Any, height: int, width: int) -> np.ndarray:
    """Return ``k_field`` as a ``(H, W)`` float32 map (broadcasting if scalar)."""
    arr = np.asarray(k_field, dtype=np.float32)
    if arr.ndim == 0:
        return np.broadcast_to(arr, (height, width)).astype(np.float32, copy=True)
    if arr.shape == (height, width):
        return arr.astype(np.float32, copy=False)
    raise ValueError(
        f"k_field must be scalar or shape ({height}, {width}); got {arr.shape}"
    )


def broadcast_ls_field(ls_field: Any, height: int, width: int) -> np.ndarray:
    """Return ``ls_field`` as a ``(H, W, 3)`` float32 map (broadcasting if needed)."""
    arr = np.asarray(ls_field, dtype=np.float32)
    if arr.shape == (3,):
        return np.broadcast_to(arr, (height, width, 3)).astype(np.float32, copy=True)
    if arr.shape == (1, 1, 3):
        return np.broadcast_to(arr, (height, width, 3)).astype(np.float32, copy=True)
    if arr.shape == (height, width, 3):
        return arr.astype(np.float32, copy=False)
    raise ValueError(
        f"ls_field must have shape (3,), (1, 1, 3), or "
        f"({height}, {width}, 3); got {arr.shape}"
    )


def apply_model(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    model_name: str,
    model_cfg: dict,
    rng: np.random.Generator,
    contrast_threshold_default: float,
    estimated_airlight: np.ndarray,
    sky_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Apply a fog model to ``rgb``.

    Returns:
        Tuple ``(foggy, k_mean, ls_base, k_map, ls_map)``:

        * ``foggy``: ``(H, W, 3)`` foggy RGB image.
        * ``k_mean``: scalar base scattering coefficient (the map mean when
          heterogeneous-k normalization is enabled; for filenames/logs).
        * ``ls_base``: ``(3,)`` base atmospheric light (for filenames/logs).
        * ``k_map``: ``(H, W)`` β-field actually used (broadcast for uniform).
        * ``ls_map``: ``(H, W, 3)`` L_s-field actually used (broadcast for uniform).
    """
    if model_name not in DEFAULT_MODEL_CONFIGS:
        raise ValueError(f"Unsupported fog model: {model_name}")
    k_mean, _visibility, _contrast_threshold = resolve_scattering_coefficient(
        model_cfg,
        rng,
        contrast_threshold_default,
    )

    al_spec = model_cfg.get("atmospheric_light", "from_sky")
    airlight_is_estimated = uses_estimated_airlight(al_spec)
    if airlight_is_estimated:
        raw_ls_base = normalize_atmospheric_light(estimated_airlight)
    else:
        sampled_al = sample_value(al_spec, rng)
        raw_ls_base = normalize_atmospheric_light(np.asarray(sampled_al))

    height, width = depth_m.shape
    ls_base = dampen_airlight(
        raw_ls_base,
        k_mean,
        model_cfg,
        rng,
        _contrast_threshold,
        estimated_airlight=airlight_is_estimated,
    )

    if model_name in ("heterogeneous_k", "heterogeneous_k_ls"):
        k_cfg = model_cfg.get("k_hetero", {})
        k_scales = resolve_scales(k_cfg, height, width, rng)
        k_noise = perlin_fbm(height, width, k_scales, rng)
        k_noise = prepare_noise_field(k_noise, k_cfg, rng)
        min_factor = float(sample_value(k_cfg.get("min_factor", 1.0), rng))
        max_factor = float(sample_value(k_cfg.get("max_factor", 1.0), rng))
        k_field = modulate_with_noise(
            np.array([k_mean], dtype=np.float32),
            k_noise,
            min_factor,
            max_factor,
            bool(k_cfg.get("normalize_to_mean", False)),
        )[..., 0]
    else:
        k_field = k_mean

    if model_name in ("heterogeneous_ls", "heterogeneous_k_ls"):
        ls_cfg = model_cfg.get("ls_hetero", {})
        ls_scales = resolve_scales(ls_cfg, height, width, rng)
        ls_noise = perlin_fbm(height, width, ls_scales, rng)
        ls_noise = prepare_noise_field(ls_noise, ls_cfg, rng)
        min_factor = float(sample_value(ls_cfg.get("min_factor", 1.0), rng))
        max_factor = float(sample_value(ls_cfg.get("max_factor", 1.0), rng))
        ls_field = modulate_with_noise(
            ls_base,
            ls_noise,
            min_factor,
            max_factor,
            bool(ls_cfg.get("normalize_to_mean", False)),
        )
        ls_field = apply_ls_gradient(
            ls_field,
            depth_m,
            k_field,
            model_cfg,
            ls_cfg,
            rng,
        )
        ls_field = np.clip(ls_field, 0.0, 1.0)
    else:
        ls_field = ls_base.reshape(1, 1, 3)

    scene_rgb, ev_map = apply_scene_illumination_np(
        rgb,
        depth_m,
        k_field,
        model_cfg,
        rng,
        sky_mask=sky_mask,
    )
    ls_field = _apply_scene_airlight_dampening_np(ls_field, ev_map, model_cfg, rng)
    foggy = apply_fog(scene_rgb, depth_m, k_field, ls_field)
    k_map = broadcast_k_field(k_field, height, width)
    ls_map = broadcast_ls_field(ls_field, height, width)
    return foggy, k_mean, ls_base, k_map, ls_map
