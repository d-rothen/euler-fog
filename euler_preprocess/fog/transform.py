from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from euler_preprocess.common.device import (
    configure_device,
    iter_batches,
    torch_generator_for_index,
)
from euler_preprocess.common.intrinsics import (
    extract_intrinsics,
    planar_to_radial_depth,
    planar_to_radial_depth_torch,
)
from euler_preprocess.common.io import load_json
from euler_preprocess.common.logging import get_logger, progress_bar
from euler_preprocess.common.noise import perlin_fbm_torch
from euler_preprocess.common.normalize import (
    _is_chw,
    _to_numpy,
    normalize_depth,
    normalize_rgb,
    normalize_rgb_torch,
    normalize_sky_mask,
)
from euler_preprocess.common.output import LegacyOutputBackend, OutputSlotSpec
from euler_preprocess.common.sampling import deep_merge, format_value, sample_value
from euler_preprocess.common.transform import Transform
from euler_preprocess.fog.atmospheric_light import AtmosphericLightResolver
from euler_preprocess.fog.augmentations import (
    FogAugmentationConfig,
    FogAugmentationSpec,
    parse_fog_augmentations,
)
from euler_preprocess.fog.capture import CaptureArtifactPipeline
from euler_preprocess.fog.logging import log_config
from euler_preprocess.fog.models import (
    AIRLIGHT_METHODS,
    DEFAULT_CONTRAST_THRESHOLD,
    DEFAULT_MODEL_CONFIGS,
    apply_ls_gradient_torch,
    modulate_with_noise_torch,
    prepare_noise_field_torch,
    render_fog_fields_torch,
    resolve_scattering_coefficient,
    resolve_model_config,
    resolve_scales,
    select_model,
)
from euler_preprocess.fog.pipeline import (
    FogPipelineResult,
    FogProcessingPipeline,
)
from euler_loading.loaders.cpu.generic import (
    write_map_2d as _write_map_2d,
    write_map_3d as _write_map_3d,
)

try:
    from ds_crawler import EULER_LAYOUT_ADDON, build_layout_addon
except ImportError:  # pragma: no cover - compatibility with older ds-crawler
    EULER_LAYOUT_ADDON = "euler_layout"

    def build_layout_addon(**kwargs):
        payload: dict[str, Any] = {
            "version": kwargs.get("version", "1.0"),
            "sample_axis": {
                "name": kwargs["sample_axis_name"],
                "location": kwargs["sample_axis_location"],
            },
        }
        family = kwargs.get("family")
        if family is not None:
            payload["family"] = family
        variant_axis_name = kwargs.get("variant_axis_name")
        if variant_axis_name is not None:
            payload["variant_axis"] = {
                "name": variant_axis_name,
                "location": kwargs.get("variant_axis_location", "file_id"),
            }
        derived_from = kwargs.get("derived_from")
        if derived_from is not None:
            payload["derived_from"] = dict(derived_from)
        return payload


SCATTERING_COEFFICIENT_SLOT = "scattering_coefficient"
ATMOSPHERIC_LIGHT_SLOT = "atmospheric_light"

_SCENARIO_PROFILE_KEYS = (
    "scenario_profiles",
    "scene_condition_profiles",
    "condition_profiles",
)
_SCENARIO_PROFILE_METADATA_KEYS = {
    "name",
    "id",
    "description",
    "weight",
    "probability",
    "profile_weight",
    "config",
    "steps",
    "step_count",
    "progressive_weight",
    "max_weight",
    "attribute_key",
    "file_id_hierarchy_name",
}
_SCENARIO_CONTROL_KEYS = {
    "model",
    "model_name",
    "fog_model",
    "clear_weather",
    "model_overrides",
    "fog_model_overrides",
    "model_config",
    "airlight_method",
}


_GPU_BATCH_SCOPE_VALUES = {"sample", "batch"}
_CONFIG_MODE_VALUES = {"sample", "sampled", "random", "default", "legacy"}
_PROGRESSIVE_ATTRIBUTE_KEY = "fog_progression"
_PROGRESSIVE_FILE_ID_HIERARCHY_NAME = "file_id"
_PROGRESSIVE_EPSILON = 1e-6
_PROGRESSIVE_UNIT_INTERVAL_KEYS = {
    "probability",
    "fog_opacity_weight",
    "highlight_protection",
    "center_weight",
    "black_noise_floor",
}
_PROGRESSIVE_NON_NEGATIVE_KEYS = {
    "beta",
    "scattering_coefficient",
    "strength",
    "min_factor",
    "max_factor",
    "top_factor",
    "bottom_factor",
    "contrast",
    "noise_contrast",
    "smooth_sigma",
    "smoothing_sigma",
    "blur_sigma",
    "smooth_sigma_fraction",
    "smoothing_sigma_fraction",
    "blur_sigma_fraction",
    "read_noise_electrons",
    "read_noise_sigma",
    "fixed_pattern_sigma",
    "row_noise_sigma",
    "column_noise_sigma",
    "hot_pixel_probability",
    "dead_pixel_probability",
}
_PROGRESSIVE_POSITIVE_KEYS = {
    "visibility_m",
    "reference_visibility_m",
    "reference_beta",
    "reference_scattering_coefficient",
    "gamma",
    "fog_opacity_gamma",
    "correlation_length_fraction",
    "min_scale_fraction",
    "max_scale_fraction",
    "iso",
    "base_iso",
    "resize_scale",
}
_PROGRESSIVE_MIN_MAX_PAIRS = (("min_factor", "max_factor"),)


@dataclass(frozen=True)
class RenderPlan:
    model_name: str
    model_cfg: dict[str, Any]
    airlight_method: str | None = None
    capture_artifacts: CaptureArtifactPipeline | None = None
    scenario_name: str | None = None
    clear_weather: bool = False


@dataclass(frozen=True)
class ProgressiveScenarioStep:
    profile: dict[str, Any]
    scenario_index: int
    scenario_name: str
    id: str
    step_index: int
    steps: int
    weight: float
    max_weight: float


@dataclass(frozen=True)
class OutputVariantSpec:
    id: str
    attribute_key: str
    file_id_hierarchy_name: str | None
    attributes: dict[str, Any]


def _freeze_distribution_specs(value: Any, rng: np.random.Generator) -> Any:
    """Resolve distribution specs while preserving ordinary config structure."""
    if isinstance(value, dict):
        if "dist" in value:
            return sample_value(value, rng)
        return {
            key: _freeze_distribution_specs(child, rng)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_freeze_distribution_specs(child, rng) for child in value]
    if isinstance(value, tuple):
        return tuple(_freeze_distribution_specs(child, rng) for child in value)
    return value


def _sanitize_variant_identifier(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._-")
    return value or "variant"


def _is_progressive_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value,
        bool,
    )


def _is_numeric_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(
        _is_progressive_number(item) for item in value
    )


# Use the canonical euler-loading ``generic.map_2d`` / ``map_3d`` modality
# annotations.  Auxiliary outputs are written as ``.npy`` files in the
# layout the matching loader expects (``map_2d`` → ``(H, W)``,
# ``map_3d`` → ``(C, H, W)`` after the writer's transpose).
_SCATTERING_INDEX_OVERLAY: dict[str, Any] = {
    "name": "scattering_coefficient",
    "type": "map_2d",
    "euler_train": {"used_as": "target"},
    "euler_loading": {
        "loader": "generic",
        "function": "map_2d",
    },
    "meta": {},
}

_ATMOSPHERIC_LIGHT_INDEX_OVERLAY: dict[str, Any] = {
    "name": "atmospheric_light",
    "type": "map_3d",
    "euler_train": {"used_as": "target"},
    "euler_loading": {
        "loader": "generic",
        "function": "map_3d",
    },
    "meta": {},
}

try:
    import torch
except ImportError:
    torch = None


class FogTransform(Transform):
    """Generate foggy versions of images from RGB + depth + semantic segmentation samples.

    Accepts an iterable of sample dicts (compatible with euler-loading's
    MultiModalDataset). Each sample must contain at minimum:
        "rgb":                      np.ndarray (H, W, 3)
        "depth":                    np.ndarray (H, W), float in meters
        "semantic_segmentation":    np.ndarray (H, W), boolean sky mask
        "id":         str
        "intrinsics": dict -- hierarchical modality containing ``"intrinsics"``
                      key mapping to a (3, 3) camera intrinsics matrix *K*.
                      When present, planar (z-buffer) depth is converted to
                      radial (Euclidean) depth before fog is applied.

    Optional:
        "full_id":  str -- hierarchical id from euler-loading (e.g.
                    "/Scene02/30-deg-right/Camera_0/00000").  When present,
                    the parent segments are used as subdirectories under the
                    model output folder so the dataset structure is preserved.
    """

    REQUIRED_MODALITIES: ClassVar[set[str]] = {"rgb", "depth", "semantic_segmentation"}
    REQUIRED_HIERARCHICAL_MODALITIES: ClassVar[set[str]] = set()
    SOURCE_MODALITY: ClassVar[str] = "rgb"
    OUTPUT_SLOT: ClassVar[str] = "rgb"
    OUTPUT_SLOTS: ClassVar[tuple[str, ...]] = (
        "rgb",
        SCATTERING_COEFFICIENT_SLOT,
        ATMOSPHERIC_LIGHT_SLOT,
    )
    OUTPUT_SLOT_SPECS: ClassVar[dict[str, OutputSlotSpec]] = {
        SCATTERING_COEFFICIENT_SLOT: OutputSlotSpec(
            source_modality="rgb",
            writer=_write_map_2d,
            index_overlay=_SCATTERING_INDEX_OVERLAY,
            output_extension=".npy",
        ),
        ATMOSPHERIC_LIGHT_SLOT: OutputSlotSpec(
            source_modality="rgb",
            writer=_write_map_3d,
            index_overlay=_ATMOSPHERIC_LIGHT_INDEX_OVERLAY,
            output_extension=".npy",
        ),
    }

    def __init__(
        self,
        config_path: str,
        out_path: str,
        suffix: str = "",
        output_backend: Any | None = None,
        output_backends: dict[str, Any] | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        if output_backends is not None:
            if "rgb" not in output_backends:
                raise ValueError(
                    "output_backends must contain the primary 'rgb' slot"
                )
            self.output_backends = dict(output_backends)
            self.output_backend = self.output_backends["rgb"]
        else:
            self.output_backend = output_backend or LegacyOutputBackend(out_path)
            self.output_backends = {"rgb": self.output_backend}
        self.out_path = self.output_backend.root
        self.suffix = suffix or ""

        self.config = load_json(self.config_path)
        self.models_cfg = (
            self.config.get("models") or self.config.get("fog_models") or {}
        )
        self.device = str(self.config.get("device", "cpu"))
        self.gpu_batch_size = max(1, int(self.config.get("gpu_batch_size", 4)))
        self.seed = self.config.get("seed")
        self.base_rng = np.random.default_rng(self.seed)
        self.contrast_threshold_default = float(
            sample_value(
                self.config.get("contrast_threshold", DEFAULT_CONTRAST_THRESHOLD),
                self.base_rng,
            )
        )
        self.depth_scale = float(self.config.get("depth_scale", 1.0))
        self.resize_depth_flag = bool(self.config.get("resize_depth", True))
        self.augmentation_config: FogAugmentationConfig = parse_fog_augmentations(
            self.config
        )
        self.augmentation_specs = list(self.augmentation_config.specs)
        self.scenario_profiles = self._parse_scenario_profiles(self.config)
        self.mode = self._parse_mode(self.config)
        self.progressive_scenario_steps = self._build_progressive_scenario_steps(
            self.config,
            self.scenario_profiles,
        )
        if self.mode == "progressive" and self.augmentation_specs:
            raise ValueError(
                "mode='progressive' cannot be combined with augmentations; "
                "progressive scenario profiles already emit variant outputs"
            )
        (
            self.gpu_scenario_scope,
            self.gpu_condition_parameter_scope,
        ) = self._parse_gpu_batching_config(self.config)
        self._configure_output_layout_metadata()
        self._written_configs: set[str] = set()
        self.torch_device = None
        self.use_gpu = False
        self.logger = get_logger()
        self._configure_device()
        log_config(
            self.logger,
            self.config,
            str(self.config_path),
            str(self.out_path),
            self.device,
            self.use_gpu,
            torch is not None,
            str(self.torch_device) if self.torch_device else None,
            self.gpu_batch_size,
            0,
            self.depth_scale,
            self.resize_depth_flag,
            self.seed,
            self.contrast_threshold_default,
        )

        airlight_method = self.config.get("airlight")
        if airlight_method is None:
            raise ValueError(
                "Config must specify 'airlight' key. "
                f"Supported values: {AIRLIGHT_METHODS}"
            )
        self.atmospheric_light = AtmosphericLightResolver(
            airlight_method,
            dcp_heuristic_config=self.config.get("dcp_heuristic", {}),
            logger=self.logger,
        )
        self.airlight_method = self.atmospheric_light.method
        self.dcp_heuristic_kwargs = self.atmospheric_light.dcp_heuristic_kwargs
        self.airlight_estimator = self.atmospheric_light.estimator
        self.airlight_estimator_torch = self.atmospheric_light.estimator_torch
        self.pipeline = FogProcessingPipeline.from_config(
            self.config,
            atmospheric_light=self.atmospheric_light,
            contrast_threshold_default=self.contrast_threshold_default,
        )

    def run(self, samples: Iterable[dict]) -> list[Path]:
        """Run the fog transform. Alias for :meth:`generate_fog`."""
        return self.generate_fog(samples)

    def generate_fog(self, samples: Iterable[dict]) -> list[Path]:
        """Generate fog on the given samples.

        Args:
            samples: Iterable of dicts, each containing "rgb", "depth",
                     "semantic_segmentation", and "id" keys.

        Returns:
            List of output file paths.
        """
        if self.use_gpu:
            return self._generate_fog_gpu(samples)
        if self.mode == "progressive":
            return self._generate_fog_cpu_progressive(samples)
        return self._generate_fog_cpu(samples)

    def _configure_device(self) -> None:
        self.torch_device, self.use_gpu = configure_device(self.device)

    def _rng_for(self, sample_index: int, augmentation_index: int | None = None):
        if self.seed is not None:
            seed_parts: list[int] = [int(self.seed), int(sample_index)]
            if augmentation_index is not None:
                seed_parts.append(int(augmentation_index))
            return np.random.default_rng(np.random.SeedSequence(seed_parts))
        return self.base_rng

    def _rng_for_batch(self, batch_index: int):
        if self.seed is not None:
            return np.random.default_rng(
                np.random.SeedSequence(
                    [int(self.seed), int(batch_index), 1_000_003]
                )
            )
        return self.base_rng

    def _rng_for_progressive(
        self,
        sample_index: int,
        scenario_index: int,
        step_index: int,
    ):
        if self.seed is not None:
            return np.random.default_rng(
                np.random.SeedSequence(
                    [
                        int(self.seed),
                        int(sample_index),
                        2_000_003,
                        int(scenario_index),
                        int(step_index),
                    ]
                )
            )
        return self.base_rng

    def _get_airlight_estimator(self, method: str):
        return self.atmospheric_light.get_estimator(method)

    def _get_airlight_estimator_torch(self, method: str):
        return self.atmospheric_light.get_estimator_torch(method)

    def _estimate_airlight_np(
        self,
        rgb: np.ndarray,
        sky_mask: np.ndarray,
        *,
        sample_id: str | None,
        method: str | None = None,
    ) -> np.ndarray:
        return self.atmospheric_light.estimate_np(
            rgb,
            sky_mask,
            sample_id=sample_id,
            method=method,
        )

    def _estimate_airlight_torch(
        self,
        rgb_t: "torch.Tensor",
        sky_mask_t: "torch.Tensor",
        *,
        sample_id: str | None,
        method: str | None = None,
    ) -> "torch.Tensor":
        return self.atmospheric_light.estimate_torch(
            rgb_t,
            sky_mask_t,
            sample_id=sample_id,
            method=method,
        )

    def _resolve_augmented_model(
        self,
        augmentation: FogAugmentationSpec,
    ) -> tuple[str, dict]:
        base_cfg = resolve_model_config(augmentation.model_name, self.models_cfg)
        return augmentation.model_name, deep_merge(base_cfg, augmentation.model_overrides)

    def _parse_scenario_profiles(
        self,
        config: dict[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        for key in _SCENARIO_PROFILE_KEYS:
            if key not in config:
                continue
            raw = config[key]
            if raw is None:
                return ()
            if not isinstance(raw, (list, tuple)):
                raise ValueError(f"{key} must be a list")
            profiles: list[dict[str, Any]] = []
            for index, entry in enumerate(raw):
                if not isinstance(entry, dict):
                    raise ValueError(f"{key}[{index}] must be an object")
                profiles.append(dict(entry))
            return tuple(profiles)
        return ()

    def _parse_mode(self, config: dict[str, Any]) -> str:
        raw = config.get("mode", config.get("scenario_mode", "sample"))
        if raw is None:
            return "sample"
        if not isinstance(raw, str):
            raise ValueError("mode must be a string")
        mode = raw.strip().lower().replace("-", "_")
        if mode == "progressive":
            return mode
        if mode in _CONFIG_MODE_VALUES:
            return "sample"
        raise ValueError("mode must be 'progressive' or omitted")

    def _build_progressive_scenario_steps(
        self,
        config: dict[str, Any],
        profiles: tuple[dict[str, Any], ...],
    ) -> tuple[ProgressiveScenarioStep, ...]:
        if self.mode != "progressive":
            return ()
        if not profiles:
            raise ValueError("mode='progressive' requires scenario_profiles")

        default_steps = int(config.get("progressive_steps", config.get("steps", 1)))
        if default_steps < 1:
            raise ValueError("progressive_steps must be >= 1")

        steps: list[ProgressiveScenarioStep] = []
        seen_ids: set[str] = set()
        for scenario_index, profile in enumerate(profiles):
            raw_steps = profile.get(
                "steps",
                profile.get("step_count", default_steps),
            )
            step_count = int(raw_steps)
            if step_count < 1:
                raise ValueError(
                    f"scenario_profiles[{scenario_index}].steps must be >= 1"
                )

            max_weight = float(
                profile.get(
                    "progressive_weight",
                    profile.get("max_weight", profile.get("weight", 1.0)),
                )
            )
            if not np.isfinite(max_weight) or max_weight < 0.0:
                raise ValueError(
                    f"scenario_profiles[{scenario_index}].weight must be "
                    "a finite non-negative number in progressive mode"
                )

            scenario_name = self._scenario_name(profile) or f"scenario_{scenario_index:03d}"
            scenario_id = _sanitize_variant_identifier(scenario_name)
            for step_index in range(step_count + 1):
                weight = (
                    0.0
                    if step_count == 0
                    else max_weight * (float(step_index) / float(step_count))
                )
                step_id = _sanitize_variant_identifier(
                    f"{scenario_id}_w{format_value(weight)}"
                )
                if step_id in seen_ids:
                    step_id = _sanitize_variant_identifier(
                        f"{scenario_index:03d}_{scenario_id}_step_{step_index:03d}"
                    )
                seen_ids.add(step_id)
                steps.append(
                    ProgressiveScenarioStep(
                        profile=dict(profile),
                        scenario_index=scenario_index,
                        scenario_name=scenario_name,
                        id=step_id,
                        step_index=step_index,
                        steps=step_count,
                        weight=weight,
                        max_weight=max_weight,
                    )
                )
        return tuple(steps)

    def _parse_gpu_batching_config(
        self,
        config: dict[str, Any],
    ) -> tuple[str, str]:
        raw = config.get("gpu_batching", {})
        if raw is None or raw is False:
            raw = {}
        if raw is True:
            raw = {"scenario_scope": "batch"}
        if not isinstance(raw, dict):
            raise ValueError("gpu_batching must be a boolean or object")

        scenario_scope = str(
            raw.get("scenario_scope", config.get("gpu_scenario_scope", "sample"))
        ).lower()
        if scenario_scope not in _GPU_BATCH_SCOPE_VALUES:
            raise ValueError(
                "gpu_batching.scenario_scope must be 'sample' or 'batch'"
            )

        if "condition_parameter_scope" in raw:
            condition_scope = str(raw["condition_parameter_scope"]).lower()
        elif "sample_condition_once_per_batch" in raw:
            condition_scope = (
                "batch" if bool(raw["sample_condition_once_per_batch"]) else "sample"
            )
        else:
            condition_scope = "batch" if scenario_scope == "batch" else "sample"
        if condition_scope not in _GPU_BATCH_SCOPE_VALUES:
            raise ValueError(
                "gpu_batching.condition_parameter_scope must be 'sample' or 'batch'"
            )
        if condition_scope == "batch" and scenario_scope != "batch":
            raise ValueError(
                "gpu_batching.condition_parameter_scope='batch' requires "
                "scenario_scope='batch'"
            )
        return scenario_scope, condition_scope

    def _sample_scenario_profile(
        self,
        rng: np.random.Generator,
    ) -> dict[str, Any] | None:
        if not self.scenario_profiles:
            return None
        weights: list[float] = []
        for index, profile in enumerate(self.scenario_profiles):
            weight = float(
                profile.get(
                    "weight",
                    profile.get("profile_weight", profile.get("probability", 1.0)),
                )
            )
            if weight < 0.0:
                raise ValueError(
                    f"scenario_profiles[{index}].weight must be non-negative"
                )
            weights.append(weight)
        weights_arr = np.asarray(weights, dtype=np.float64)
        total = float(weights_arr.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("scenario_profiles must contain a positive weight")
        index = int(rng.choice(len(self.scenario_profiles), p=weights_arr / total))
        return self.scenario_profiles[index]

    def _scenario_payload(self, profile: dict[str, Any]) -> dict[str, Any]:
        raw = profile.get("config", profile)
        if not isinstance(raw, dict):
            raise ValueError("scenario profile config must be an object")
        return dict(raw)

    def _scenario_name(self, profile: dict[str, Any]) -> str | None:
        name = profile.get("name", profile.get("id"))
        if name is None and isinstance(profile.get("config"), dict):
            config = profile["config"]
            name = config.get("name", config.get("id"))
        return None if name is None else str(name)

    def _scenario_config_override(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in payload.items()
            if key not in _SCENARIO_PROFILE_METADATA_KEYS
            and key not in _SCENARIO_CONTROL_KEYS
        }

    def _scenario_airlight_method(self, payload: dict[str, Any]) -> str | None:
        method = payload.get("airlight_method", payload.get("airlight"))
        if method is None:
            return None
        if not isinstance(method, str):
            return None
        if method not in AIRLIGHT_METHODS:
            raise ValueError(
                f"scenario airlight_method must be one of {AIRLIGHT_METHODS}, "
                f"got {method!r}"
            )
        return method

    def _scenario_clear_weather(self, payload: dict[str, Any]) -> bool:
        clear_weather = payload.get("clear_weather", False)
        if not isinstance(clear_weather, bool):
            raise ValueError("scenario clear_weather must be a boolean")
        return clear_weather

    def _clear_weather_progressive_model_config(
        self,
        model_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the no-fog target used to ramp a progressive clear scenario.

        Sampled clear scenarios bypass the weather render entirely. Progressive
        mode still needs a numeric target to interpolate toward, so it uses zero
        scattering and disables scene illumination at the final step.
        """
        clear_cfg = copy.deepcopy(model_cfg)
        clear_cfg.pop("visibility_m", None)
        clear_cfg.pop("beta", None)
        clear_cfg["scattering_coefficient"] = 0.0
        clear_cfg["scene_illumination"] = {"enabled": False}
        return clear_cfg

    def _scenario_model_name(
        self,
        payload: dict[str, Any],
        effective_config: dict[str, Any],
        rng: np.random.Generator,
    ) -> str:
        for key in ("model", "model_name", "fog_model"):
            if key in payload:
                return str(sample_value(payload[key], rng))
        return select_model(effective_config, rng)

    def _scenario_model_overrides(self, payload: dict[str, Any]) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        for key in ("model_config", "model_overrides", "fog_model_overrides"):
            raw = payload.get(key)
            if raw is None:
                continue
            if not isinstance(raw, dict):
                raise ValueError(f"scenario {key} must be an object")
            overrides = deep_merge(overrides, raw)
        return overrides

    def _resolve_render_plan(
        self,
        rng: np.random.Generator,
        augmentation: FogAugmentationSpec | None = None,
        *,
        scenario: dict[str, Any] | None = None,
        sample_scenario: bool = True,
        freeze_sampled_parameters: bool = False,
    ) -> RenderPlan:
        if sample_scenario:
            scenario = self._sample_scenario_profile(rng)
        if scenario is None:
            effective_config = (
                _freeze_distribution_specs(self.config, rng)
                if freeze_sampled_parameters
                else self.config
            )
            models_cfg = (
                effective_config.get("models")
                or effective_config.get("fog_models")
                or {}
            )
            if augmentation is not None:
                base_cfg = resolve_model_config(augmentation.model_name, models_cfg)
                model_cfg = deep_merge(base_cfg, augmentation.model_overrides)
                if freeze_sampled_parameters:
                    model_cfg = _freeze_distribution_specs(model_cfg, rng)
                return RenderPlan(
                    model_name=augmentation.model_name,
                    model_cfg=model_cfg,
                    airlight_method=augmentation.airlight_method,
                    capture_artifacts=(
                        CaptureArtifactPipeline.from_config(effective_config)
                        if freeze_sampled_parameters
                        else None
                    ),
            )
            model_name = select_model(effective_config, rng)
            model_cfg = resolve_model_config(model_name, models_cfg)
            if freeze_sampled_parameters:
                model_cfg = _freeze_distribution_specs(model_cfg, rng)
            return RenderPlan(
                model_name=model_name,
                model_cfg=model_cfg,
                capture_artifacts=(
                    CaptureArtifactPipeline.from_config(effective_config)
                    if freeze_sampled_parameters
                    else None
                ),
            )

        payload = self._scenario_payload(scenario)
        effective_config = deep_merge(
            self.config,
            self._scenario_config_override(payload),
        )
        if freeze_sampled_parameters:
            effective_config = _freeze_distribution_specs(effective_config, rng)
        models_cfg = (
            effective_config.get("models")
            or effective_config.get("fog_models")
            or {}
        )
        scenario_overrides = self._scenario_model_overrides(payload)

        if augmentation is not None:
            model_name = augmentation.model_name
            model_cfg = resolve_model_config(model_name, models_cfg)
            model_cfg = deep_merge(model_cfg, scenario_overrides)
            model_cfg = deep_merge(model_cfg, augmentation.model_overrides)
            airlight_method = (
                augmentation.airlight_method or self._scenario_airlight_method(payload)
            )
        else:
            model_name = self._scenario_model_name(payload, effective_config, rng)
            model_cfg = resolve_model_config(model_name, models_cfg)
            model_cfg = deep_merge(model_cfg, scenario_overrides)
            airlight_method = self._scenario_airlight_method(payload)
        if freeze_sampled_parameters:
            model_cfg = _freeze_distribution_specs(model_cfg, rng)

        return RenderPlan(
            model_name=model_name,
            model_cfg=model_cfg,
            airlight_method=airlight_method,
            capture_artifacts=CaptureArtifactPipeline.from_config(effective_config),
            scenario_name=self._scenario_name(scenario),
            clear_weather=self._scenario_clear_weather(payload),
        )

    def _resolve_progressive_render_plan(
        self,
        rng: np.random.Generator,
        step: ProgressiveScenarioStep,
    ) -> RenderPlan:
        payload = self._scenario_payload(step.profile)
        base_effective_config = _freeze_distribution_specs(self.config, rng)
        frozen_payload = _freeze_distribution_specs(payload, rng)
        target_effective_config = deep_merge(
            base_effective_config,
            self._scenario_config_override(frozen_payload),
        )

        model_name = self._scenario_model_name(
            frozen_payload,
            target_effective_config,
            rng,
        )
        base_models_cfg = (
            base_effective_config.get("models")
            or base_effective_config.get("fog_models")
            or {}
        )
        target_models_cfg = (
            target_effective_config.get("models")
            or target_effective_config.get("fog_models")
            or {}
        )
        base_model_cfg = resolve_model_config(model_name, base_models_cfg)
        target_model_cfg = resolve_model_config(model_name, target_models_cfg)
        target_model_cfg = deep_merge(
            target_model_cfg,
            self._scenario_model_overrides(frozen_payload),
        )
        if self._scenario_clear_weather(frozen_payload):
            target_model_cfg = self._clear_weather_progressive_model_config(
                target_model_cfg
            )
        base_model_cfg = _freeze_distribution_specs(base_model_cfg, rng)
        target_model_cfg = _freeze_distribution_specs(target_model_cfg, rng)

        step_effective_config = self._sanitize_progressive_blended_config(
            self._progressive_blend_value(
                base_effective_config,
                target_effective_config,
                step.weight,
            )
        )
        model_cfg = self._sanitize_progressive_blended_config(
            self._progressive_model_config(
                base_model_cfg,
                target_model_cfg,
                step.weight,
                rng,
            )
        )
        airlight_method = (
            None
            if step.weight <= 0.0
            else self._scenario_airlight_method(frozen_payload)
        )

        return RenderPlan(
            model_name=model_name,
            model_cfg=model_cfg,
            airlight_method=airlight_method,
            capture_artifacts=CaptureArtifactPipeline.from_config(
                step_effective_config
            ),
            scenario_name=step.scenario_name,
        )

    def _progressive_model_config(
        self,
        base_cfg: dict[str, Any],
        target_cfg: dict[str, Any],
        weight: float,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        if weight <= 0.0:
            return copy.deepcopy(base_cfg)
        if np.isclose(weight, 1.0):
            return copy.deepcopy(target_cfg)

        cfg = self._progressive_blend_value(base_cfg, target_cfg, weight)
        base_beta, _base_visibility, _base_contrast = resolve_scattering_coefficient(
            base_cfg,
            rng,
            self.contrast_threshold_default,
        )
        target_beta, _target_visibility, _target_contrast = resolve_scattering_coefficient(
            target_cfg,
            rng,
            self.contrast_threshold_default,
        )
        beta = max(0.0, float(base_beta + (target_beta - base_beta) * weight))
        cfg.pop("visibility_m", None)
        cfg.pop("beta", None)
        cfg["scattering_coefficient"] = beta
        return cfg

    def _progressive_blend_value(
        self,
        base: Any,
        target: Any,
        weight: float,
        *,
        key: str | None = None,
    ) -> Any:
        if weight <= 0.0:
            return copy.deepcopy(base)
        if np.isclose(weight, 1.0):
            return copy.deepcopy(target)

        if _is_progressive_number(base) and _is_progressive_number(target):
            return float(base) + (float(target) - float(base)) * weight

        if _is_numeric_sequence(base) and _is_numeric_sequence(target):
            if len(base) == len(target):
                return [
                    self._progressive_blend_value(
                        base_item,
                        target_item,
                        weight,
                        key=key,
                    )
                    for base_item, target_item in zip(base, target)
                ]

        if isinstance(base, dict) and isinstance(target, dict):
            merged: dict[str, Any] = {}
            for child_key in sorted(set(base) | set(target)):
                if child_key not in target:
                    merged[child_key] = copy.deepcopy(base[child_key])
                    continue
                if child_key not in base:
                    neutral = self._progressive_neutral_value(
                        child_key,
                        target[child_key],
                    )
                    merged[child_key] = self._progressive_blend_value(
                        neutral,
                        target[child_key],
                        weight,
                        key=child_key,
                    )
                    continue
                merged[child_key] = self._progressive_blend_value(
                    base[child_key],
                    target[child_key],
                    weight,
                    key=child_key,
                )
            return merged

        if _is_progressive_number(target):
            neutral = self._progressive_neutral_number(key, float(target))
            return neutral + (float(target) - neutral) * weight

        if _is_numeric_sequence(target):
            return [
                self._progressive_blend_value(
                    self._progressive_neutral_number(key, float(item)),
                    item,
                    weight,
                    key=key,
                )
                for item in target
            ]

        return copy.deepcopy(target)

    def _sanitize_progressive_blended_config(
        self,
        value: Any,
        *,
        key: str | None = None,
    ) -> Any:
        if isinstance(value, dict):
            sanitized = {
                child_key: self._sanitize_progressive_blended_config(
                    child,
                    key=child_key,
                )
                for child_key, child in value.items()
            }
            for min_key, max_key in _PROGRESSIVE_MIN_MAX_PAIRS:
                min_value = sanitized.get(min_key)
                max_value = sanitized.get(max_key)
                if (
                    _is_progressive_number(min_value)
                    and _is_progressive_number(max_value)
                    and float(max_value) < float(min_value)
                ):
                    sanitized[max_key] = float(min_value)
            return sanitized

        if isinstance(value, list):
            return [
                self._sanitize_progressive_blended_config(item, key=key)
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                self._sanitize_progressive_blended_config(item, key=key)
                for item in value
            )

        if not _is_progressive_number(value):
            return copy.deepcopy(value)

        numeric = float(value)
        if not np.isfinite(numeric):
            label = key or "value"
            raise ValueError(f"progressive blended {label} must be finite")
        if key in _PROGRESSIVE_UNIT_INTERVAL_KEYS:
            return float(np.clip(numeric, 0.0, 1.0))
        if key in _PROGRESSIVE_POSITIVE_KEYS:
            return max(numeric, _PROGRESSIVE_EPSILON)
        if key in _PROGRESSIVE_NON_NEGATIVE_KEYS:
            return max(numeric, 0.0)
        return copy.deepcopy(value)

    def _progressive_neutral_value(self, key: str | None, target: Any) -> Any:
        if _is_progressive_number(target):
            return self._progressive_neutral_number(key, float(target))
        if _is_numeric_sequence(target):
            return [
                self._progressive_neutral_number(key, float(item))
                for item in target
            ]
        if isinstance(target, dict):
            return {
                child_key: self._progressive_neutral_value(child_key, child)
                for child_key, child in target.items()
            }
        return None

    def _progressive_neutral_number(
        self,
        key: str | None,
        target: float,
    ) -> float:
        del target
        if key in {
            "gain",
            "exposure_gain",
            "red_chroma_gain",
            "blue_chroma_gain",
            "min_factor",
            "max_factor",
            "top_factor",
            "bottom_factor",
            "gamma",
            "max_gain",
            "manual_gain_weight",
        }:
            return 1.0
        if key == "quality":
            return 100.0
        if key == "iso":
            return 100.0
        return 0.0

    def _resolve_gpu_batch_render_plan(self, batch_index: int) -> RenderPlan | None:
        if self.gpu_scenario_scope != "batch":
            return None
        rng = self._rng_for_batch(batch_index)
        scenario = self._sample_scenario_profile(rng)
        return self._resolve_render_plan(
            rng,
            scenario=scenario,
            sample_scenario=False,
            freeze_sampled_parameters=(
                self.gpu_condition_parameter_scope == "batch"
            ),
        )

    def _source_extension(self, sample: dict, backend: Any | None = None) -> str:
        meta = sample.get("meta")
        source_modality = (
            getattr(backend, "source_modality", None) or self.SOURCE_MODALITY or "rgb"
        )
        if isinstance(meta, dict):
            source_meta = meta.get(source_modality)
            if isinstance(source_meta, dict) and "path" in source_meta:
                suffix = Path(str(source_meta["path"])).suffix
                if suffix:
                    return suffix
        return ".png"

    def _layout_family(self) -> str | None:
        raw = self.config.get("dataset_family")
        return raw if isinstance(raw, str) and raw else None

    def _augmentation_hierarchy_separator(self, backend: Any) -> str:
        separator = getattr(getattr(backend, "dataset_writer", None), "_separator", None)
        if isinstance(separator, str) and separator and separator != "+":
            return separator
        return ":"

    def _variant_layout_config(self) -> tuple[str | None, str] | None:
        if self.augmentation_specs:
            return (
                self.augmentation_config.file_id_hierarchy_name,
                self.augmentation_config.attribute_key,
            )
        if self.progressive_scenario_steps:
            return (
                _PROGRESSIVE_FILE_ID_HIERARCHY_NAME,
                _PROGRESSIVE_ATTRIBUTE_KEY,
            )
        return None

    def _configure_output_layout_metadata(self) -> None:
        """Declare fog outputs as variants grouped by source sample id."""
        layout_config = self._variant_layout_config()
        if layout_config is None:
            return

        sample_axis_name, attribute_key = layout_config
        if not sample_axis_name:
            return

        for backend in self.output_backends.values():
            if not getattr(backend, "is_source_backed", False):
                continue

            separator = self._augmentation_hierarchy_separator(backend)
            set_separator = getattr(backend, "set_hierarchy_separator", None)
            if callable(set_separator):
                set_separator(separator)

            layout = build_layout_addon(
                family=self._layout_family(),
                sample_axis_name=sample_axis_name,
                sample_axis_location="hierarchy",
                variant_axis_name=attribute_key,
                variant_axis_location="file_id",
                derived_from={
                    "source_modality": getattr(backend, "source_modality", "rgb"),
                    "source_id_attribute": f"{attribute_key}.source_id",
                    "source_full_id_attribute": f"{attribute_key}.source_full_id",
                },
            )
            add_head_addon = getattr(backend, "add_head_addon", None)
            if callable(add_head_addon):
                add_head_addon(EULER_LAYOUT_ADDON, layout)

    def _file_id_hierarchy_key(
        self,
        sample_id: str,
        backend: Any,
        hierarchy_name: str | None,
    ) -> str:
        name = hierarchy_name
        separator = self._augmentation_hierarchy_separator(backend)
        if name and separator:
            return f"{name}{separator}{sample_id}"
        return sample_id

    def _variant_full_id(
        self,
        sample: dict,
        variant_id: str,
        backend: Any,
        hierarchy_name: str | None,
    ) -> str:
        sample_id = str(sample.get("id", "?"))
        full_id = str(sample.get("full_id") or f"/{sample_id}")
        parts = [part for part in full_id.split("/") if part]
        parent_parts = parts[:-1] if parts else []
        file_id_key = self._file_id_hierarchy_key(
            sample_id,
            backend,
            hierarchy_name,
        )
        return "/" + "/".join(parent_parts + [file_id_key, variant_id])

    def _output_variant_attributes(
        self,
        sample: dict,
        variant: OutputVariantSpec,
        *,
        model_name: str,
        beta: float,
        airlight: np.ndarray,
    ) -> dict[str, Any]:
        source_id = str(sample.get("id", "?"))
        payload = {
            "id": variant.id,
            "source_id": source_id,
            "source_full_id": str(sample.get("full_id") or f"/{source_id}"),
            "model": model_name,
            "scattering_coefficient": float(beta),
            "atmospheric_light": [
                float(v) for v in np.asarray(airlight, dtype=np.float32).reshape(-1)[:3]
            ],
            **variant.attributes,
        }
        return {variant.attribute_key: payload}

    def _augmentation_output_variant(
        self,
        augmentation: FogAugmentationSpec,
    ) -> OutputVariantSpec:
        return OutputVariantSpec(
            id=augmentation.id,
            attribute_key=self.augmentation_config.attribute_key,
            file_id_hierarchy_name=self.augmentation_config.file_id_hierarchy_name,
            attributes=dict(augmentation.attributes),
        )

    def _progressive_output_variant(
        self,
        step: ProgressiveScenarioStep,
    ) -> OutputVariantSpec:
        return OutputVariantSpec(
            id=step.id,
            attribute_key=_PROGRESSIVE_ATTRIBUTE_KEY,
            file_id_hierarchy_name=_PROGRESSIVE_FILE_ID_HIERARCHY_NAME,
            attributes={
                "scenario": step.scenario_name,
                "scenario_index": step.scenario_index,
                "step": step.step_index,
                "steps": step.steps,
                "weight": float(step.weight),
                "max_weight": float(step.max_weight),
            },
        )

    def _write_primary_output(
        self,
        sample: dict,
        foggy: np.ndarray,
        *,
        sample_id: str,
        model_name: str,
        beta: float,
        airlight: np.ndarray,
        full_id: str | None,
        variant: OutputVariantSpec | None = None,
    ) -> Path:
        if self.output_backend.is_source_backed:
            if variant is None:
                return self.output_backend.write(sample, foggy)
            output_full_id = self._variant_full_id(
                sample,
                variant.id,
                self.output_backend,
                variant.file_id_hierarchy_name,
            )
            output_basename = (
                f"{variant.id}{self._source_extension(sample, self.output_backend)}"
            )
            attributes = self._output_variant_attributes(
                sample,
                variant,
                model_name=model_name,
                beta=beta,
                airlight=airlight,
            )
            return self.output_backend.write(
                sample,
                foggy,
                output_full_id=output_full_id,
                output_basename=output_basename,
                attributes=attributes,
            )

        output_path = self._build_output_path(
            sample_id,
            model_name,
            beta,
            airlight,
            full_id=full_id,
            variant_id=variant.id if variant else None,
        )
        return self.output_backend.write(
            sample,
            foggy,
            default_path=output_path,
        )

    def _generate_fog_cpu_progressive(self, samples: Iterable[dict]) -> list[Path]:
        try:
            total = len(samples)  # type: ignore[arg-type]
        except TypeError:
            samples = list(samples)
            total = len(samples)
        saved_paths: list[Path] = []

        with progress_bar(total, "CPU", self.logger) as bar:
            for index, sample in enumerate(samples):
                rgb = normalize_rgb(sample["rgb"])

                depth = normalize_depth(
                    sample["depth"], rgb.shape[:2], self.resize_depth_flag
                )
                depth = depth * self.depth_scale
                depth = np.maximum(depth, 0.0)

                intrinsics = extract_intrinsics(sample)
                if intrinsics is not None:
                    depth = planar_to_radial_depth(depth, intrinsics)

                sky_mask = normalize_sky_mask(sample["semantic_segmentation"])

                for step in self.progressive_scenario_steps:
                    rng = self._rng_for_progressive(
                        index,
                        step.scenario_index,
                        step.step_index,
                    )
                    plan = self._resolve_progressive_render_plan(rng, step)
                    variant = self._progressive_output_variant(step)
                    result = self.pipeline.process_np(
                        rgb=rgb,
                        depth_m=depth,
                        sky_mask=sky_mask,
                        model_name=plan.model_name,
                        model_cfg=plan.model_cfg,
                        rng=rng,
                        sample_id=sample.get("id"),
                        intrinsics=intrinsics,
                        airlight_method=plan.airlight_method,
                        capture_artifacts=plan.capture_artifacts,
                        clear_weather=plan.clear_weather,
                    )
                    saved_paths.append(
                        self._write_primary_output(
                            sample,
                            result.rgb,
                            sample_id=sample["id"],
                            model_name=plan.model_name,
                            beta=result.beta,
                            airlight=result.airlight,
                            full_id=sample.get("full_id"),
                            variant=variant,
                        )
                    )
                    if not self.output_backend.is_source_backed:
                        self._write_model_config(
                            plan.model_name,
                            plan.model_cfg,
                            saved_paths,
                        )
                    self._write_auxiliary(
                        sample,
                        k_map=result.k_map,
                        ls_map=result.ls_map,
                        sample_id=sample["id"],
                        model_name=plan.model_name,
                        full_id=sample.get("full_id"),
                        beta=result.beta,
                        airlight=result.airlight,
                        variant=variant,
                    )

                if bar is not None:
                    bar.update(1)
        self._finalize_backends()
        return saved_paths

    def _generate_fog_cpu(self, samples: Iterable[dict]) -> list[Path]:
        try:
            total = len(samples)  # type: ignore[arg-type]
        except TypeError:
            samples = list(samples)
            total = len(samples)
        saved_paths: list[Path] = []

        with progress_bar(total, "CPU", self.logger) as bar:
            for index, sample in enumerate(samples):
                rgb = normalize_rgb(sample["rgb"])

                depth = normalize_depth(
                    sample["depth"], rgb.shape[:2], self.resize_depth_flag
                )
                depth = depth * self.depth_scale
                depth = np.maximum(depth, 0.0)

                intrinsics = extract_intrinsics(sample)
                if intrinsics is not None:
                    depth = planar_to_radial_depth(depth, intrinsics)

                sky_mask = normalize_sky_mask(sample["semantic_segmentation"])

                if self.augmentation_specs:
                    for aug_index, augmentation in enumerate(self.augmentation_specs):
                        rng = self._rng_for(index, aug_index)
                        plan = self._resolve_render_plan(rng, augmentation)
                        variant = self._augmentation_output_variant(augmentation)
                        result = self.pipeline.process_np(
                            rgb=rgb,
                            depth_m=depth,
                            sky_mask=sky_mask,
                            model_name=plan.model_name,
                            model_cfg=plan.model_cfg,
                            rng=rng,
                            sample_id=sample.get("id"),
                            intrinsics=intrinsics,
                            airlight_method=plan.airlight_method,
                            capture_artifacts=plan.capture_artifacts,
                            clear_weather=plan.clear_weather,
                        )
                        saved_paths.append(
                            self._write_primary_output(
                                sample,
                                result.rgb,
                                sample_id=sample["id"],
                                model_name=plan.model_name,
                                beta=result.beta,
                                airlight=result.airlight,
                                full_id=sample.get("full_id"),
                                variant=variant,
                            )
                        )
                        if not self.output_backend.is_source_backed:
                            self._write_model_config(
                                plan.model_name,
                                plan.model_cfg,
                                saved_paths,
                            )
                        self._write_auxiliary(
                            sample,
                            k_map=result.k_map,
                            ls_map=result.ls_map,
                            sample_id=sample["id"],
                            model_name=plan.model_name,
                            full_id=sample.get("full_id"),
                            beta=result.beta,
                            airlight=result.airlight,
                            variant=variant,
                        )
                else:
                    rng = self._rng_for(index)
                    plan = self._resolve_render_plan(rng)
                    result = self.pipeline.process_np(
                        rgb=rgb,
                        depth_m=depth,
                        sky_mask=sky_mask,
                        model_name=plan.model_name,
                        model_cfg=plan.model_cfg,
                        rng=rng,
                        sample_id=sample.get("id"),
                        intrinsics=intrinsics,
                        airlight_method=plan.airlight_method,
                        capture_artifacts=plan.capture_artifacts,
                        clear_weather=plan.clear_weather,
                    )
                    saved_paths.append(
                        self._write_primary_output(
                            sample,
                            result.rgb,
                            sample_id=sample["id"],
                            model_name=plan.model_name,
                            beta=result.beta,
                            airlight=result.airlight,
                            full_id=sample.get("full_id"),
                        )
                    )
                    if not self.output_backend.is_source_backed:
                        self._write_model_config(
                            plan.model_name,
                            plan.model_cfg,
                            saved_paths,
                        )
                    self._write_auxiliary(
                        sample,
                        k_map=result.k_map,
                        ls_map=result.ls_map,
                        sample_id=sample["id"],
                        model_name=plan.model_name,
                        full_id=sample.get("full_id"),
                    )

                if bar is not None:
                    bar.update(1)
        self._finalize_backends()
        return saved_paths

    def _process_torch_pipeline(
        self,
        rgb_t: "torch.Tensor",
        depth_t: "torch.Tensor",
        sky_mask_t: "torch.Tensor",
        model_name: str,
        model_cfg: dict,
        rng: np.random.Generator,
        torch_gen: "torch.Generator",
        *,
        sample_id: str | None = None,
        intrinsics: np.ndarray | None = None,
        airlight_method: str | None = None,
        capture_artifacts: CaptureArtifactPipeline | None = None,
        clear_weather: bool = False,
    ) -> tuple["torch.Tensor", float, "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Run the complete Torch render using the CPU pipeline's semantics."""
        rgb_for_render = (
            rgb_t
            if clear_weather
            else self.pipeline.prepare_render_rgb_torch(rgb_t, model_cfg)
        )
        estimated_airlight = (
            torch.zeros(3, device=rgb_t.device, dtype=torch.float32)
            if clear_weather
            else self._estimate_airlight_torch(
                rgb_for_render,
                sky_mask_t,
                sample_id=sample_id,
                method=airlight_method,
            )
        )
        return self._apply_model_torch(
            rgb_for_render,
            depth_t,
            model_name,
            model_cfg,
            rng,
            estimated_airlight,
            torch_gen,
            sample_id=sample_id,
            intrinsics=intrinsics,
            depth_m=depth_t,
            sky_mask=sky_mask_t,
            capture_artifacts=capture_artifacts,
            clear_weather=clear_weather,
        )

    def _apply_model_torch(
        self,
        rgb_t: "torch.Tensor",
        depth_t: "torch.Tensor",
        model_name: str,
        model_cfg: dict,
        rng: np.random.Generator,
        estimated_airlight_t: "torch.Tensor",
        torch_gen: "torch.Generator",
        sample_id: str | None = None,
        intrinsics: np.ndarray | None = None,
        depth_m: Any | None = None,
        sky_mask: Any | None = None,
        capture_artifacts: CaptureArtifactPipeline | None = None,
        clear_weather: bool = False,
    ) -> tuple["torch.Tensor", float, "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        if clear_weather:
            height = int(depth_t.shape[0])
            width = int(depth_t.shape[1])
            airlight = torch.zeros(3, device=self.torch_device, dtype=torch.float32)
            k_map = torch.zeros(
                (height, width),
                device=self.torch_device,
                dtype=torch.float32,
            )
            ls_map = torch.zeros(
                (height, width, 3),
                device=self.torch_device,
                dtype=torch.float32,
            )
            return self._finalize_torch_pipeline_result(
                rgb_t,
                0.0,
                airlight,
                k_map,
                ls_map,
                rng=rng,
                sample_id=sample_id,
                model_cfg=model_cfg,
                sky_mask=sky_mask,
                intrinsics=intrinsics,
                depth_m=depth_t if depth_m is None else depth_m,
                capture_artifacts=capture_artifacts,
                clear_weather=True,
            )
        if model_name not in DEFAULT_MODEL_CONFIGS:
            raise ValueError(f"Unsupported fog model: {model_name}")
        k_mean, ls_base = self.atmospheric_light.resolve_model_torch(
            model_cfg=model_cfg,
            rng=rng,
            contrast_threshold_default=self.contrast_threshold_default,
            estimated_airlight_t=estimated_airlight_t,
            device=self.torch_device,
        )
        height = int(depth_t.shape[0])
        width = int(depth_t.shape[1])

        if model_name in ("heterogeneous_k", "heterogeneous_k_ls"):
            k_cfg = model_cfg.get("k_hetero", {})
            k_scales = resolve_scales(k_cfg, height, width, rng)
            k_noise = perlin_fbm_torch(
                height,
                width,
                k_scales,
                torch_gen,
                self.torch_device,
            )
            k_noise = prepare_noise_field_torch(k_noise, k_cfg, rng)
            min_factor = float(sample_value(k_cfg.get("min_factor", 1.0), rng))
            max_factor = float(sample_value(k_cfg.get("max_factor", 1.0), rng))
            k_field = modulate_with_noise_torch(
                torch.tensor([k_mean], device=self.torch_device, dtype=torch.float32),
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
            ls_noise = perlin_fbm_torch(
                height,
                width,
                ls_scales,
                torch_gen,
                self.torch_device,
            )
            ls_noise = prepare_noise_field_torch(ls_noise, ls_cfg, rng)
            min_factor = float(sample_value(ls_cfg.get("min_factor", 1.0), rng))
            max_factor = float(sample_value(ls_cfg.get("max_factor", 1.0), rng))
            ls_field = modulate_with_noise_torch(
                ls_base,
                ls_noise,
                min_factor,
                max_factor,
                bool(ls_cfg.get("normalize_to_mean", False)),
            )
            ls_field = apply_ls_gradient_torch(
                ls_field,
                depth_t,
                k_field,
                model_cfg,
                ls_cfg,
                rng,
            )
            ls_field = torch.clamp(ls_field, 0.0, 1.0)
        else:
            ls_field = ls_base.view(1, 1, 3)

        foggy, k_map, ls_map = render_fog_fields_torch(
            rgb_t,
            depth_t,
            k_field,
            ls_field,
            model_cfg,
            rng,
            sky_mask=sky_mask,
        )
        return self._finalize_torch_pipeline_result(
            foggy,
            k_mean,
            ls_base,
            k_map,
            ls_map,
            rng=rng,
            sample_id=sample_id,
            model_cfg=model_cfg,
            sky_mask=sky_mask,
            intrinsics=intrinsics,
            depth_m=depth_t if depth_m is None else depth_m,
            capture_artifacts=capture_artifacts,
        )

    def _finalize_torch_pipeline_result(
        self,
        foggy: "torch.Tensor",
        beta: float,
        airlight: "torch.Tensor",
        k_map: "torch.Tensor",
        ls_map: "torch.Tensor",
        *,
        rng: np.random.Generator,
        sample_id: str | None,
        model_cfg: dict,
        sky_mask: Any | None,
        intrinsics: np.ndarray | None = None,
        depth_m: Any | None = None,
        capture_artifacts: CaptureArtifactPipeline | None = None,
        clear_weather: bool = False,
    ) -> tuple["torch.Tensor", float, "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        result = FogPipelineResult(
            rgb=self.pipeline.restore_render_rgb_torch(
                foggy,
                model_cfg,
                clear_weather=clear_weather,
            ),
            beta=beta,
            airlight=airlight,
            k_map=k_map,
            ls_map=ls_map,
        )
        result = self.pipeline.apply_capture_torch(
            result,
            context=self.pipeline.capture_context(
                sample_id=sample_id,
                rng=rng,
                model_cfg=model_cfg,
                sky_mask=sky_mask,
                airlight=airlight,
                device=foggy.device,
                intrinsics=intrinsics,
                depth_m=depth_m,
                k_map=k_map,
            ),
            capture_artifacts=capture_artifacts,
        )
        return result.rgb, result.beta, result.airlight, result.k_map, result.ls_map

    def _generate_fog_gpu(self, samples: Iterable[dict]) -> list[Path]:
        if torch is None or self.torch_device is None:
            raise RuntimeError("Torch device not configured for GPU execution.")
        if self.mode == "progressive":
            return self._generate_fog_gpu_progressive(samples)
        if self.augmentation_specs:
            return self._generate_fog_gpu_augmented(samples)
        device = self.torch_device
        try:
            total = len(samples)  # type: ignore[arg-type]
        except TypeError:
            samples = list(samples)
            total = len(samples)
        saved_paths: list[Path] = []

        with progress_bar(total, "GPU", self.logger) as bar:
            for batch_index, batch in enumerate(
                iter_batches(enumerate(samples), self.gpu_batch_size)
            ):
                batch_plan = self._resolve_gpu_batch_render_plan(batch_index)
                items: list[dict] = []
                for global_index, sample in batch:
                    rgb = _to_numpy(sample["rgb"])
                    if _is_chw(rgb):
                        rgb = np.transpose(rgb, (1, 2, 0))
                    depth = normalize_depth(
                        sample["depth"], rgb.shape[:2], self.resize_depth_flag
                    )
                    intrinsics = extract_intrinsics(sample)
                    if self.seed is not None:
                        rng = np.random.default_rng(
                            np.random.SeedSequence([self.seed, global_index])
                        )
                    else:
                        rng = self.base_rng
                    plan = batch_plan or self._resolve_render_plan(rng)
                    items.append(
                        {
                            "sample_id": sample["id"],
                            "full_id": sample.get("full_id"),
                            "meta": sample.get("meta"),
                            "rgb": rgb,
                            "depth": depth,
                            "intrinsics": intrinsics,
                            "sky_mask": normalize_sky_mask(sample["semantic_segmentation"]),
                            "rng": rng,
                            "model_name": plan.model_name,
                            "model_cfg": plan.model_cfg,
                            "airlight_method": plan.airlight_method,
                            "capture_artifacts": plan.capture_artifacts,
                            "scenario_name": plan.scenario_name,
                            "clear_weather": plan.clear_weather,
                            "index": global_index,
                        }
                    )

                if not items:
                    continue

                grouped: dict[tuple[int, int], list[dict]] = {}
                for item in items:
                    shape = (item["rgb"].shape[0], item["rgb"].shape[1])
                    grouped.setdefault(shape, []).append(item)

                for group_items in grouped.values():
                    uniform_groups: dict[tuple[int, int, str | None], list[dict]] = {}
                    other_items = []
                    for item in group_items:
                        if (
                            item["clear_weather"]
                            or item["model_name"] != "uniform"
                            or self.seed is None
                        ):
                            other_items.append(item)
                            continue
                        capture_artifacts = item.get("capture_artifacts")
                        capture_key = (
                            0 if capture_artifacts is None else id(capture_artifacts)
                        )
                        group_key = (
                            capture_key,
                            id(item["model_cfg"]),
                            item.get("airlight_method"),
                        )
                        uniform_groups.setdefault(group_key, []).append(item)

                    for uniform_items in uniform_groups.values():
                        rgb_batch = torch.stack(
                            [
                                normalize_rgb_torch(item["rgb"], device)
                                for item in uniform_items
                            ],
                            dim=0,
                        )
                        rgb_batch = self.pipeline.prepare_render_rgb_torch(
                            rgb_batch,
                            uniform_items[0]["model_cfg"],
                        )
                        depth_tensors = []
                        for item in uniform_items:
                            depth_t = torch.from_numpy(item["depth"]).to(
                                device=device,
                                dtype=torch.float32,
                            )
                            depth_t = torch.clamp(
                                depth_t * self.depth_scale,
                                min=0.0,
                            )
                            K_np = item.get("intrinsics")
                            if K_np is not None:
                                K_t = torch.from_numpy(K_np).to(
                                    device=device,
                                    dtype=torch.float32,
                                )
                                depth_t = planar_to_radial_depth_torch(depth_t, K_t)
                            depth_tensors.append(depth_t)
                        k_means, ls_base = (
                            self.atmospheric_light.resolve_uniform_batch_torch(
                                rgb_batch=rgb_batch,
                                items=uniform_items,
                                device=device,
                                contrast_threshold_default=(
                                    self.contrast_threshold_default
                                ),
                                method=uniform_items[0].get("airlight_method"),
                            )
                        )
                        rendered = [
                            render_fog_fields_torch(
                                rgb_batch[idx],
                                depth_tensors[idx],
                                k_means[idx],
                                ls_base[idx].view(1, 1, 3),
                                item["model_cfg"],
                                item["rng"],
                                sky_mask=torch.from_numpy(item["sky_mask"]).to(
                                    device=device,
                                    dtype=torch.bool,
                                ),
                            )
                            for idx, item in enumerate(uniform_items)
                        ]
                        foggy = torch.stack([result[0] for result in rendered], dim=0)
                        k_maps = [result[1] for result in rendered]
                        ls_maps = [result[2] for result in rendered]
                        foggy = self.pipeline.restore_render_rgb_torch(
                            foggy,
                            uniform_items[0]["model_cfg"],
                        )
                        capture_artifacts = uniform_items[0].get("capture_artifacts")
                        foggy = self.pipeline.apply_capture_torch_batch(
                            foggy,
                            contexts=tuple(
                                self.pipeline.capture_context(
                                    sample_id=item["sample_id"],
                                    rng=item["rng"],
                                    model_cfg=item["model_cfg"],
                                    sky_mask=item["sky_mask"],
                                    airlight=ls_base[idx],
                                    device=device,
                                    intrinsics=item.get("intrinsics"),
                                    depth_m=depth_tensors[idx],
                                    k_map=k_maps[idx],
                                )
                                for idx, item in enumerate(uniform_items)
                            ),
                            capture_artifacts=capture_artifacts,
                        )

                        for idx, item in enumerate(uniform_items):
                            foggy_img = (
                                torch.clamp(foggy[idx], 0.0, 1.0).cpu().numpy()
                            )
                            airlight_np = ls_base[idx].detach().cpu().numpy()
                            sample_ref = {
                                "id": item["sample_id"],
                                "full_id": item.get("full_id"),
                                "meta": item.get("meta"),
                            }
                            if self.output_backend.is_source_backed:
                                saved_paths.append(
                                    self.output_backend.write(sample_ref, foggy_img)
                                )
                            else:
                                output_path = self._build_output_path(
                                    item["sample_id"],
                                    item["model_name"],
                                    k_means[idx],
                                    airlight_np,
                                    full_id=item.get("full_id"),
                                )
                                saved_paths.append(
                                    self.output_backend.write(
                                        sample_ref,
                                        foggy_img,
                                        default_path=output_path,
                                    )
                                )
                                self._write_model_config(
                                    item["model_name"], item["model_cfg"], saved_paths
                                )

                            if (
                                SCATTERING_COEFFICIENT_SLOT in self.output_backends
                                or ATMOSPHERIC_LIGHT_SLOT in self.output_backends
                            ):
                                self._write_auxiliary(
                                    sample_ref,
                                    k_map=k_maps[idx].detach().cpu().numpy(),
                                    ls_map=ls_maps[idx].detach().cpu().numpy(),
                                    sample_id=item["sample_id"],
                                    model_name=item["model_name"],
                                    full_id=item.get("full_id"),
                                )

                    for item in other_items:
                        rgb_t = normalize_rgb_torch(item["rgb"], device)
                        depth_t = torch.from_numpy(item["depth"]).to(
                            device=device, dtype=torch.float32
                        )
                        depth_t = torch.clamp(depth_t * self.depth_scale, min=0.0)
                        K_np = item.get("intrinsics")
                        if K_np is not None:
                            K_t = torch.from_numpy(K_np).to(
                                device=device, dtype=torch.float32,
                            )
                            depth_t = planar_to_radial_depth_torch(depth_t, K_t)
                        sky_mask_t = (
                            torch.from_numpy(item["sky_mask"]).to(device).bool()
                        )
                        torch_gen = torch_generator_for_index(
                            self.torch_device,
                            self.seed,
                            self.base_rng,
                            item["index"],
                        )
                        foggy_t, beta, airlight_t, k_map_t, ls_map_t = (
                            self._process_torch_pipeline(
                                rgb_t,
                                depth_t,
                                sky_mask_t,
                                item["model_name"],
                                item["model_cfg"],
                                item["rng"],
                                torch_gen,
                                sample_id=item["sample_id"],
                                intrinsics=item.get("intrinsics"),
                                airlight_method=item.get("airlight_method"),
                                capture_artifacts=item.get("capture_artifacts"),
                                clear_weather=item["clear_weather"],
                            )
                        )
                        foggy_img = torch.clamp(foggy_t, 0.0, 1.0).cpu().numpy()
                        airlight_np = airlight_t.detach().cpu().numpy()
                        sample_ref = {
                            "id": item["sample_id"],
                            "full_id": item.get("full_id"),
                            "meta": item.get("meta"),
                        }
                        if self.output_backend.is_source_backed:
                            saved_paths.append(
                                self.output_backend.write(sample_ref, foggy_img)
                            )
                        else:
                            output_path = self._build_output_path(
                                item["sample_id"],
                                item["model_name"],
                                beta,
                                airlight_np,
                                full_id=item.get("full_id"),
                            )
                            saved_paths.append(
                                self.output_backend.write(
                                    sample_ref,
                                    foggy_img,
                                    default_path=output_path,
                                )
                            )
                            self._write_model_config(
                                item["model_name"], item["model_cfg"], saved_paths
                            )

                        if (
                            SCATTERING_COEFFICIENT_SLOT in self.output_backends
                            or ATMOSPHERIC_LIGHT_SLOT in self.output_backends
                        ):
                            k_map_np = k_map_t.detach().cpu().numpy()
                            ls_map_np = ls_map_t.detach().cpu().numpy()
                            self._write_auxiliary(
                                sample_ref,
                                k_map=k_map_np,
                                ls_map=ls_map_np,
                                sample_id=item["sample_id"],
                                model_name=item["model_name"],
                                full_id=item.get("full_id"),
                            )

                if bar is not None:
                    bar.update(len(batch))

        self._finalize_backends()
        return saved_paths

    def _generate_fog_gpu_progressive(self, samples: Iterable[dict]) -> list[Path]:
        if torch is None or self.torch_device is None:
            raise RuntimeError("Torch device not configured for GPU execution.")
        device = self.torch_device
        try:
            total = len(samples)  # type: ignore[arg-type]
        except TypeError:
            samples = list(samples)
            total = len(samples)
        saved_paths: list[Path] = []

        with progress_bar(total, "GPU", self.logger) as bar:
            for index, sample in enumerate(samples):
                rgb_np = _to_numpy(sample["rgb"])
                if _is_chw(rgb_np):
                    rgb_np = np.transpose(rgb_np, (1, 2, 0))
                depth_np = normalize_depth(
                    sample["depth"], rgb_np.shape[:2], self.resize_depth_flag
                )
                rgb_t = normalize_rgb_torch(rgb_np, device)
                depth_t = torch.from_numpy(depth_np).to(
                    device=device,
                    dtype=torch.float32,
                )
                depth_t = torch.clamp(depth_t * self.depth_scale, min=0.0)
                intrinsics = extract_intrinsics(sample)
                if intrinsics is not None:
                    K_t = torch.from_numpy(intrinsics).to(
                        device=device,
                        dtype=torch.float32,
                    )
                    depth_t = planar_to_radial_depth_torch(depth_t, K_t)

                sky_mask_np = normalize_sky_mask(sample["semantic_segmentation"])
                sky_mask_t = torch.from_numpy(sky_mask_np).to(
                    device=device,
                    dtype=torch.bool,
                )

                for variant_index, step in enumerate(self.progressive_scenario_steps):
                    rng = self._rng_for_progressive(
                        index,
                        step.scenario_index,
                        step.step_index,
                    )
                    plan = self._resolve_progressive_render_plan(rng, step)
                    variant = self._progressive_output_variant(step)
                    torch_gen = torch_generator_for_index(
                        self.torch_device,
                        self.seed,
                        self.base_rng,
                        index * 1_000_000 + variant_index,
                    )
                    foggy_t, beta, airlight_t, k_map_t, ls_map_t = (
                        self._process_torch_pipeline(
                            rgb_t,
                            depth_t,
                            sky_mask_t,
                            plan.model_name,
                            plan.model_cfg,
                            rng,
                            torch_gen,
                            sample_id=sample.get("id"),
                            intrinsics=intrinsics,
                            airlight_method=plan.airlight_method,
                            capture_artifacts=plan.capture_artifacts,
                            clear_weather=plan.clear_weather,
                        )
                    )
                    foggy_img = torch.clamp(foggy_t, 0.0, 1.0).cpu().numpy()
                    airlight_np = airlight_t.detach().cpu().numpy()
                    saved_paths.append(
                        self._write_primary_output(
                            sample,
                            foggy_img,
                            sample_id=sample["id"],
                            model_name=plan.model_name,
                            beta=beta,
                            airlight=airlight_np,
                            full_id=sample.get("full_id"),
                            variant=variant,
                        )
                    )
                    if not self.output_backend.is_source_backed:
                        self._write_model_config(
                            plan.model_name,
                            plan.model_cfg,
                            saved_paths,
                        )

                    if (
                        SCATTERING_COEFFICIENT_SLOT in self.output_backends
                        or ATMOSPHERIC_LIGHT_SLOT in self.output_backends
                    ):
                        self._write_auxiliary(
                            sample,
                            k_map=k_map_t.detach().cpu().numpy(),
                            ls_map=ls_map_t.detach().cpu().numpy(),
                            sample_id=sample["id"],
                            model_name=plan.model_name,
                            full_id=sample.get("full_id"),
                            beta=beta,
                            airlight=airlight_np,
                            variant=variant,
                        )

                if bar is not None:
                    bar.update(1)

        self._finalize_backends()
        return saved_paths

    def _generate_fog_gpu_augmented(self, samples: Iterable[dict]) -> list[Path]:
        if torch is None or self.torch_device is None:
            raise RuntimeError("Torch device not configured for GPU execution.")
        device = self.torch_device
        try:
            total = len(samples)  # type: ignore[arg-type]
        except TypeError:
            samples = list(samples)
            total = len(samples)
        saved_paths: list[Path] = []

        with progress_bar(total, "GPU", self.logger) as bar:
            for index, sample in enumerate(samples):
                rgb_np = _to_numpy(sample["rgb"])
                if _is_chw(rgb_np):
                    rgb_np = np.transpose(rgb_np, (1, 2, 0))
                depth_np = normalize_depth(
                    sample["depth"], rgb_np.shape[:2], self.resize_depth_flag
                )
                rgb_t = normalize_rgb_torch(rgb_np, device)
                depth_t = torch.from_numpy(depth_np).to(
                    device=device,
                    dtype=torch.float32,
                )
                depth_t = torch.clamp(depth_t * self.depth_scale, min=0.0)
                intrinsics = extract_intrinsics(sample)
                if intrinsics is not None:
                    K_t = torch.from_numpy(intrinsics).to(
                        device=device,
                        dtype=torch.float32,
                    )
                    depth_t = planar_to_radial_depth_torch(depth_t, K_t)

                sky_mask_np = normalize_sky_mask(sample["semantic_segmentation"])
                sky_mask_t = torch.from_numpy(sky_mask_np).to(
                    device=device,
                    dtype=torch.bool,
                )

                for aug_index, augmentation in enumerate(self.augmentation_specs):
                    rng = self._rng_for(index, aug_index)
                    plan = self._resolve_render_plan(rng, augmentation)
                    variant = self._augmentation_output_variant(augmentation)
                    torch_gen = torch_generator_for_index(
                        self.torch_device,
                        self.seed,
                        self.base_rng,
                        index * 100_000 + aug_index,
                    )
                    foggy_t, beta, airlight_t, k_map_t, ls_map_t = (
                        self._process_torch_pipeline(
                            rgb_t,
                            depth_t,
                            sky_mask_t,
                            plan.model_name,
                            plan.model_cfg,
                            rng,
                            torch_gen,
                            sample_id=sample.get("id"),
                            intrinsics=intrinsics,
                            airlight_method=plan.airlight_method,
                            capture_artifacts=plan.capture_artifacts,
                            clear_weather=plan.clear_weather,
                        )
                    )
                    foggy_img = torch.clamp(foggy_t, 0.0, 1.0).cpu().numpy()
                    airlight_np = airlight_t.detach().cpu().numpy()
                    saved_paths.append(
                        self._write_primary_output(
                            sample,
                            foggy_img,
                            sample_id=sample["id"],
                            model_name=plan.model_name,
                            beta=beta,
                            airlight=airlight_np,
                            full_id=sample.get("full_id"),
                            variant=variant,
                        )
                    )
                    if not self.output_backend.is_source_backed:
                        self._write_model_config(
                            plan.model_name,
                            plan.model_cfg,
                            saved_paths,
                        )

                    if (
                        SCATTERING_COEFFICIENT_SLOT in self.output_backends
                        or ATMOSPHERIC_LIGHT_SLOT in self.output_backends
                    ):
                        self._write_auxiliary(
                            sample,
                            k_map=k_map_t.detach().cpu().numpy(),
                            ls_map=ls_map_t.detach().cpu().numpy(),
                            sample_id=sample["id"],
                            model_name=plan.model_name,
                            full_id=sample.get("full_id"),
                            beta=beta,
                            airlight=airlight_np,
                            variant=variant,
                        )

                if bar is not None:
                    bar.update(1)

        self._finalize_backends()
        return saved_paths

    def _build_output_path(
        self,
        sample_id: str,
        model_name: str,
        beta: float,
        airlight: np.ndarray,
        full_id: str | None = None,
        variant_id: str | None = None,
    ) -> Path:
        if variant_id is not None:
            filename = f"{variant_id}.png"
        elif self.suffix:
            filename = f"{sample_id}_{self.suffix}.png"
        else:
            beta_str = format_value(beta)
            r_str, g_str, b_str = (format_value(v) for v in airlight)
            filename = (
                f"beta_{beta_str}_airlight_{r_str}_{g_str}_{b_str}_rgb_{sample_id}.png"
            )
        base = self.out_path / model_name
        if full_id:
            # full_id is e.g. "/Scene02/30-deg-right/Camera_0/00000"
            # Use all segments except the last (the frame id) as subdirs.
            parts = [p for p in full_id.split("/") if p]
            if len(parts) > 1:
                base = base.joinpath(*parts[:-1])
        if variant_id is not None:
            base = base / sample_id
        return base / filename

    def _write_model_config(
        self, model_name: str, model_cfg: dict, saved_paths: list
    ) -> None:
        if self.output_backend.is_source_backed:
            return
        if model_name in self._written_configs:
            return
        target_dir = self.out_path / model_name
        config_path = target_dir / "config.json"

        enriched_config = {**model_cfg, "size": len(saved_paths)}
        self.output_backend.write_json(config_path, enriched_config)
        self._written_configs.add(model_name)

    def _write_auxiliary(
        self,
        sample: dict,
        *,
        k_map: np.ndarray,
        ls_map: np.ndarray,
        sample_id: str,
        model_name: str,
        full_id: str | None,
        beta: float | None = None,
        airlight: np.ndarray | None = None,
        variant: OutputVariantSpec | None = None,
    ) -> None:
        """Write the per-pixel β / L_s maps to their slots, if active."""
        scattering_backend = self.output_backends.get(SCATTERING_COEFFICIENT_SLOT)
        if scattering_backend is not None:
            self._write_aux_to_backend(
                scattering_backend,
                sample,
                k_map,
                sample_id=sample_id,
                model_name=model_name,
                full_id=full_id,
                beta=beta,
                airlight=airlight,
                variant=variant,
            )
        airlight_backend = self.output_backends.get(ATMOSPHERIC_LIGHT_SLOT)
        if airlight_backend is not None:
            self._write_aux_to_backend(
                airlight_backend,
                sample,
                ls_map,
                sample_id=sample_id,
                model_name=model_name,
                full_id=full_id,
                beta=beta,
                airlight=airlight,
                variant=variant,
            )

    def _write_aux_to_backend(
        self,
        backend: Any,
        sample: dict,
        value: np.ndarray,
        *,
        sample_id: str,
        model_name: str,
        full_id: str | None,
        beta: float | None = None,
        airlight: np.ndarray | None = None,
        variant: OutputVariantSpec | None = None,
    ) -> None:
        if backend.is_source_backed:
            if variant is None:
                backend.write(sample, value)
                return
            output_full_id = self._variant_full_id(
                sample,
                variant.id,
                backend,
                variant.file_id_hierarchy_name,
            )
            output_basename = f"{variant.id}{backend.output_extension or '.npy'}"
            attributes = (
                self._output_variant_attributes(
                    sample,
                    variant,
                    model_name=model_name,
                    beta=float(beta) if beta is not None else float(np.mean(value)),
                    airlight=(
                        airlight
                        if airlight is not None
                        else np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
                    ),
                )
            )
            backend.write(
                sample,
                value,
                output_full_id=output_full_id,
                output_basename=output_basename,
                attributes=attributes,
            )
            return
        # Legacy disk fallback: mirror the RGB output's hierarchy but emit .npy.
        base = backend.root / model_name
        if full_id:
            parts = [p for p in full_id.split("/") if p]
            if len(parts) > 1:
                base = base.joinpath(*parts[:-1])
        if variant is not None:
            base = base / sample_id
            target = base / f"{variant.id}.npy"
            backend.write(sample, value, default_path=target)
            return
        target = base / f"{sample_id}.npy"
        backend.write(sample, value, default_path=target)

    def _finalize_backends(self) -> None:
        for backend in self.output_backends.values():
            backend.finalize()


# Backward compatibility alias
Foggify = FogTransform
