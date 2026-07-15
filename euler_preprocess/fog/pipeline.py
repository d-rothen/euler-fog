from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from euler_preprocess.common.color import (
    linear_to_srgb,
    linear_to_srgb_torch,
    srgb_to_linear,
    srgb_to_linear_torch,
)
from euler_preprocess.fog.atmospheric_light import AtmosphericLightResolver
from euler_preprocess.fog.capture import CaptureArtifactPipeline, CaptureContext
from euler_preprocess.fog.models import apply_model


@dataclass(frozen=True)
class FogPipelineResult:
    """Fog pipeline output plus the physical maps used to render it."""

    rgb: Any
    beta: float
    airlight: Any
    k_map: Any
    ls_map: Any


class FogProcessingPipeline:
    """Run ideal scene effects first, then capture-specific artifacts.

    The current ideal scene effect is physics-based fog rendering. Capture
    artifacts are an explicit second stage so exposure, vignetting, and sensor
    noise can be added later without changing the rendering path or auxiliary
    outputs.
    """

    def __init__(
        self,
        *,
        atmospheric_light: AtmosphericLightResolver,
        contrast_threshold_default: float,
        capture_artifacts: CaptureArtifactPipeline | None = None,
        render_input_space: str = "linear",
    ) -> None:
        self.atmospheric_light = atmospheric_light
        self.contrast_threshold_default = float(contrast_threshold_default)
        self.capture_artifacts = capture_artifacts or CaptureArtifactPipeline()
        self.render_input_space = _normalize_render_input_space(render_input_space)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        atmospheric_light: AtmosphericLightResolver,
        contrast_threshold_default: float,
    ) -> "FogProcessingPipeline":
        return cls(
            atmospheric_light=atmospheric_light,
            contrast_threshold_default=contrast_threshold_default,
            capture_artifacts=CaptureArtifactPipeline.from_config(config),
            render_input_space=str(config.get("render_input_space", "linear")),
        )

    def render_scene_np(
        self,
        *,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        sky_mask: np.ndarray,
        model_name: str,
        model_cfg: dict,
        rng: np.random.Generator,
        sample_id: str | None,
        airlight_method: str | None = None,
    ) -> FogPipelineResult:
        render_input_space = self.render_input_space_for(model_cfg)
        rgb_for_render = srgb_to_linear(rgb) if render_input_space == "srgb" else rgb
        estimated_airlight = self.atmospheric_light.estimate_np(
            rgb_for_render,
            sky_mask,
            sample_id=sample_id,
            method=airlight_method,
        )
        foggy, beta, airlight, k_map, ls_map = apply_model(
            rgb_for_render,
            depth_m,
            model_name,
            model_cfg,
            rng,
            self.contrast_threshold_default,
            estimated_airlight,
            sky_mask=sky_mask,
        )
        if render_input_space == "srgb":
            foggy = linear_to_srgb(foggy)
        return FogPipelineResult(
            rgb=foggy,
            beta=beta,
            airlight=airlight,
            k_map=k_map,
            ls_map=ls_map,
        )

    def render_input_space_for(self, model_cfg: dict) -> str:
        """Resolve the input encoding used by both scene-rendering backends."""
        return _normalize_render_input_space(
            model_cfg.get("render_input_space", self.render_input_space)
        )

    def prepare_render_rgb_torch(self, rgb, model_cfg: dict):
        """Convert display RGB to the scene-linear domain when configured."""
        if self.render_input_space_for(model_cfg) == "srgb":
            return srgb_to_linear_torch(rgb)
        return rgb

    def restore_render_rgb_torch(
        self,
        rgb,
        model_cfg: dict,
        *,
        clear_weather: bool = False,
    ):
        """Return a rendered Torch image to the CPU pathway's output domain."""
        if not clear_weather and self.render_input_space_for(model_cfg) == "srgb":
            return linear_to_srgb_torch(rgb)
        return rgb

    def capture_context(
        self,
        *,
        sample_id: str | None,
        rng: np.random.Generator,
        model_cfg: dict,
        sky_mask,
        airlight,
        intrinsics=None,
        depth_m=None,
        k_map=None,
        device=None,
    ) -> CaptureContext:
        """Build the complete capture context shared by CPU and Torch routes."""
        return CaptureContext(
            sample_id=sample_id,
            rng=rng,
            device=device,
            intrinsics=intrinsics,
            depth_m=depth_m,
            k_map=k_map,
            attributes={
                "sky_mask": sky_mask,
                "airlight": airlight,
                "render_input_space": model_cfg.get(
                    "render_input_space",
                    self.render_input_space,
                ),
            },
        )

    def process_np(
        self,
        *,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        sky_mask: np.ndarray,
        model_name: str,
        model_cfg: dict,
        rng: np.random.Generator,
        sample_id: str | None,
        intrinsics: np.ndarray | None = None,
        airlight_method: str | None = None,
        capture_artifacts: CaptureArtifactPipeline | None = None,
        clear_weather: bool = False,
    ) -> FogPipelineResult:
        if clear_weather:
            height, width = depth_m.shape
            result = FogPipelineResult(
                rgb=rgb,
                beta=0.0,
                airlight=np.zeros(3, dtype=np.float32),
                k_map=np.zeros((height, width), dtype=np.float32),
                ls_map=np.zeros((height, width, 3), dtype=np.float32),
            )
        else:
            result = self.render_scene_np(
                rgb=rgb,
                depth_m=depth_m,
                sky_mask=sky_mask,
                model_name=model_name,
                model_cfg=model_cfg,
                rng=rng,
                sample_id=sample_id,
                airlight_method=airlight_method,
            )
        return self.apply_capture_np(
            result,
            context=self.capture_context(
                sample_id=sample_id,
                rng=rng,
                model_cfg=model_cfg,
                sky_mask=sky_mask,
                airlight=result.airlight,
                intrinsics=intrinsics,
                depth_m=depth_m,
                k_map=result.k_map,
            ),
            capture_artifacts=capture_artifacts,
        )

    def apply_capture_np(
        self,
        result: FogPipelineResult,
        *,
        context: CaptureContext,
        capture_artifacts: CaptureArtifactPipeline | None = None,
    ) -> FogPipelineResult:
        artifacts = capture_artifacts or self.capture_artifacts
        rgb = artifacts.apply_np(result.rgb, context)
        return replace(result, rgb=rgb)

    def apply_capture_torch(
        self,
        result: FogPipelineResult,
        *,
        context: CaptureContext,
        capture_artifacts: CaptureArtifactPipeline | None = None,
    ) -> FogPipelineResult:
        artifacts = capture_artifacts or self.capture_artifacts
        rgb = artifacts.apply_torch(result.rgb, context)
        return replace(result, rgb=rgb)

    def apply_capture_torch_batch(
        self,
        rgb_batch,
        *,
        contexts: tuple[CaptureContext, ...],
        capture_artifacts: CaptureArtifactPipeline | None = None,
    ):
        artifacts = capture_artifacts or self.capture_artifacts
        return artifacts.apply_torch_batch(rgb_batch, contexts)


def _normalize_render_input_space(value: str) -> str:
    space = str(value).strip().lower()
    if space in {"srgb", "s_rgb", "display", "gamma"}:
        return "srgb"
    if space in {"linear", "scene_linear", "scene-linear"}:
        return "linear"
    raise ValueError("render_input_space must be 'linear' or 'srgb'")
