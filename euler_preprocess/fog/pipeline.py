from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

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
    ) -> None:
        self.atmospheric_light = atmospheric_light
        self.contrast_threshold_default = float(contrast_threshold_default)
        self.capture_artifacts = capture_artifacts or CaptureArtifactPipeline()

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
        estimated_airlight = self.atmospheric_light.estimate_np(
            rgb,
            sky_mask,
            sample_id=sample_id,
            method=airlight_method,
        )
        foggy, beta, airlight, k_map, ls_map = apply_model(
            rgb,
            depth_m,
            model_name,
            model_cfg,
            rng,
            self.contrast_threshold_default,
            estimated_airlight,
        )
        return FogPipelineResult(
            rgb=foggy,
            beta=beta,
            airlight=airlight,
            k_map=k_map,
            ls_map=ls_map,
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
    ) -> FogPipelineResult:
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
            context=CaptureContext(
                sample_id=sample_id,
                rng=rng,
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
