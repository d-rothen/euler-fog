from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class CaptureContext:
    """Metadata passed to camera/capture artifact stages."""

    sample_id: str | None = None
    rng: Any | None = None
    device: Any | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)


class CaptureArtifactStage:
    """Base class for post-render camera artifact stages.

    Future stages such as exposure, vignetting, and sensor noise should operate
    on the already rendered scene image and leave physical auxiliary maps
    untouched.
    """

    name = "capture_artifact"

    def apply_np(self, image, context: CaptureContext):
        return image

    def apply_torch(self, image, context: CaptureContext):
        return image

    def apply_torch_batch(self, images, contexts: tuple[CaptureContext, ...]):
        if not contexts:
            return images
        processed = [
            self.apply_torch(images[index], context)
            for index, context in enumerate(contexts)
        ]
        return _stack_like(images, processed)


class CaptureArtifactPipeline:
    """Ordered post-fog camera/capture artifact pipeline.

    With no configured stages this is intentionally zero-copy: images are
    returned unchanged. The config surface is present so later exposure/noise
    stages can be added without changing the fog transform's control flow.
    """

    def __init__(self, stages: tuple[CaptureArtifactStage, ...] = ()) -> None:
        self.stages = stages

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "CaptureArtifactPipeline":
        raw = config.get("capture_artifacts", config.get("capture"))
        if raw is None or raw is False:
            return cls()
        if raw is True:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("capture must be a boolean or object")
        if raw.get("enabled", True) is False:
            return cls()

        stages = raw.get("stages", ())
        if stages is None:
            stages = ()
        if not isinstance(stages, (list, tuple)):
            raise ValueError("capture.stages must be a list")
        if stages:
            names = [
                str(stage.get("type", stage.get("name", stage)))
                if isinstance(stage, dict)
                else str(stage)
                for stage in stages
            ]
            raise NotImplementedError(
                "Capture artifact stages are not implemented yet: "
                + ", ".join(names)
            )
        return cls()

    def apply_np(self, image, context: CaptureContext):
        for stage in self.stages:
            image = stage.apply_np(image, context)
        return image

    def apply_torch(self, image, context: CaptureContext):
        for stage in self.stages:
            image = stage.apply_torch(image, context)
        return image

    def apply_torch_batch(self, images, contexts: tuple[CaptureContext, ...]):
        if not self.stages:
            return images
        for stage in self.stages:
            images = stage.apply_torch_batch(images, contexts)
        return images


def _stack_like(reference, images: list[Any]):
    if not images:
        return reference
    if hasattr(reference, "new_empty") and hasattr(reference, "dim"):
        import torch

        return torch.stack(images, dim=0)
    raise TypeError("Unsupported batch image type for capture pipeline")
