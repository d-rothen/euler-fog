from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

from euler_preprocess.common.intrinsics import extract_intrinsics, planar_to_radial_depth
from euler_preprocess.common.io import load_json
from euler_preprocess.common.logging import get_logger, progress_bar
from euler_preprocess.common.normalize import normalize_depth
from euler_preprocess.common.output import LegacyOutputBackend
from euler_preprocess.common.transform import Transform


class RadialTransform(Transform):
    """Convert planar (z-buffer) depth to radial (Euclidean) depth.

    Requires camera intrinsics (hierarchical modality ``intrinsics``) to
    compute the per-pixel ray direction.

    Config JSON keys:
        ``device`` (str): ``"cpu"`` (default). GPU support reserved for future use.

    Each output sample is saved as a ``.npy`` float32 depth map.
    """

    REQUIRED_MODALITIES: ClassVar[set[str]] = {"depth"}
    REQUIRED_HIERARCHICAL_MODALITIES: ClassVar[set[str]] = {"intrinsics"}
    SOURCE_MODALITY: ClassVar[str] = "depth"
    OUTPUT_SLOT: ClassVar[str] = "depth"
    OUTPUT_INDEX_META_OVERRIDES: ClassVar[dict[str, object]] = {"radial_depth": True}

    def __init__(
        self,
        config_path: str,
        out_path: str,
        output_backend: Any | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.output_backend = output_backend or LegacyOutputBackend(out_path)
        self.out_path = self.output_backend.root

        self.config = load_json(self.config_path)
        self.logger = get_logger()
        self.logger.info(
            "RadialTransform: output=%s", self.out_path,
        )

    def run(self, samples: Iterable[dict]) -> list[Path]:
        try:
            total = len(samples)  # type: ignore[arg-type]
        except TypeError:
            samples = list(samples)
            total = len(samples)
        saved_paths: list[Path] = []

        with progress_bar(total, "radial", self.logger) as bar:
            for sample in samples:
                depth = normalize_depth(sample["depth"])
                K = extract_intrinsics(sample)
                if K is None:
                    raise ValueError(
                        f"Sample '{sample.get('id', '?')}' has no intrinsics. "
                        "RadialTransform requires the 'intrinsics' hierarchical modality."
                    )

                radial = planar_to_radial_depth(depth, K)

                output_path = self._build_output_path(
                    sample["id"], full_id=sample.get("full_id"),
                )
                saved_paths.append(
                    self.output_backend.write(
                        sample,
                        radial,
                        default_path=output_path,
                    )
                )

                if bar is not None:
                    bar.update(1)

        self.output_backend.finalize()
        return saved_paths

    def _build_output_path(
        self, sample_id: str, full_id: str | None = None,
    ) -> Path:
        base = self.out_path
        if full_id:
            parts = [p for p in full_id.split("/") if p]
            if len(parts) > 1:
                base = base.joinpath(*parts[:-1])
        return base / f"{sample_id}.npy"
