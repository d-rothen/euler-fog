from __future__ import annotations

import json
import tempfile
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ds_crawler import DatasetWriter, ZipDatasetWriter
from ds_crawler.zip_utils import (
    METADATA_DIR,
    OUTPUT_FILENAME as DS_CRAWLER_INDEX_FILENAME,
    get_metadata_entry_name,
)
from euler_loading import MultiModalDataset
from euler_loading.dataset import create_dataset_writer_from_index
from euler_loading.loaders._writer_utils import supports_stream_target

from euler_preprocess.common.io import OutputWriter


PIPELINE_OUTPUTS_MANIFEST_VERSION = 1
_PIPELINE_OUTPUT_STORAGE_KINDS = {"directory", "zip", "file"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class OutputSlotSpec:
    """Auxiliary-slot spec for transforms producing more than one modality.

    Auxiliary slots reuse the source modality's hierarchy/indexing so the
    written files line up with the input dataset, but supply their own writer
    and ds-crawler metadata so the resulting on-disk dataset advertises the
    correct modality type and loader.

    Attributes:
        source_modality: Name of the input modality whose hierarchy and
            per-sample basenames are mirrored when writing auxiliary outputs.
        writer: Writer callable invoked as ``writer(target, value, meta)``.
            ``target`` is either a filesystem path (``str``/``PathLike``) or a
            binary stream (when the writer is marked stream-supported and the
            output is a zip).
        index_overlay: Mapping merged on top of the source modality's
            ``index_output`` to produce the ds-crawler head metadata for this
            slot.  Use this to override ``name``/``type``/``euler_train``/
            ``euler_loading``/``meta`` while inheriting indexing/hierarchy.
        output_extension: When set (e.g. ``".npy"``), source basenames are
            rewritten with this extension before writing.
        meta: Optional ``modality_meta`` passed to the writer.  Defaults to
            the ``meta`` field from the merged ``index_overlay`` when set there.
    """

    source_modality: str
    writer: Callable[..., None]
    index_overlay: dict[str, Any]
    output_extension: str | None = None
    meta: dict[str, Any] | None = None


@dataclass(frozen=True)
class PipelineOutputTargetConfig:
    """Runtime-resolved pipeline output target for a single transform output."""

    slot: str
    model_modality_id: int | None
    dataset_type: str
    relative_path: str
    path: str
    storage: str

    def __post_init__(self) -> None:
        if not self.slot:
            raise ValueError("pipeline.output_targets[].slot is required")
        if self.model_modality_id is not None and self.model_modality_id <= 0:
            raise ValueError(
                "pipeline.output_targets[].modelModalityId must be positive"
            )
        if not self.dataset_type:
            raise ValueError("pipeline.output_targets[].datasetType is required")
        if not self.relative_path:
            raise ValueError("pipeline.output_targets[].relativePath is required")
        if not self.path:
            raise ValueError("pipeline.output_targets[].path is required")
        if self.storage not in _PIPELINE_OUTPUT_STORAGE_KINDS:
            raise ValueError(
                "pipeline.output_targets[].storage must be one of: "
                "directory, zip, file"
            )
        if self.storage == "zip" and not self.path.lower().endswith(".zip"):
            raise ValueError(
                "pipeline.output_targets[].path must end with .zip when storage='zip'"
            )
        if self.storage == "zip" and not self.relative_path.lower().endswith(".zip"):
            raise ValueError(
                "pipeline.output_targets[].relativePath must end with .zip "
                "when storage='zip'"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineOutputTargetConfig":
        raw_model_modality_id = data.get("modelModalityId")
        return cls(
            slot=data.get("slot", ""),
            model_modality_id=(
                None
                if raw_model_modality_id in (None, "")
                else int(raw_model_modality_id)
            ),
            dataset_type=data.get("datasetType", ""),
            relative_path=data.get("relativePath", ""),
            path=data.get("path", ""),
            storage=data.get("storage", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "slot": self.slot,
            "datasetType": self.dataset_type,
            "relativePath": self.relative_path,
            "storage": self.storage,
        }
        if self.model_modality_id is not None:
            result["modelModalityId"] = self.model_modality_id
        return result


@dataclass
class PipelineRuntimeConfig:
    """Pipeline-provided runtime context for transform outputs."""

    output_root: str | None = None
    outputs_manifest_path: str | None = None
    output_targets: list[PipelineOutputTargetConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen_slots: set[str] = set()
        for target in self.output_targets:
            if target.slot in seen_slots:
                raise ValueError(
                    f"pipeline.output_targets contains duplicate slot '{target.slot}'"
                )
            seen_slots.add(target.slot)

    def get_output_target(self, slot: str) -> PipelineOutputTargetConfig | None:
        for target in self.output_targets:
            if target.slot == slot:
                return target
        return None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineRuntimeConfig":
        output_targets_data = data.get("output_targets")
        if output_targets_data is None:
            parsed_targets: list[PipelineOutputTargetConfig] = []
        else:
            if not isinstance(output_targets_data, list):
                raise ValueError("pipeline.output_targets must be a list")
            parsed_targets = [
                PipelineOutputTargetConfig.from_dict(entry)
                for entry in output_targets_data
            ]

        output_root = data.get("output_root")
        outputs_manifest_path = data.get("outputs_manifest_path")
        return cls(
            output_root=output_root if isinstance(output_root, str) else None,
            outputs_manifest_path=(
                outputs_manifest_path
                if isinstance(outputs_manifest_path, str)
                else None
            ),
            output_targets=parsed_targets,
        )


def parse_pipeline_config(config: dict[str, Any]) -> PipelineRuntimeConfig | None:
    pipeline_data = config.get("pipeline")
    if pipeline_data is None:
        return None
    if not isinstance(pipeline_data, dict):
        raise ValueError("pipeline must be a JSON object")
    return PipelineRuntimeConfig.from_dict(pipeline_data)


def _write_pipeline_outputs_manifest(
    manifest_path: Path,
    output_targets: list[PipelineOutputTargetConfig],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "version": PIPELINE_OUTPUTS_MANIFEST_VERSION,
                "outputs": [target.to_dict() for target in output_targets],
            },
            handle,
            indent=2,
        )
        handle.write("\n")


def _set_stream_name(stream: Any, basename: str) -> None:
    try:
        stream.name = basename
    except Exception:
        pass


def _merge_top_level_mapping(target: dict[str, Any], overlay: dict[str, Any]) -> None:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            target[key].update(value)
        elif isinstance(value, dict):
            target[key] = dict(value)
        else:
            target[key] = value


def _patch_output_index(
    dataset_writer: DatasetWriter | ZipDatasetWriter,
    overrides: dict[str, Any],
    *,
    filename: str = DS_CRAWLER_INDEX_FILENAME,
) -> None:
    if not overrides:
        return

    metadata_scope = getattr(dataset_writer, "_metadata_scope", None)
    if isinstance(dataset_writer, ZipDatasetWriter):
        entry_name = get_metadata_entry_name(
            filename,
            metadata_scope=metadata_scope,
        )
        archive_path = Path(dataset_writer.root)
        replacement_name = archive_path.with_suffix(archive_path.suffix + ".tmp")
        with zipfile.ZipFile(archive_path, "r") as source_zip:
            with zipfile.ZipFile(replacement_name, "w") as target_zip:
                replaced = False
                for info in source_zip.infolist():
                    data = source_zip.read(info.filename)
                    if info.filename == entry_name and not replaced:
                        payload = json.loads(data.decode("utf-8"))
                        _merge_top_level_mapping(payload, overrides)
                        data = json.dumps(payload, indent=2).encode("utf-8")
                        replaced = True
                    target_zip.writestr(info, data)
        replacement_name.replace(archive_path)
        return

    output_path = Path(dataset_writer.root) / METADATA_DIR
    if metadata_scope is not None:
        output_path = output_path / metadata_scope
    output_path = output_path / filename
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    _merge_top_level_mapping(payload, overrides)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _head_mapping(index_output: Mapping[str, Any]) -> dict[str, Any] | None:
    head = index_output.get("head")
    return head if isinstance(head, dict) else None


def _modality_key(index_output: Mapping[str, Any]) -> str | None:
    raw_type = index_output.get("type")
    if isinstance(raw_type, str) and raw_type:
        return raw_type
    head = _head_mapping(index_output)
    if head is None:
        return None
    modality = head.get("modality")
    if isinstance(modality, dict):
        key = modality.get("key")
        if isinstance(key, str) and key:
            return key
    return None


def _modality_meta(index_output: Mapping[str, Any]) -> dict[str, Any]:
    meta = index_output.get("meta")
    if isinstance(meta, dict):
        return dict(meta)
    head = _head_mapping(index_output)
    modality = head.get("modality") if head else None
    if isinstance(modality, dict) and isinstance(modality.get("meta"), dict):
        return dict(modality["meta"])
    return {}


def _legacy_index_overrides(index_output: Mapping[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}

    meta = index_output.get("meta")
    if isinstance(meta, dict):
        overrides["meta"] = dict(meta)

    euler_train = index_output.get("euler_train")
    if isinstance(euler_train, dict):
        train_overlay = dict(euler_train)
    else:
        train_overlay = {}
        head = _head_mapping(index_output)
        addons = head.get("addons") if head else None
        if isinstance(addons, dict) and isinstance(addons.get("euler_train"), dict):
            train_overlay.update(addons["euler_train"])
    modality_key = _modality_key(index_output)
    if modality_key and "modality_type" not in train_overlay:
        train_overlay["modality_type"] = modality_key
    if train_overlay:
        overrides["euler_train"] = train_overlay

    return overrides


def _set_head_meta(index_output: dict[str, Any], meta: dict[str, Any]) -> None:
    head = index_output.get("head")
    if not isinstance(head, dict):
        return
    new_head = json.loads(json.dumps(head))
    new_head.setdefault("modality", {})
    new_head["modality"]["meta"] = dict(meta)
    index_output["head"] = new_head


def _split_hierarchy_value(key: str, separator: str | None) -> str:
    if separator and separator in key:
        return key.split(separator, 1)[1]
    return key


def _relative_path_from_full_id(
    full_id: str,
    basename: str,
    *,
    separator: str | None,
) -> str:
    parts = [p for p in full_id.split("/") if p]
    parent_parts = parts[:-1]
    dir_parts = [_split_hierarchy_value(part, separator) for part in parent_parts]
    return str(Path(*dir_parts) / basename) if dir_parts else basename


class LegacyOutputBackend:
    """Legacy output backend used by direct transform invocation."""

    is_source_backed = False

    def __init__(self, root: str | Path) -> None:
        self._writer = OutputWriter(root)
        self.root = self._writer.root

    def write(
        self,
        sample: dict[str, Any],
        value: Any,
        *,
        default_path: Path | None = None,
        output_full_id: str | None = None,
        output_basename: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Path:
        del output_full_id, output_basename, attributes
        if default_path is None:
            raise ValueError("default_path is required for legacy outputs")

        self._writer.mkdir(default_path.parent)
        suffix = default_path.suffix.lower()
        if suffix == ".npy":
            self._writer.save_depth_npy(default_path, value)
            return default_path
        if suffix in _IMAGE_EXTENSIONS:
            self._writer.save_image(default_path, value)
            return default_path
        raise ValueError(f"Unsupported legacy output suffix: {suffix}")

    def write_json(self, path: Path, data: dict[str, Any]) -> None:
        self._writer.mkdir(path.parent)
        self._writer.write_json(path, data)

    def finalize(self) -> None:
        self._writer.close()


class SourceBackedOutputBackend:
    """Source-aware output backend mirroring an input modality's writer metadata."""

    is_source_backed = True

    def __init__(
        self,
        *,
        source_modality: str,
        root: Path,
        dataset_writer: DatasetWriter | ZipDatasetWriter,
        modality_writer: Any,
        modality_meta: dict[str, Any] | None,
        output_extension: str | None = None,
        pipeline_manifest_path: Path | None = None,
        pipeline_manifest_targets: list[PipelineOutputTargetConfig] | None = None,
        index_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.source_modality = source_modality
        self.root = root
        self.dataset_writer = dataset_writer
        self.modality_writer = modality_writer
        self.modality_meta = modality_meta
        self.output_extension = output_extension
        self.pipeline_manifest_path = pipeline_manifest_path
        self.pipeline_manifest_targets = pipeline_manifest_targets or []
        self.index_overrides = index_overrides or {}

    def write(
        self,
        sample: dict[str, Any],
        value: Any,
        *,
        default_path: Path | None = None,
        output_full_id: str | None = None,
        output_basename: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Path:
        del default_path

        sample_id = sample.get("id", "?")
        full_id = str(output_full_id or sample.get("full_id") or f"/{sample_id}")
        meta = sample.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(
                "Source-backed outputs require samples from MultiModalDataset "
                "so sample['meta'] is available."
            )

        source_meta = meta.get(self.source_modality)
        if not isinstance(source_meta, dict) or "path" not in source_meta:
            raise ValueError(
                f"Source-backed output for modality '{self.source_modality}' "
                "requires sample['meta'][source_modality]['path']."
            )

        source_path = Path(str(source_meta["path"]))
        if output_basename is not None:
            basename = output_basename
            relative_path = _relative_path_from_full_id(
                full_id,
                basename,
                separator=getattr(self.dataset_writer, "_separator", None),
            )
        elif self.output_extension is not None:
            basename = source_path.stem + self.output_extension
            relative_path = str(source_path.with_suffix(self.output_extension))
        else:
            basename = source_path.name
            relative_path = str(source_path)
        source_meta_copy = dict(source_meta)
        source_attributes = source_meta_copy.get("attributes")
        if isinstance(source_attributes, dict):
            entry_attributes: dict[str, Any] | None = dict(source_attributes)
        else:
            entry_attributes = None
        if attributes:
            entry_attributes = {**(entry_attributes or {}), **attributes}
        source_entry_for_writer = dict(source_meta_copy)
        if output_full_id is not None or output_basename is not None:
            # The caller is intentionally writing a new logical layout
            # (e.g. source sample -> augmentation).  Let DatasetWriter derive
            # properties from the new full_id/basename instead of copying the
            # source file's old path and basename captures.
            source_entry_for_writer.pop("path_properties", None)
            source_entry_for_writer.pop("basename_properties", None)
            source_entry_for_writer.pop("attributes", None)

        if isinstance(self.dataset_writer, ZipDatasetWriter):
            if supports_stream_target(self.modality_writer):
                with self.dataset_writer.open(
                    full_id,
                    basename,
                    source_entry=source_entry_for_writer,
                    attributes=entry_attributes,
                ) as stream:
                    _set_stream_name(stream, basename)
                    self.modality_writer(stream, value, self.modality_meta)
            else:
                with tempfile.TemporaryDirectory() as tmpdir:
                    temp_path = Path(tmpdir) / basename
                    self.modality_writer(str(temp_path), value, self.modality_meta)
                    self.dataset_writer.write(
                        full_id,
                        basename,
                        temp_path.read_bytes(),
                        source_entry=source_entry_for_writer,
                        attributes=entry_attributes,
                    )
            return Path(f"{self.dataset_writer.root}::{relative_path}")

        target_path = self.dataset_writer.get_path(
            full_id,
            basename,
            source_entry=source_entry_for_writer,
            attributes=entry_attributes,
        )
        self.modality_writer(str(target_path), value, self.modality_meta)
        return target_path

    def set_hierarchy_separator(self, separator: str) -> None:
        """Set the writer hierarchy separator used for future entries."""
        setattr(self.dataset_writer, "_separator", separator)

    def add_head_addon(self, name: str, payload: dict[str, Any]) -> None:
        """Add a dataset-head addon before the writer saves its artifacts."""
        head = getattr(self.dataset_writer, "_dataset_head", None)
        addons = getattr(head, "addons", None)
        if not isinstance(addons, dict):
            raise RuntimeError("Unsupported dataset writer head object")
        addons[name] = dict(payload)
        self.index_overrides[name] = dict(payload)

    def write_json(self, path: Path, data: dict[str, Any]) -> None:
        raise RuntimeError(
            "Source-backed outputs do not support auxiliary JSON sidecars."
        )

    def finalize(self) -> None:
        self.dataset_writer.save_index(filename=DS_CRAWLER_INDEX_FILENAME)
        _patch_output_index(
            self.dataset_writer,
            self.index_overrides,
            filename=DS_CRAWLER_INDEX_FILENAME,
        )
        if self.pipeline_manifest_path and self.pipeline_manifest_targets:
            _write_pipeline_outputs_manifest(
                self.pipeline_manifest_path,
                self.pipeline_manifest_targets,
            )


def _resolve_output_slot(
    config: dict[str, Any],
    transform_class: type,
) -> str | None:
    explicit_slot = config.get("output_slot")
    if explicit_slot is not None:
        if not isinstance(explicit_slot, str) or not explicit_slot:
            raise ValueError("output_slot must be a non-empty string")
        return explicit_slot

    output_slot = getattr(transform_class, "OUTPUT_SLOT", None)
    if isinstance(output_slot, str) and output_slot:
        return output_slot

    source_modality = getattr(transform_class, "SOURCE_MODALITY", None)
    if isinstance(source_modality, str) and source_modality:
        return source_modality

    return None


def _select_pipeline_target(
    config: dict[str, Any],
    pipeline: PipelineRuntimeConfig | None,
    transform_class: type,
) -> PipelineOutputTargetConfig | None:
    if pipeline is None or not pipeline.output_targets:
        return None

    slot = _resolve_output_slot(config, transform_class)
    if slot is not None:
        target = pipeline.get_output_target(slot)
        if target is not None:
            return target
        if config.get("output_slot") is not None:
            raise ValueError(
                f"pipeline.output_targets does not contain slot '{slot}'"
            )

    # Auxiliary slot names declared by the transform are reserved for
    # OUTPUT_SLOT_SPECS and routed by prepare_output_backends; ignore them
    # when picking the *primary* target.  This lets pipeline configs use
    # arbitrary slot aliases (e.g. ``"fog"`` for the primary RGB output)
    # alongside named auxiliary outputs.
    aux_slots = set((getattr(transform_class, "OUTPUT_SLOT_SPECS", None) or {}).keys())
    primary_candidates = [
        t for t in pipeline.output_targets if t.slot not in aux_slots
    ]

    if len(primary_candidates) == 1:
        return primary_candidates[0]

    if not primary_candidates:
        raise ValueError(
            f"pipeline.output_targets has no primary target for "
            f"{transform_class.__name__}; only auxiliary slots present "
            f"({sorted(aux_slots)}). Add a target for the primary output."
        )

    raise ValueError(
        "pipeline.output_targets contains multiple primary entries "
        f"({[t.slot for t in primary_candidates]}); set top-level "
        "'output_slot' to select the target for this transform."
    )


def _resolve_output_root(
    config: dict[str, Any],
    pipeline: PipelineRuntimeConfig | None,
    pipeline_target: PipelineOutputTargetConfig | None,
) -> Path:
    if pipeline_target is not None:
        if pipeline_target.storage == "file":
            raise ValueError(
                f"Pipeline output target '{pipeline_target.slot}' uses "
                "unsupported storage='file'"
            )
        return Path(pipeline_target.path)

    output_path = config.get("output_path")
    if not output_path and pipeline and pipeline.output_root:
        output_path = pipeline.output_root

    if not output_path:
        raise ValueError(
            "output_path is required unless provided via pipeline.output_root "
            "or pipeline.output_targets[].path"
        )

    return Path(str(output_path))


def prepare_output_backend(
    config: dict[str, Any],
    dataset: MultiModalDataset,
    transform_class: type,
) -> SourceBackedOutputBackend:
    """Create a source-backed output backend for a transform run.

    Used for the *primary* output slot.  Transforms with auxiliary outputs
    should use :func:`prepare_output_backends` (plural).
    """

    source_modality = getattr(transform_class, "SOURCE_MODALITY", None)
    if not isinstance(source_modality, str) or not source_modality:
        raise ValueError(
            f"{transform_class.__name__} does not declare SOURCE_MODALITY"
        )

    pipeline = parse_pipeline_config(config)
    pipeline_target = _select_pipeline_target(config, pipeline, transform_class)
    root = _resolve_output_root(config, pipeline, pipeline_target)
    zip_mode = (
        pipeline_target.storage == "zip"
        if pipeline_target is not None
        else root.suffix.lower() == ".zip"
    )

    try:
        modality_writer = dataset.get_writer(source_modality)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"No euler-loading writer is available for source modality "
            f"'{source_modality}': {exc}"
        ) from exc

    index_output = dataset.get_modality_index(source_modality)
    meta_overrides = getattr(transform_class, "OUTPUT_INDEX_META_OVERRIDES", None)
    if isinstance(meta_overrides, dict) and meta_overrides:
        merged_meta = _modality_meta(index_output)
        merged_meta.update(meta_overrides)
        index_output = dict(index_output)
        index_output["meta"] = merged_meta
        _set_head_meta(index_output, merged_meta)

    dataset_writer = create_dataset_writer_from_index(
        index_output=index_output,
        root=root,
        zip=zip_mode,
    )

    pipeline_manifest_path = None
    if (
        pipeline is not None
        and pipeline.outputs_manifest_path
        and pipeline_target is not None
    ):
        pipeline_manifest_path = Path(pipeline.outputs_manifest_path)

    return SourceBackedOutputBackend(
        source_modality=source_modality,
        root=root,
        dataset_writer=dataset_writer,
        modality_writer=modality_writer,
        modality_meta=_modality_meta(index_output),
        pipeline_manifest_path=pipeline_manifest_path,
        pipeline_manifest_targets=[pipeline_target] if pipeline_target else [],
        index_overrides=_legacy_index_overrides(index_output),
    )


def _build_auxiliary_backend(
    *,
    spec: OutputSlotSpec,
    pipeline_target: PipelineOutputTargetConfig,
    dataset: MultiModalDataset,
) -> SourceBackedOutputBackend:
    """Create a backend for an auxiliary slot using its OutputSlotSpec.

    The auxiliary backend does not own the pipeline manifest — that is
    aggregated and written by the primary backend.
    """

    if pipeline_target.storage == "file":
        raise ValueError(
            f"Pipeline output target '{pipeline_target.slot}' uses "
            "unsupported storage='file'"
        )

    root = Path(pipeline_target.path)
    zip_mode = pipeline_target.storage == "zip"

    base_index = dataset.get_modality_index(spec.source_modality)
    index_output = _build_auxiliary_index(base_index, spec)

    dataset_writer = create_dataset_writer_from_index(
        index_output=index_output,
        root=root,
        zip=zip_mode,
    )

    modality_meta = spec.meta
    if modality_meta is None:
        head = index_output.get("head") or {}
        modality_meta = (head.get("modality") or {}).get("meta")
        if modality_meta is None:
            modality_meta = index_output.get("meta")

    return SourceBackedOutputBackend(
        source_modality=spec.source_modality,
        root=root,
        dataset_writer=dataset_writer,
        modality_writer=spec.writer,
        modality_meta=modality_meta,
        output_extension=spec.output_extension,
        pipeline_manifest_path=None,
        pipeline_manifest_targets=[],
        index_overrides=_legacy_index_overrides(index_output),
    )


def _build_auxiliary_index(
    base_index: dict[str, Any], spec: OutputSlotSpec
) -> dict[str, Any]:
    """Apply ``spec.index_overlay`` to a copy of ``base_index``.

    The overlay's recognised keys map to fields used by ds-crawler's writer
    construction.  ``name`` / ``type`` rewrite the dataset id+name and
    modality key on both the contract head and the legacy top-level fields;
    ``meta`` overrides the modality's meta dict; ``euler_train`` /
    ``euler_loading`` replace those addon entries.  Any other overlay keys
    are passed through as top-level fields for the legacy writer path.
    """

    overlay = dict(spec.index_overlay)
    index_output: dict[str, Any] = {**base_index}

    head = base_index.get("head")
    if isinstance(head, dict):
        new_head = json.loads(json.dumps(head))  # deep copy via JSON
        new_head.setdefault("dataset", {})
        new_head.setdefault("modality", {})
        new_head.setdefault("addons", {})

        if "name" in overlay:
            name = overlay["name"]
            new_head["dataset"]["id"] = name
            new_head["dataset"]["name"] = name
        if "type" in overlay:
            new_head["modality"]["key"] = overlay["type"]
        if "meta" in overlay:
            new_head["modality"]["meta"] = dict(overlay["meta"])
        if "euler_train" in overlay:
            new_head["addons"]["euler_train"] = dict(overlay["euler_train"])
        if "euler_loading" in overlay:
            new_head["addons"]["euler_loading"] = dict(overlay["euler_loading"])

        index_output["head"] = new_head

    # Legacy top-level fields used by the non-contract writer construction
    # path.  Preserved alongside the head for compatibility.
    for key, value in overlay.items():
        if isinstance(value, dict):
            index_output[key] = dict(value)
        else:
            index_output[key] = value

    return index_output


def _resolve_primary_slot(transform_class: type) -> str:
    """Return the *primary* slot name declared by *transform_class*."""

    output_slots = getattr(transform_class, "OUTPUT_SLOTS", None)
    if output_slots:
        return output_slots[0]

    output_slot = getattr(transform_class, "OUTPUT_SLOT", None)
    if isinstance(output_slot, str) and output_slot:
        return output_slot

    source_modality = getattr(transform_class, "SOURCE_MODALITY", None)
    if isinstance(source_modality, str) and source_modality:
        return source_modality

    raise ValueError(
        f"{transform_class.__name__} declares no output slot "
        "(set OUTPUT_SLOT, OUTPUT_SLOTS, or SOURCE_MODALITY)"
    )


def prepare_output_backends(
    config: dict[str, Any],
    dataset: MultiModalDataset,
    transform_class: type,
) -> dict[str, SourceBackedOutputBackend]:
    """Create per-slot output backends for *transform_class*.

    Returns ``{slot_name: backend}``.  The primary slot (the first entry of
    ``OUTPUT_SLOTS``, falling back to ``OUTPUT_SLOT`` / ``SOURCE_MODALITY``) is
    always present.  Auxiliary slots declared in
    :attr:`Transform.OUTPUT_SLOT_SPECS` are included only when the dataset
    config's ``pipeline.output_targets`` contains a matching entry; otherwise
    the slot is silently omitted (auxiliary outputs are opt-in).

    The returned dict's iteration order matches the declared
    ``OUTPUT_SLOTS`` order.
    """

    primary_slot = _resolve_primary_slot(transform_class)
    primary_backend = prepare_output_backend(config, dataset, transform_class)
    backends: dict[str, SourceBackedOutputBackend] = {primary_slot: primary_backend}

    slot_specs = getattr(transform_class, "OUTPUT_SLOT_SPECS", None) or {}
    pipeline = parse_pipeline_config(config)

    declared_slots = getattr(transform_class, "OUTPUT_SLOTS", ()) or ()
    aux_slots = [s for s in declared_slots if s != primary_slot]
    if pipeline is not None and slot_specs:
        for slot in aux_slots:
            spec = slot_specs.get(slot)
            if spec is None:
                continue
            target = pipeline.get_output_target(slot)
            if target is None:
                continue
            backends[slot] = _build_auxiliary_backend(
                spec=spec,
                pipeline_target=target,
                dataset=dataset,
            )

    # Aggregate every slot we actually wrote into the manifest the primary
    # backend will emit, so a single manifest documents the full set.
    if (
        pipeline is not None
        and pipeline.outputs_manifest_path
        and len(backends) > 1
    ):
        manifest_targets: list[PipelineOutputTargetConfig] = list(
            primary_backend.pipeline_manifest_targets
        )
        for slot in aux_slots:
            if slot not in backends:
                continue
            target = pipeline.get_output_target(slot)
            if target is not None:
                manifest_targets.append(target)
        primary_backend.pipeline_manifest_targets = manifest_targets

    return backends
