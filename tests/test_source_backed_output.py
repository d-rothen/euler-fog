"""Tests for source-backed output writing and pipeline routing."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np

from ds_crawler import DatasetWriter
from euler_loading import Modality, MultiModalDataset
from euler_loading.loaders.cpu.generic_dense_depth import (
    depth as load_depth,
    rgb as load_rgb,
    sky_mask as load_sky_mask,
    write_depth,
    write_rgb,
    write_sky_mask,
)
from euler_loading.loaders.cpu.vkitti2 import (
    read_intrinsics,
    write_intrinsics,
)

from euler_preprocess.common.output import prepare_output_backend
from euler_preprocess.fog.transform import FogTransform
from euler_preprocess.radial.transform import RadialTransform


def _write_rgb_dataset(root: Path) -> None:
    writer = DatasetWriter(
        root,
        name="synthetic_rgb",
        type="rgb",
        euler_train={"used_as": "input", "modality_type": "rgb"},
        separator=None,
        meta={"range": [0, 255]},
        euler_loading={"loader": "generic_dense_depth", "function": "rgb"},
    )
    rgb = np.dstack([
        np.full((4, 6), 32, dtype=np.uint8),
        np.full((4, 6), 96, dtype=np.uint8),
        np.full((4, 6), 160, dtype=np.uint8),
    ])
    path = writer.get_path("Scene01/Camera_0/00001", "00001.png")
    write_rgb(str(path), rgb)
    writer.save_index()


def _write_depth_dataset(root: Path) -> None:
    writer = DatasetWriter(
        root,
        name="synthetic_depth",
        type="depth",
        euler_train={"used_as": "input", "modality_type": "depth"},
        separator=None,
        meta={
            "radial_depth": False,
            "scale_to_meters": 1.0,
            "range": [0, 1000],
        },
        euler_loading={"loader": "generic_dense_depth", "function": "depth"},
    )
    depth = np.full((4, 6), 25.0, dtype=np.float32)
    path = writer.get_path("Scene01/Camera_0/00001", "00001.npy")
    write_depth(str(path), depth)
    writer.save_index()


def _write_sky_mask_dataset(root: Path) -> None:
    writer = DatasetWriter(
        root,
        name="synthetic_sky_mask",
        type="sky_mask",
        euler_train={"used_as": "condition", "modality_type": "sky_mask"},
        separator=None,
        meta={"sky_mask": [255, 255, 255]},
        euler_loading={"loader": "generic_dense_depth", "function": "sky_mask"},
    )
    mask = np.zeros((4, 6), dtype=bool)
    mask[0, :] = True
    path = writer.get_path("Scene01/Camera_0/00001", "00001.png")
    write_sky_mask(str(path), mask, {"sky_mask": [255, 255, 255]})
    writer.save_index()


def _write_intrinsics_dataset(root: Path) -> None:
    writer = DatasetWriter(
        root,
        name="synthetic_intrinsics",
        type="intrinsics",
        euler_train={"used_as": "condition", "modality_type": "intrinsics"},
        separator=None,
        euler_loading={"loader": "vkitti2", "function": "read_intrinsics"},
    )
    intrinsics = np.array(
        [[500.0, 0.0, 2.0], [0.0, 500.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    path = writer.get_path("Scene01/intrinsics", "intrinsics.txt")
    write_intrinsics(str(path), intrinsics)
    writer.save_index()


def _make_dataset(tmp_path: Path) -> MultiModalDataset:
    rgb_root = tmp_path / "rgb"
    depth_root = tmp_path / "depth"
    sky_root = tmp_path / "sky_mask"
    intrinsics_root = tmp_path / "intrinsics"

    _write_rgb_dataset(rgb_root)
    _write_depth_dataset(depth_root)
    _write_sky_mask_dataset(sky_root)
    _write_intrinsics_dataset(intrinsics_root)

    return MultiModalDataset(
        modalities={
            "rgb": Modality(str(rgb_root), loader=load_rgb, writer=write_rgb),
            "depth": Modality(str(depth_root), loader=load_depth, writer=write_depth),
            "semantic_segmentation": Modality(
                str(sky_root),
                loader=load_sky_mask,
                writer=write_sky_mask,
            ),
        },
        hierarchical_modalities={
            "intrinsics": Modality(
                str(intrinsics_root),
                loader=read_intrinsics,
                writer=write_intrinsics,
            ),
        },
    )


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _make_fog_config(path: Path) -> Path:
    return _write_json(
        path,
        {
            "airlight": "from_sky",
            "device": "cpu",
            "seed": 7,
            "models": {
                "uniform": {
                    "visibility_m": {"dist": "constant", "value": 200.0},
                    "atmospheric_light": "from_sky",
                }
            },
            "selection": {"mode": "fixed", "model": "uniform"},
        },
    )


def _make_radial_config(path: Path) -> Path:
    return _write_json(path, {})


def test_fog_uses_source_backed_writer_metadata(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    config = {"output_path": str(tmp_path / "foggy")}
    output_backend = prepare_output_backend(config, dataset, FogTransform)
    transform = FogTransform(
        config_path=str(_make_fog_config(tmp_path / "fog_config.json")),
        out_path=str(output_backend.root),
        output_backend=output_backend,
    )

    saved_paths = transform.run(dataset)

    assert saved_paths == [tmp_path / "foggy" / "Scene01" / "Camera_0" / "00001.png"]
    assert saved_paths[0].exists()

    output_index = json.loads(
        (tmp_path / "foggy" / ".ds_crawler" / "index.json").read_text()
    )
    assert output_index["euler_train"]["modality_type"] == "rgb"
    assert output_index["euler_loading"]["function"] == "rgb"
    assert output_index["meta"]["range"] == [0, 255]

    reloaded = MultiModalDataset(
        modalities={
            "rgb": Modality(str(tmp_path / "foggy"), loader=load_rgb),
        },
    )
    assert len(reloaded) == 1


def test_pipeline_target_controls_root_and_manifest(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    pipeline_root = tmp_path / "pipeline_root"
    manifest_path = pipeline_root / ".euler_pipeline" / "pipeline_outputs.json"
    config = {
        "pipeline": {
            "output_root": str(pipeline_root),
            "outputs_manifest_path": str(manifest_path),
            "output_targets": [
                {
                    "slot": "rgb",
                    "modelModalityId": 71,
                    "datasetType": "rgb",
                    "relativePath": "foggy_rgb",
                    "path": str(pipeline_root / "foggy_rgb"),
                    "storage": "directory",
                }
            ],
        }
    }
    output_backend = prepare_output_backend(config, dataset, FogTransform)
    transform = FogTransform(
        config_path=str(_make_fog_config(tmp_path / "fog_pipeline_config.json")),
        out_path=str(output_backend.root),
        output_backend=output_backend,
    )

    transform.run(dataset)

    assert (pipeline_root / "foggy_rgb" / "Scene01" / "Camera_0" / "00001.png").exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest == {
        "version": 1,
        "outputs": [
            {
                "slot": "rgb",
                "modelModalityId": 71,
                "datasetType": "rgb",
                "relativePath": "foggy_rgb",
                "storage": "directory",
            }
        ],
    }


def test_pipeline_target_allows_missing_model_modality_id(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    pipeline_root = tmp_path / "pipeline_root_optional_id"
    manifest_path = pipeline_root / ".euler_pipeline" / "pipeline_outputs.json"
    config = {
        "pipeline": {
            "output_root": str(pipeline_root),
            "outputs_manifest_path": str(manifest_path),
            "output_targets": [
                {
                    "slot": "rgb",
                    "datasetType": "rgb",
                    "relativePath": "foggy_rgb",
                    "path": str(pipeline_root / "foggy_rgb"),
                    "storage": "directory",
                }
            ],
        }
    }
    output_backend = prepare_output_backend(config, dataset, FogTransform)
    transform = FogTransform(
        config_path=str(_make_fog_config(tmp_path / "fog_pipeline_optional_id.json")),
        out_path=str(output_backend.root),
        output_backend=output_backend,
    )

    transform.run(dataset)

    manifest = json.loads(manifest_path.read_text())
    assert manifest == {
        "version": 1,
        "outputs": [
            {
                "slot": "rgb",
                "datasetType": "rgb",
                "relativePath": "foggy_rgb",
                "storage": "directory",
            }
        ],
    }


def test_fog_source_backed_output_can_write_zip(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    config = {"output_path": str(tmp_path / "foggy_rgb.zip")}
    output_backend = prepare_output_backend(config, dataset, FogTransform)
    transform = FogTransform(
        config_path=str(_make_fog_config(tmp_path / "fog_zip_config.json")),
        out_path=str(output_backend.root),
        output_backend=output_backend,
    )

    saved_paths = transform.run(dataset)

    assert saved_paths == [Path(f"{tmp_path / 'foggy_rgb.zip'}::Scene01/Camera_0/00001.png")]

    with zipfile.ZipFile(tmp_path / "foggy_rgb.zip", "r") as zf:
        names = set(zf.namelist())
        assert "Scene01/Camera_0/00001.png" in names
        assert ".ds_crawler/index.json" in names
        output_index = json.loads(zf.read(".ds_crawler/index.json"))

    assert output_index["euler_train"]["modality_type"] == "rgb"
    assert output_index["euler_loading"]["function"] == "rgb"


def test_radial_source_backed_output_sets_radial_depth_metadata(
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(tmp_path)
    config = {"output_path": str(tmp_path / "radial")}
    output_backend = prepare_output_backend(config, dataset, RadialTransform)
    transform = RadialTransform(
        config_path=str(_make_radial_config(tmp_path / "radial_config.json")),
        out_path=str(output_backend.root),
        output_backend=output_backend,
    )

    saved_paths = transform.run(dataset)

    assert saved_paths == [tmp_path / "radial" / "Scene01" / "Camera_0" / "00001.npy"]
    result = np.load(saved_paths[0])
    assert result.shape == (4, 6)

    output_index = json.loads(
        (tmp_path / "radial" / ".ds_crawler" / "index.json").read_text()
    )
    assert output_index["euler_loading"]["function"] == "depth"
    assert output_index["meta"]["radial_depth"] is True
    assert output_index["meta"]["scale_to_meters"] == 1.0
    assert output_index["meta"]["range"] == [0, 1000]
