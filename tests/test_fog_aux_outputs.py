"""Tests for FogTransform writing scattering-coefficient and airlight maps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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

from euler_loading.loaders.cpu.generic import (
    map_2d as load_map_2d,
    map_3d as load_map_3d,
)

from euler_preprocess.common.output import prepare_output_backends
import euler_preprocess.fog.capture as capture_module
from euler_preprocess.fog.capture import CaptureArtifactPipeline, CaptureContext
from euler_preprocess.fog.models import visibility_to_k
from euler_preprocess.fog.transform import (
    ATMOSPHERIC_LIGHT_SLOT,
    SCATTERING_COEFFICIENT_SLOT,
    FogTransform,
)


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


def _make_dataset(tmp_path: Path) -> MultiModalDataset:
    rgb_root = tmp_path / "rgb"
    depth_root = tmp_path / "depth"
    sky_root = tmp_path / "sky_mask"

    _write_rgb_dataset(rgb_root)
    _write_depth_dataset(depth_root)
    _write_sky_mask_dataset(sky_root)

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
    )


def _write_fog_config(path: Path, *, visibility_m: float = 200.0) -> Path:
    cfg = {
        "airlight": "from_sky",
        "device": "cpu",
        "seed": 7,
        "contrast_threshold": 0.05,
        "models": {
            "uniform": {
                "visibility_m": {"dist": "constant", "value": visibility_m},
                "atmospheric_light": [0.4, 0.5, 0.6],
            }
        },
        "selection": {"mode": "fixed", "model": "uniform"},
    }
    path.write_text(json.dumps(cfg))
    return path


def _write_stepped_fog_config(path: Path) -> Path:
    cfg = {
        "airlight": "from_sky",
        "device": "cpu",
        "seed": 7,
        "contrast_threshold": 0.05,
        "augmentations": {
            "visibility_m": [10.0, 20.0],
            "atmospheric_light": [0.4, 0.5, 0.6],
        },
    }
    path.write_text(json.dumps(cfg))
    return path


def _build_pipeline_config(
    pipeline_root: Path,
    manifest_path: Path,
    *,
    include_scattering: bool,
    include_airlight: bool,
) -> dict:
    output_targets = [
        {
            "slot": "rgb",
            "datasetType": "rgb",
            "relativePath": "foggy_rgb",
            "path": str(pipeline_root / "foggy_rgb"),
            "storage": "directory",
        }
    ]
    if include_scattering:
        output_targets.append(
            {
                "slot": SCATTERING_COEFFICIENT_SLOT,
                "datasetType": "scattering_coefficient",
                "relativePath": "scattering",
                "path": str(pipeline_root / "scattering"),
                "storage": "directory",
            }
        )
    if include_airlight:
        output_targets.append(
            {
                "slot": ATMOSPHERIC_LIGHT_SLOT,
                "datasetType": "atmospheric_light",
                "relativePath": "airlight",
                "path": str(pipeline_root / "airlight"),
                "storage": "directory",
            }
        )
    return {
        "pipeline": {
            "output_root": str(pipeline_root),
            "outputs_manifest_path": str(manifest_path),
            "output_targets": output_targets,
        }
    }


def test_aux_slots_omitted_when_pipeline_has_no_targets(tmp_path: Path) -> None:
    """Without aux output_targets, prepare_output_backends only yields rgb."""
    dataset = _make_dataset(tmp_path)
    config = {"output_path": str(tmp_path / "foggy")}
    backends = prepare_output_backends(config, dataset, FogTransform)
    assert set(backends.keys()) == {"rgb"}


def test_writes_scattering_and_airlight_maps(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    pipeline_root = tmp_path / "pipeline_root"
    manifest_path = pipeline_root / ".euler_pipeline" / "pipeline_outputs.json"
    config = _build_pipeline_config(
        pipeline_root,
        manifest_path,
        include_scattering=True,
        include_airlight=True,
    )

    visibility_m = 200.0
    backends = prepare_output_backends(config, dataset, FogTransform)
    assert set(backends.keys()) == {
        "rgb",
        SCATTERING_COEFFICIENT_SLOT,
        ATMOSPHERIC_LIGHT_SLOT,
    }

    transform = FogTransform(
        config_path=str(_write_fog_config(tmp_path / "fog_cfg.json", visibility_m=visibility_m)),
        out_path=str(backends["rgb"].root),
        output_backends=backends,
    )
    saved_paths = transform.run(dataset)

    rgb_path = pipeline_root / "foggy_rgb" / "Scene01" / "Camera_0" / "00001.png"
    assert rgb_path.exists()
    assert saved_paths == [rgb_path]

    scattering_path = (
        pipeline_root / "scattering" / "Scene01" / "Camera_0" / "00001.npy"
    )
    airlight_path = pipeline_root / "airlight" / "Scene01" / "Camera_0" / "00001.npy"
    assert scattering_path.exists()
    assert airlight_path.exists()

    # Round-trip via the canonical loaders so layout conventions match.
    k_map = load_map_2d(str(scattering_path))
    ls_map = load_map_3d(str(airlight_path))

    assert k_map.shape == (4, 6)
    assert k_map.dtype == np.float32
    expected_k = visibility_to_k(visibility_m, 0.05)
    np.testing.assert_allclose(k_map, expected_k, atol=1e-6)

    assert ls_map.shape == (4, 6, 3)
    assert ls_map.dtype == np.float32
    expected_ls = np.broadcast_to(
        np.array([0.4, 0.5, 0.6], dtype=np.float32), (4, 6, 3)
    )
    np.testing.assert_allclose(ls_map, expected_ls, atol=1e-6)

    # And verify the on-disk layout matches the writer's contract:
    # scattering = (H, W) directly, airlight = (C, H, W).
    raw_scattering = np.load(scattering_path)
    assert raw_scattering.shape == (4, 6)
    raw_airlight = np.load(airlight_path)
    assert raw_airlight.shape == (3, 4, 6)


def test_stepped_augmentations_write_file_id_layout_and_attributes(
    tmp_path: Path,
) -> None:
    dataset = _make_dataset(tmp_path)
    pipeline_root = tmp_path / "pipeline_root_stepped"
    manifest_path = pipeline_root / ".euler_pipeline" / "pipeline_outputs.json"
    config = _build_pipeline_config(
        pipeline_root,
        manifest_path,
        include_scattering=True,
        include_airlight=True,
    )

    backends = prepare_output_backends(config, dataset, FogTransform)
    transform = FogTransform(
        config_path=str(_write_stepped_fog_config(tmp_path / "fog_stepped.json")),
        out_path=str(backends["rgb"].root),
        output_backends=backends,
    )

    saved_paths = transform.run(dataset)

    assert saved_paths == [
        pipeline_root / "foggy_rgb" / "Scene01" / "Camera_0" / "00001" / "mor_10m.png",
        pipeline_root / "foggy_rgb" / "Scene01" / "Camera_0" / "00001" / "mor_20m.png",
    ]
    for path in saved_paths:
        assert path.exists()

    scattering_path = (
        pipeline_root
        / "scattering"
        / "Scene01"
        / "Camera_0"
        / "00001"
        / "mor_10m.npy"
    )
    airlight_path = (
        pipeline_root
        / "airlight"
        / "Scene01"
        / "Camera_0"
        / "00001"
        / "mor_10m.npy"
    )
    assert scattering_path.exists()
    assert airlight_path.exists()

    k_map = load_map_2d(str(scattering_path))
    expected_k = visibility_to_k(10.0, 0.05)
    np.testing.assert_allclose(k_map, expected_k, atol=1e-6)

    output_index = json.loads(
        (pipeline_root / "foggy_rgb" / ".ds_crawler" / "index.json").read_text()
    )
    node = output_index["dataset"]["children"]["Scene01"]["children"]["Camera_0"]
    file_id_node = node["children"]["file_id:00001"]
    entries = {entry["id"]: entry for entry in file_id_node["files"]}
    assert set(entries) == {"mor_10m", "mor_20m"}
    assert entries["mor_10m"]["path_properties"]["file_id"] == "00001"
    assert entries["mor_10m"]["basename_properties"]["ext"] == "png"
    attrs = entries["mor_10m"]["attributes"]["fog_augmentation"]
    assert attrs["id"] == "mor_10m"
    assert attrs["source_id"] == "00001"
    assert attrs["meteorological_visibility_m"] == 10.0
    assert attrs["model"] == "uniform"
    np.testing.assert_allclose(attrs["atmospheric_light"], [0.4, 0.5, 0.6])
    assert output_index["euler_layout"]["sample_axis"] == {
        "name": "file_id",
        "location": "hierarchy",
    }
    assert output_index["euler_layout"]["variant_axis"] == {
        "name": "fog_augmentation",
        "location": "file_id",
    }
    output_head = json.loads(
        (pipeline_root / "foggy_rgb" / ".ds_crawler" / "dataset-head.json").read_text()
    )
    assert output_head["addons"]["euler_layout"] == output_index["euler_layout"]


def test_only_scattering_target_writes_only_scattering(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    pipeline_root = tmp_path / "pipeline_root_scat_only"
    manifest_path = pipeline_root / ".euler_pipeline" / "pipeline_outputs.json"
    config = _build_pipeline_config(
        pipeline_root,
        manifest_path,
        include_scattering=True,
        include_airlight=False,
    )

    backends = prepare_output_backends(config, dataset, FogTransform)
    assert set(backends.keys()) == {"rgb", SCATTERING_COEFFICIENT_SLOT}

    transform = FogTransform(
        config_path=str(_write_fog_config(tmp_path / "fog_cfg.json")),
        out_path=str(backends["rgb"].root),
        output_backends=backends,
    )
    transform.run(dataset)

    assert (pipeline_root / "scattering" / "Scene01" / "Camera_0" / "00001.npy").exists()
    assert not (pipeline_root / "airlight").exists()


def test_pipeline_manifest_lists_all_active_slots(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    pipeline_root = tmp_path / "pipeline_root_manifest"
    manifest_path = pipeline_root / ".euler_pipeline" / "pipeline_outputs.json"
    config = _build_pipeline_config(
        pipeline_root,
        manifest_path,
        include_scattering=True,
        include_airlight=True,
    )

    backends = prepare_output_backends(config, dataset, FogTransform)
    transform = FogTransform(
        config_path=str(_write_fog_config(tmp_path / "fog_cfg.json")),
        out_path=str(backends["rgb"].root),
        output_backends=backends,
    )
    transform.run(dataset)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["version"] == 1
    slots = [target["slot"] for target in manifest["outputs"]]
    assert slots == ["rgb", SCATTERING_COEFFICIENT_SLOT, ATMOSPHERIC_LIGHT_SLOT]


def test_aux_outputs_carry_correct_index_metadata(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    pipeline_root = tmp_path / "pipeline_root_meta"
    manifest_path = pipeline_root / ".euler_pipeline" / "pipeline_outputs.json"
    config = _build_pipeline_config(
        pipeline_root,
        manifest_path,
        include_scattering=True,
        include_airlight=True,
    )

    backends = prepare_output_backends(config, dataset, FogTransform)
    transform = FogTransform(
        config_path=str(_write_fog_config(tmp_path / "fog_cfg.json")),
        out_path=str(backends["rgb"].root),
        output_backends=backends,
    )
    transform.run(dataset)

    scattering_index = json.loads(
        (pipeline_root / "scattering" / ".ds_crawler" / "index.json").read_text()
    )
    assert scattering_index["name"] == "scattering_coefficient"
    assert scattering_index["type"] == "map_2d"
    assert scattering_index["euler_loading"]["loader"] == "generic"
    assert scattering_index["euler_loading"]["function"] == "map_2d"
    assert scattering_index["euler_train"]["used_as"] == "target"

    airlight_index = json.loads(
        (pipeline_root / "airlight" / ".ds_crawler" / "index.json").read_text()
    )
    assert airlight_index["name"] == "atmospheric_light"
    assert airlight_index["type"] == "map_3d"
    assert airlight_index["euler_loading"]["loader"] == "generic"
    assert airlight_index["euler_loading"]["function"] == "map_3d"
    assert airlight_index["euler_train"]["used_as"] == "target"


def test_primary_slot_auto_selected_when_aliased(tmp_path: Path) -> None:
    """A pipeline target whose slot is *not* one of the aux slot names is
    automatically picked up as the primary RGB target — even when its slot
    name doesn't match the transform's primary slot (e.g. ``"fog"``).
    """
    dataset = _make_dataset(tmp_path)
    pipeline_root = tmp_path / "pipeline_root_alias"
    manifest_path = pipeline_root / ".euler_pipeline" / "pipeline_outputs.json"
    config = {
        "pipeline": {
            "output_root": str(pipeline_root),
            "outputs_manifest_path": str(manifest_path),
            "output_targets": [
                {
                    "slot": "fog",
                    "datasetType": "rgb",
                    "relativePath": "foggy_rgb.zip",
                    "path": str(pipeline_root / "foggy_rgb.zip"),
                    "storage": "zip",
                },
                {
                    "slot": ATMOSPHERIC_LIGHT_SLOT,
                    "datasetType": "rgb",
                    "relativePath": "atmospheric_light.zip",
                    "path": str(pipeline_root / "atmospheric_light.zip"),
                    "storage": "zip",
                },
                {
                    "slot": SCATTERING_COEFFICIENT_SLOT,
                    "datasetType": "rgb",
                    "relativePath": "scattering_coefficient.zip",
                    "path": str(pipeline_root / "scattering_coefficient.zip"),
                    "storage": "zip",
                },
            ],
        }
    }

    backends = prepare_output_backends(config, dataset, FogTransform)

    assert set(backends.keys()) == {
        "rgb",
        SCATTERING_COEFFICIENT_SLOT,
        ATMOSPHERIC_LIGHT_SLOT,
    }
    # Primary backend points at the "fog" target, not a literal "rgb" target.
    assert backends["rgb"].root == pipeline_root / "foggy_rgb.zip"

    transform = FogTransform(
        config_path=str(_write_fog_config(tmp_path / "fog_cfg.json")),
        out_path=str(backends["rgb"].root),
        output_backends=backends,
    )
    transform.run(dataset)

    import zipfile

    with zipfile.ZipFile(pipeline_root / "foggy_rgb.zip", "r") as zf:
        assert "Scene01/Camera_0/00001.png" in zf.namelist()
    with zipfile.ZipFile(pipeline_root / "scattering_coefficient.zip", "r") as zf:
        assert "Scene01/Camera_0/00001.npy" in zf.namelist()
    with zipfile.ZipFile(pipeline_root / "atmospheric_light.zip", "r") as zf:
        assert "Scene01/Camera_0/00001.npy" in zf.namelist()

    # Manifest still lists every active slot in declaration order.
    manifest = json.loads(manifest_path.read_text())
    slots = [target["slot"] for target in manifest["outputs"]]
    assert slots == ["fog", SCATTERING_COEFFICIENT_SLOT, ATMOSPHERIC_LIGHT_SLOT]


def test_capture_pipeline_empty_config_is_noop() -> None:
    pipeline = CaptureArtifactPipeline.from_config({"capture": {"stages": []}})
    image = np.full((2, 3, 3), 0.5, dtype=np.float32)

    result = pipeline.apply_np(image, context=CaptureContext(sample_id="sample"))

    assert result is image


def test_capture_pipeline_rejects_unknown_stages() -> None:
    with pytest.raises(ValueError, match="not_a_stage"):
        CaptureArtifactPipeline.from_config(
            {"capture": {"stages": [{"type": "not_a_stage"}]}}
        )


def test_capture_camera_preset_is_deterministic_and_changes_image() -> None:
    pipeline = CaptureArtifactPipeline.from_config({"capture": {"preset": "camera"}})
    x = np.linspace(0.1, 0.95, 40, dtype=np.float32)
    y = np.linspace(0.2, 0.9, 32, dtype=np.float32)
    image = np.dstack(
        [
            np.broadcast_to(x, (32, 40)),
            np.broadcast_to(y[:, None], (32, 40)),
            np.full((32, 40), 0.65, dtype=np.float32),
        ]
    )

    first = pipeline.apply_np(
        image,
        context=CaptureContext(rng=np.random.default_rng(123)),
    )
    second = pipeline.apply_np(
        image,
        context=CaptureContext(rng=np.random.default_rng(123)),
    )

    assert first.shape == image.shape
    assert first.dtype == np.float32
    assert 0.0 <= float(first.min()) <= float(first.max()) <= 1.0
    np.testing.assert_allclose(first, second)
    assert not np.allclose(first, image)


def test_capture_custom_stage_chain_can_resize_and_quantize() -> None:
    pipeline = CaptureArtifactPipeline.from_config(
        {
            "capture": {
                "stages": [
                    {"type": "exposure", "gain": 0.5},
                    {
                        "type": "transport",
                        "resize": [8, 6],
                        "bit_depth": 4,
                        "jpeg": {"enabled": False},
                    },
                ]
            }
        }
    )
    image = np.full((12, 16, 3), 0.8, dtype=np.float32)

    result = pipeline.apply_np(
        image,
        context=CaptureContext(rng=np.random.default_rng(7)),
    )

    assert result.shape == (6, 8, 3)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.round(result * 15.0) / 15.0)


def test_capture_torch_stage_dispatch_keeps_deterministic_stages_on_torch(
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")

    def fail_cpu_transfer(_image):
        raise AssertionError("deterministic torch stages should not use NumPy fallback")

    monkeypatch.setattr(capture_module, "_torch_to_numpy_image", fail_cpu_transfer)
    pipeline = CaptureArtifactPipeline.from_config(
        {
            "capture": {
                "stages": [
                    {
                        "type": "exposure",
                        "gain": 0.5,
                        "white_balance": [1.0, 1.0, 1.0],
                        "white_balance_jitter": 0.0,
                    },
                    {
                        "type": "isp",
                        "denoise_sigma": 0.0,
                        "color_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "tone_map": "none",
                        "gamma": "linear",
                        "local_contrast_strength": 0.0,
                        "sharpen_amount": 0.0,
                        "saturation": 1.0,
                    },
                    {
                        "type": "transport",
                        "resize": [8, 6],
                        "bit_depth": 4,
                        "jpeg": {"enabled": False},
                    },
                ]
            }
        }
    )
    image = torch.full((12, 16, 3), 0.8, dtype=torch.float32)

    result = pipeline.apply_torch(
        image,
        context=CaptureContext(rng=np.random.default_rng(7)),
    )

    assert torch.is_tensor(result)
    assert result.device == image.device
    assert result.dtype == image.dtype
    assert result.shape == (6, 8, 3)
    torch.testing.assert_close(result, torch.round(result * 15.0) / 15.0)


def test_capture_torch_matches_numpy_for_deterministic_post_sensor_stack() -> None:
    torch = pytest.importorskip("torch")
    config = {
        "capture": {
            "stages": [
                {
                    "type": "exposure",
                    "gain": 0.75,
                    "white_balance": [1.05, 0.95, 1.0],
                    "white_balance_jitter": 0.0,
                },
                {
                    "type": "isp",
                    "denoise_sigma": 0.0,
                    "color_matrix": [
                        [1.0, 0.02, 0.0],
                        [0.0, 0.98, 0.01],
                        [0.02, 0.0, 1.0],
                    ],
                    "tone_map": "reinhard",
                    "tone_map_strength": 0.2,
                    "gamma": "srgb",
                    "local_contrast_strength": 0.0,
                    "sharpen_amount": 0.0,
                    "saturation": 0.85,
                },
                {
                    "type": "transport",
                    "bit_depth": 0,
                    "jpeg": {"enabled": False},
                },
            ]
        }
    }
    pipeline = CaptureArtifactPipeline.from_config(config)
    x = np.linspace(0.05, 0.9, 9, dtype=np.float32)
    y = np.linspace(0.1, 0.8, 7, dtype=np.float32)
    image = np.dstack(
        [
            np.broadcast_to(x, (7, 9)),
            np.broadcast_to(y[:, None], (7, 9)),
            np.full((7, 9), 0.4, dtype=np.float32),
        ]
    )

    expected = pipeline.apply_np(
        image,
        context=CaptureContext(rng=np.random.default_rng(12)),
    )
    actual_t = pipeline.apply_torch(
        torch.from_numpy(image),
        context=CaptureContext(rng=np.random.default_rng(12)),
    )

    assert torch.is_tensor(actual_t)
    np.testing.assert_allclose(actual_t.detach().cpu().numpy(), expected, atol=1e-6)


def test_optics_torch_stage_uses_torch_for_supported_effects(monkeypatch) -> None:
    torch = pytest.importorskip("torch")

    def fail_cpu_transfer(_image):
        raise AssertionError("supported optics effects should not use NumPy fallback")

    monkeypatch.setattr(capture_module, "_torch_to_numpy_image", fail_cpu_transfer)
    pipeline = CaptureArtifactPipeline.from_config(
        {
            "capture": {
                "stages": [
                    {
                        "type": "optics",
                        "lens_distortion": 0.012,
                        "lens_distortion_k2": 0.002,
                        "chromatic_aberration_px": 0.35,
                        "depth_chromatic_fringing": {
                            "enabled": True,
                            "strength_px": 0.8,
                            "depth_weight": 0.4,
                            "fog_weight": 0.4,
                            "dark_weight": 0.2,
                            "gamma": 1.0,
                            "max_alpha": 0.6,
                            "blur_sigma": 0.0,
                        },
                        "blur_sigma": 0.25,
                        "motion_blur": {
                            "enabled": True,
                            "probability": 1.0,
                            "length_px": 3.0,
                            "angle_deg": 4.0,
                        },
                        "bloom": {
                            "enabled": True,
                            "threshold": 0.45,
                            "strength": 0.05,
                            "sigma": 1.2,
                        },
                        "veiling_glare_strength": 0.02,
                        "vignetting_strength": 0.18,
                        "vignetting_radius": 1.05,
                        "windshield_haze": {
                            "enabled": True,
                            "probability": 1.0,
                            "strength": 0.025,
                            "blur_sigma": 2.0,
                            "color": [0.82, 0.86, 0.88],
                        },
                        "droplets": {"enabled": False},
                    }
                ]
            }
        }
    )
    x = torch.linspace(0.05, 0.95, 48)
    y = torch.linspace(0.1, 0.85, 32)
    image = torch.stack(
        [
            x.view(1, -1).expand(32, 48),
            y.view(-1, 1).expand(32, 48),
            torch.full((32, 48), 0.65),
        ],
        dim=-1,
    )
    depth = torch.linspace(4.0, 80.0, 48).view(1, -1).expand(32, 48)
    k_map = torch.full((32, 48), 0.08)
    intrinsics = np.array(
        [[42.0, 0.0, 24.0], [0.0, 40.0, 15.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    result = pipeline.apply_torch(
        image,
        context=CaptureContext(
            rng=np.random.default_rng(31),
            depth_m=depth,
            k_map=k_map,
            intrinsics=intrinsics,
        ),
    )

    assert torch.is_tensor(result)
    assert result.device == image.device
    assert result.shape == image.shape
    assert 0.0 <= float(result.min()) <= float(result.max()) <= 1.0
    assert not torch.allclose(result, image)


def test_optics_torch_matches_numpy_for_intrinsics_vignetting() -> None:
    torch = pytest.importorskip("torch")
    pipeline = CaptureArtifactPipeline.from_config(
        {
            "capture": {
                "stages": [
                    {
                        "type": "optics",
                        "lens_distortion": 0.0,
                        "chromatic_aberration_px": 0.0,
                        "depth_chromatic_fringing": {"enabled": False},
                        "blur_sigma": 0.0,
                        "motion_blur": {"enabled": False},
                        "bloom": {"enabled": False},
                        "veiling_glare_strength": 0.0,
                        "vignetting_strength": 0.2,
                        "vignetting_radius": 1.1,
                        "windshield_haze": {"enabled": False},
                        "droplets": {"enabled": False},
                    }
                ]
            }
        }
    )
    image = np.full((12, 16, 3), 0.75, dtype=np.float32)
    intrinsics = np.array(
        [[14.0, 0.0, 8.0], [0.0, 13.0, 5.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    expected = pipeline.apply_np(
        image,
        context=CaptureContext(
            rng=np.random.default_rng(41),
            intrinsics=intrinsics,
        ),
    )
    actual_t = pipeline.apply_torch(
        torch.from_numpy(image),
        context=CaptureContext(
            rng=np.random.default_rng(41),
            intrinsics=intrinsics,
        ),
    )

    np.testing.assert_allclose(actual_t.detach().cpu().numpy(), expected, atol=1e-6)


def test_sensor_torch_matches_numpy_for_deterministic_bayer_path() -> None:
    torch = pytest.importorskip("torch")
    pipeline = CaptureArtifactPipeline.from_config(
        {
            "capture": {
                "stages": [
                    {
                        "type": "sensor",
                        "input_space": "linear",
                        "exposure_gain": 1.0,
                        "auto_exposure": {"enabled": False},
                        "white_balance": [1.0, 1.0, 1.0],
                        "white_balance_jitter": 0.0,
                        "channel_gain_sigma": 0.0,
                        "camera_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "clip": 1.0,
                        "bayer_pattern": "RGGB",
                        "shot_noise_electrons": 0.0,
                        "read_noise_sigma": 0.0,
                        "fixed_pattern_sigma": 0.0,
                        "row_noise_sigma": 0.0,
                        "column_noise_sigma": 0.0,
                        "black_level": [0.0, 0.0, 0.0],
                        "black_level_jitter": 0.0,
                        "white_level": [1.0, 1.0, 1.0],
                        "white_level_jitter": 0.0,
                        "adc_bit_depth": 0,
                        "post_demosaic_bit_depth": 0,
                        "hot_pixel_probability": 0.0,
                        "dead_pixel_probability": 0.0,
                        "demosaic": True,
                        "shadow_recovery_noise": {"enabled": False},
                    }
                ]
            }
        }
    )
    x = np.linspace(0.05, 0.95, 18, dtype=np.float32)
    y = np.linspace(0.1, 0.8, 14, dtype=np.float32)
    image = np.dstack(
        [
            np.broadcast_to(x, (14, 18)),
            np.broadcast_to(y[:, None], (14, 18)),
            np.full((14, 18), 0.45, dtype=np.float32),
        ]
    )

    expected = pipeline.apply_np(
        image,
        context=CaptureContext(rng=np.random.default_rng(51)),
    )
    actual_t = pipeline.apply_torch(
        torch.from_numpy(image),
        context=CaptureContext(rng=np.random.default_rng(51)),
    )

    np.testing.assert_allclose(actual_t.detach().cpu().numpy(), expected, atol=1e-6)


def test_sensor_torch_stage_uses_torch_for_noise_and_shadow_path(
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")

    def fail_cpu_transfer(_image):
        raise AssertionError("sensor torch path should not use NumPy fallback")

    monkeypatch.setattr(capture_module, "_torch_to_numpy_image", fail_cpu_transfer)
    pipeline = CaptureArtifactPipeline.from_config(
        {
            "capture": {
                "stages": [
                    {
                        "type": "sensor",
                        "input_space": "linear",
                        "exposure_gain": 1.0,
                        "auto_exposure": {
                            "enabled": True,
                            "target_luminance": 0.2,
                            "metering": "center_weighted",
                            "resolve_iso": True,
                            "max_iso": 800.0,
                        },
                        "white_balance": [1.0, 1.0, 1.0],
                        "white_balance_jitter": 0.0,
                        "channel_gain_sigma": 0.0,
                        "camera_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "bayer_pattern": "RGGB",
                        "iso": 400.0,
                        "base_iso": 100.0,
                        "full_well_electrons": [8000.0, 9000.0, 8200.0],
                        "read_noise_electrons": 2.5,
                        "fixed_pattern_sigma": 0.0005,
                        "row_noise_sigma": 0.0008,
                        "row_banding_correlation_px": 16.0,
                        "column_noise_sigma": 0.0005,
                        "column_banding_correlation_px": 16.0,
                        "black_level": [0.0, 0.0, 0.0],
                        "black_level_jitter": 0.0,
                        "white_level": [1.0, 1.0, 1.0],
                        "white_level_jitter": 0.0,
                        "adc_bit_depth": 12,
                        "post_demosaic_bit_depth": 12,
                        "hot_pixel_probability": 0.0005,
                        "dead_pixel_probability": 0.0005,
                        "demosaic": True,
                        "noise_modulation": {
                            "enabled": True,
                            "dark_gain": 0.6,
                            "depth_gain": 0.3,
                            "fog_gain": 0.4,
                            "max_gain": 2.0,
                            "smooth_sigma": 0.4,
                            "black_noise_floor": 0.3,
                            "black_suppression_luminance": 0.02,
                        },
                        "shadow_recovery_noise": {
                            "enabled": True,
                            "luminance_threshold": 0.35,
                            "luminance_softness": 0.2,
                            "gamma": 1.0,
                            "strength": 0.6,
                            "luma_sigma": 0.002,
                            "chroma_sigma": 0.006,
                            "chroma_mode": "balanced",
                            "red_chroma_gain": 0.8,
                            "blue_chroma_gain": 1.4,
                            "chroma_luminance_preservation": 1.0,
                            "smooth_sigma": 0.2,
                            "fog_weight": 0.2,
                            "depth_weight": 0.2,
                        },
                    }
                ]
            }
        }
    )
    image = torch.full((32, 40, 3), 0.32, dtype=torch.float32)
    image[:, :12] = 0.08
    depth = torch.linspace(3.0, 70.0, 40).view(1, -1).expand(32, 40)
    k_map = torch.full((32, 40), 0.06)

    result = pipeline.apply_torch(
        image,
        context=CaptureContext(
            rng=np.random.default_rng(61),
            depth_m=depth,
            k_map=k_map,
        ),
    )

    assert torch.is_tensor(result)
    assert result.device == image.device
    assert result.shape == image.shape
    assert 0.0 <= float(result.min()) <= float(result.max()) <= 1.0
    assert not torch.allclose(result, image)


def test_capture_torch_pipeline_runs_on_mps_without_image_fallback(
    monkeypatch,
) -> None:
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS backend is not available")

    def fail_cpu_transfer(_image):
        raise AssertionError("MPS capture path should not use image NumPy fallback")

    monkeypatch.setattr(capture_module, "_torch_to_numpy_image", fail_cpu_transfer)
    pipeline = CaptureArtifactPipeline.from_config(
        {
            "capture": {
                "stages": [
                    {
                        "type": "optics",
                        "lens_distortion": 0.002,
                        "chromatic_aberration_px": 0.1,
                        "depth_chromatic_fringing": {
                            "enabled": True,
                            "strength_px": 0.2,
                            "depth_weight": 0.5,
                            "fog_weight": 0.5,
                            "dark_weight": 0.0,
                            "blur_sigma": 0.0,
                        },
                        "blur_sigma": 0.1,
                        "motion_blur": {
                            "enabled": True,
                            "probability": 1.0,
                            "length_px": 2.0,
                            "angle_deg": 2.0,
                        },
                        "bloom": {
                            "enabled": True,
                            "threshold": 0.6,
                            "strength": 0.02,
                            "sigma": 0.5,
                        },
                        "veiling_glare_strength": 0.0,
                        "vignetting_strength": 0.05,
                        "windshield_haze": {"enabled": False},
                        "droplets": {"enabled": False},
                    },
                    {
                        "type": "sensor",
                        "input_space": "linear",
                        "exposure_gain": 1.0,
                        "auto_exposure": {
                            "enabled": True,
                            "resolve_iso": False,
                        },
                        "white_balance": [1.0, 1.0, 1.0],
                        "white_balance_jitter": 0.0,
                        "channel_gain_sigma": 0.0,
                        "camera_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "bayer_pattern": "RGGB",
                        "shot_noise_electrons": 0.0,
                        "read_noise_sigma": 0.0,
                        "fixed_pattern_sigma": 0.0,
                        "row_noise_sigma": 0.0,
                        "column_noise_sigma": 0.0,
                        "black_level": [0.0, 0.0, 0.0],
                        "black_level_jitter": 0.0,
                        "white_level": [1.0, 1.0, 1.0],
                        "white_level_jitter": 0.0,
                        "adc_bit_depth": 0,
                        "post_demosaic_bit_depth": 0,
                        "hot_pixel_probability": 0.0,
                        "dead_pixel_probability": 0.0,
                        "shadow_recovery_noise": {"enabled": False},
                    },
                    {
                        "type": "isp",
                        "denoise_sigma": 0.0,
                        "tone_map": "none",
                        "gamma": "linear",
                        "local_contrast_strength": 0.0,
                        "sharpen_amount": 0.0,
                        "saturation": 1.0,
                    },
                    {
                        "type": "transport",
                        "bit_depth": 8,
                        "jpeg": {"enabled": False},
                    },
                ]
            }
        }
    )
    device = torch.device("mps")
    image = torch.rand((16, 20, 3), device=device, dtype=torch.float32)
    depth = torch.full((16, 20), 20.0, device=device, dtype=torch.float32)
    k_map = torch.full((16, 20), 0.04, device=device, dtype=torch.float32)

    result = pipeline.apply_torch(
        image,
        context=CaptureContext(
            rng=np.random.default_rng(71),
            depth_m=depth,
            k_map=k_map,
        ),
    )

    result_cpu = result.detach().cpu()
    assert torch.is_tensor(result)
    assert result.device.type == "mps"
    assert result.shape == image.shape
    assert 0.0 <= float(result_cpu.min()) <= float(result_cpu.max()) <= 1.0


def test_capture_stage_condition_profiles_override_stage_config() -> None:
    pipeline = CaptureArtifactPipeline.from_config(
        {
            "capture": {
                "stages": [
                    {
                        "type": "exposure",
                        "gain": 1.0,
                        "condition_profile": "selected",
                        "condition_profiles": [
                            {"name": "selected", "weight": 0.0, "gain": 0.5},
                            {"name": "unused", "weight": 1.0, "gain": 2.0},
                        ],
                    }
                ]
            }
        }
    )
    image = np.full((2, 3, 3), 0.8, dtype=np.float32)

    result = pipeline.apply_np(
        image,
        context=CaptureContext(rng=np.random.default_rng(7)),
    )

    np.testing.assert_allclose(result, 0.4)


def test_capture_overrides_can_force_named_condition_profile() -> None:
    pipeline = CaptureArtifactPipeline.from_config(
        {
            "capture": {
                "stages": [
                    {
                        "type": "exposure",
                        "gain": 1.0,
                        "condition_profiles": [
                            {"name": "clean", "weight": 1.0, "gain": 1.0},
                            {"name": "dark", "weight": 0.0, "gain": 0.25},
                        ],
                    }
                ]
            },
            "capture_overrides": {
                "exposure": {
                    "condition_profile": "dark",
                }
            },
        }
    )
    image = np.full((2, 3, 3), 0.8, dtype=np.float32)

    result = pipeline.apply_np(
        image,
        context=CaptureContext(rng=np.random.default_rng(7)),
    )

    np.testing.assert_allclose(result, 0.2)


def test_scenario_profile_correlates_model_and_capture_overrides(
    tmp_path: Path,
) -> None:
    cfg = {
        "airlight": "from_sky",
        "device": "cpu",
        "seed": 7,
        "contrast_threshold": 0.05,
        "capture": {
            "stages": [
                {
                    "type": "exposure",
                    "gain": 1.0,
                    "condition_profiles": [
                        {"name": "clean", "weight": 1.0, "gain": 1.0},
                        {"name": "dark", "weight": 0.0, "gain": 0.25},
                    ],
                }
            ]
        },
        "scenario_profiles": [
            {
                "name": "underexposed_dense",
                "weight": 1.0,
                "model": "uniform",
                "models": {
                    "uniform": {
                        "visibility_m": {"dist": "constant", "value": 20.0},
                        "atmospheric_light": [0.2, 0.2, 0.2],
                    }
                },
                "capture_overrides": {
                    "exposure": {
                        "condition_profile": "dark",
                    }
                },
            }
        ],
    }
    config_path = tmp_path / "scenario_config.json"
    config_path.write_text(json.dumps(cfg))
    transform = FogTransform(
        config_path=str(config_path),
        out_path=str(tmp_path / "out"),
    )

    plan = transform._resolve_render_plan(np.random.default_rng(3))
    assert plan.scenario_name == "underexposed_dense"
    assert plan.model_name == "uniform"
    assert plan.model_cfg["visibility_m"]["value"] == 20.0

    rgb = np.full((4, 5, 3), 0.8, dtype=np.float32)
    depth = np.zeros((4, 5), dtype=np.float32)
    sky_mask = np.ones((4, 5), dtype=bool)
    result = transform.pipeline.process_np(
        rgb=rgb,
        depth_m=depth,
        sky_mask=sky_mask,
        model_name=plan.model_name,
        model_cfg=plan.model_cfg,
        rng=np.random.default_rng(3),
        sample_id="sample",
        capture_artifacts=plan.capture_artifacts,
    )

    np.testing.assert_allclose(result.rgb, 0.2)


def test_capture_camera_profile_supplies_stage_defaults() -> None:
    pipeline = CaptureArtifactPipeline.from_config(
        {
            "camera_profiles": {
                "unit_test": {
                    "sensor": {
                        "iso": 800.0,
                        "bayer_pattern": "RGGB",
                        "read_noise_electrons": 3.0,
                    }
                }
            },
            "camera_profile": "unit_test",
            "capture": {"stages": [{"type": "sensor", "iso": 400.0}]},
        }
    )

    assert len(pipeline.stages) == 1
    assert pipeline.stages[0].config["iso"] == 400.0
    assert pipeline.stages[0].config["bayer_pattern"] == "RGGB"
    assert pipeline.stages[0].config["read_noise_electrons"] == 3.0


def test_optics_stage_uses_intrinsics_principal_point() -> None:
    pipeline = CaptureArtifactPipeline.from_config(
        {
            "capture": {
                "stages": [
                    {
                        "type": "optics",
                        "lens_distortion": 0.25,
                        "blur_sigma": 0.0,
                        "chromatic_aberration_px": 0.0,
                        "motion_blur": {"enabled": False},
                        "bloom": {"enabled": False},
                        "vignetting_strength": 0.0,
                        "veiling_glare_strength": 0.0,
                        "windshield_haze": {"enabled": False},
                        "droplets": {"enabled": False},
                    }
                ]
            }
        }
    )
    x = np.linspace(0.0, 1.0, 18, dtype=np.float32)
    y = np.linspace(0.0, 1.0, 14, dtype=np.float32)
    image = np.dstack(
        [
            np.broadcast_to(x, (14, 18)),
            np.broadcast_to(y[:, None], (14, 18)),
            np.full((14, 18), 0.4, dtype=np.float32),
        ]
    )
    centered_k = np.array(
        [[12.0, 0.0, 8.5], [0.0, 12.0, 6.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    shifted_k = np.array(
        [[12.0, 0.0, 3.0], [0.0, 12.0, 10.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    centered = pipeline.apply_np(
        image,
        CaptureContext(rng=np.random.default_rng(11), intrinsics=centered_k),
    )
    shifted = pipeline.apply_np(
        image,
        CaptureContext(rng=np.random.default_rng(11), intrinsics=shifted_k),
    )

    assert not np.allclose(centered, shifted)


def test_sensor_stage_supports_raw_and_post_demosaic_quantization() -> None:
    pipeline = CaptureArtifactPipeline.from_config(
        {
            "capture": {
                "stages": [
                    {
                        "type": "sensor",
                        "input_space": "linear",
                        "exposure_gain": 1.0,
                        "white_balance": [1.0, 1.0, 1.0],
                        "white_balance_jitter": 0.0,
                        "channel_gain_sigma": 0.0,
                        "camera_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "bayer_pattern": "RGGB",
                        "shot_noise_electrons": 0.0,
                        "read_noise_electrons": 0.0,
                        "fixed_pattern_sigma": 0.0,
                        "row_noise_sigma": 0.0,
                        "column_noise_sigma": 0.0,
                        "black_level": [0.0, 0.0, 0.0],
                        "black_level_jitter": 0.0,
                        "white_level": [1.0, 1.0, 1.0],
                        "white_level_jitter": 0.0,
                        "adc_bit_depth": 4,
                        "post_demosaic_bit_depth": 4,
                        "hot_pixel_probability": 0.0,
                        "dead_pixel_probability": 0.0,
                    }
                ]
            }
        }
    )
    image = np.full((8, 10, 3), 0.41, dtype=np.float32)

    result = pipeline.apply_np(
        image,
        CaptureContext(rng=np.random.default_rng(5)),
    )

    assert result.shape == image.shape
    np.testing.assert_allclose(result, np.round(result * 15.0) / 15.0)


def _deterministic_sensor_pipeline(
    *,
    auto_exposure: dict,
    exposure_gain: float = 1.0,
    sensor_overrides: dict | None = None,
) -> CaptureArtifactPipeline:
    sensor = {
        "type": "sensor",
        "input_space": "linear",
        "exposure_gain": exposure_gain,
        "auto_exposure": auto_exposure,
        "white_balance": [1.0, 1.0, 1.0],
        "white_balance_jitter": 0.0,
        "channel_gain_sigma": 0.0,
        "camera_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "bayer_pattern": "RGGB",
        "shot_noise_electrons": 0.0,
        "read_noise_electrons": 0.0,
        "fixed_pattern_sigma": 0.0,
        "row_noise_sigma": 0.0,
        "column_noise_sigma": 0.0,
        "black_level": [0.0, 0.0, 0.0],
        "black_level_jitter": 0.0,
        "white_level": [1.0, 1.0, 1.0],
        "white_level_jitter": 0.0,
        "adc_bit_depth": 0,
        "post_demosaic_bit_depth": 0,
        "hot_pixel_probability": 0.0,
        "dead_pixel_probability": 0.0,
    }
    if sensor_overrides:
        sensor.update(sensor_overrides)
    return CaptureArtifactPipeline.from_config(
        {
            "capture": {
                "stages": [sensor]
            }
        }
    )


def test_sensor_auto_exposure_meters_rendered_luminance() -> None:
    pipeline = _deterministic_sensor_pipeline(
        auto_exposure={
            "enabled": True,
            "metering": "mean",
            "target_luminance": 0.2,
            "highlight_protection": 0.0,
            "min_gain": 0.1,
            "max_gain": 8.0,
        }
    )

    dark = pipeline.apply_np(
        np.full((12, 16, 3), 0.08, dtype=np.float32),
        CaptureContext(rng=np.random.default_rng(23)),
    )
    bright = pipeline.apply_np(
        np.full((12, 16, 3), 0.4, dtype=np.float32),
        CaptureContext(rng=np.random.default_rng(23)),
    )

    np.testing.assert_allclose(float(dark.mean()), 0.2, atol=0.015)
    np.testing.assert_allclose(float(bright.mean()), 0.2, atol=0.015)


def test_sensor_auto_exposure_keeps_exposure_gain_as_compensation() -> None:
    pipeline = _deterministic_sensor_pipeline(
        exposure_gain=0.5,
        auto_exposure={
            "enabled": True,
            "metering": "mean",
            "target_luminance": 0.2,
            "highlight_protection": 0.0,
            "manual_gain_weight": 1.0,
            "min_gain": 0.1,
            "max_gain": 8.0,
        },
    )

    result = pipeline.apply_np(
        np.full((12, 16, 3), 0.1, dtype=np.float32),
        CaptureContext(rng=np.random.default_rng(29)),
    )

    np.testing.assert_allclose(float(result.mean()), 0.1, atol=0.015)


def test_sensor_auto_exposure_can_protect_highlights() -> None:
    pipeline = _deterministic_sensor_pipeline(
        auto_exposure={
            "enabled": True,
            "metering": "mean",
            "target_luminance": 0.4,
            "highlight_percentile": 80.0,
            "highlight_target": 0.72,
            "highlight_protection": 1.0,
            "min_gain": 0.1,
            "max_gain": 8.0,
        }
    )
    image = np.full((16, 16, 3), 0.05, dtype=np.float32)
    image[:, :6] = 0.9

    result = pipeline.apply_np(
        image,
        CaptureContext(rng=np.random.default_rng(31)),
    )

    assert float(result.max()) <= 0.74
    assert float(result.mean()) < 0.32


def test_shadow_recovery_noise_is_local_to_dark_regions() -> None:
    pipeline = _deterministic_sensor_pipeline(
        auto_exposure={"enabled": False},
        sensor_overrides={
            "shadow_recovery_noise": {
                "enabled": True,
                "luminance_threshold": 0.18,
                "luminance_softness": 0.06,
                "gamma": 1.2,
                "luma_sigma": 0.02,
                "chroma_sigma": 0.025,
                "blotch_sigma": 0.0,
                "smooth_sigma": 0.0,
            },
        },
    )
    image = np.full((48, 64, 3), 0.72, dtype=np.float32)
    image[:, 32:] = 0.06

    result = pipeline.apply_np(
        image,
        CaptureContext(rng=np.random.default_rng(37)),
    )

    bright_std = float(result[:, :24].std())
    dark_std = float(result[:, 40:].std())
    assert dark_std > bright_std + 0.012


def test_shadow_recovery_chroma_noise_preserves_luminance() -> None:
    pipeline = _deterministic_sensor_pipeline(
        auto_exposure={"enabled": False},
        sensor_overrides={
            "shadow_recovery_noise": {
                "enabled": True,
                "luminance_threshold": 0.5,
                "luminance_softness": 0.3,
                "gamma": 1.0,
                "luma_sigma": 0.0,
                "chroma_sigma": 0.03,
                "chroma_mode": "balanced",
                "chroma_luminance_preservation": 1.0,
                "blotch_sigma": 0.0,
                "smooth_sigma": 0.0,
            },
        },
    )
    image = np.full((96, 128, 3), 0.25, dtype=np.float32)

    result = pipeline.apply_np(
        image,
        CaptureContext(rng=np.random.default_rng(41)),
    )

    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luminance_delta = np.sum((result - image) * weights.reshape(1, 1, 3), axis=-1)
    channel_delta = result - image
    red_std = float(channel_delta[..., 0].std())
    blue_std = float(channel_delta[..., 2].std())

    assert float(channel_delta.std()) > 0.006
    assert float(luminance_delta.std()) < 1e-6
    assert abs(red_std - blue_std) / max(red_std, blue_std) < 0.08


def test_shadow_recovery_noise_is_suppressed_in_pitch_black_regions() -> None:
    pipeline = _deterministic_sensor_pipeline(
        auto_exposure={"enabled": False},
        sensor_overrides={
            "shadow_recovery_noise": {
                "enabled": True,
                "luminance_threshold": 0.2,
                "luminance_softness": 0.08,
                "gamma": 1.0,
                "luma_sigma": 0.0,
                "chroma_sigma": 0.035,
                "chroma_mode": "balanced",
                "chroma_luminance_preservation": 1.0,
                "blotch_sigma": 0.0,
                "smooth_sigma": 0.0,
                "black_noise_floor": 0.1,
                "black_suppression_luminance": 0.03,
                "black_suppression_softness": 0.07,
            },
        },
    )
    image = np.full((64, 96, 3), 0.4, dtype=np.float32)
    image[:, :32] = 0.0
    image[:, 32:64] = 0.08

    result = pipeline.apply_np(
        image,
        CaptureContext(rng=np.random.default_rng(43)),
    )

    black_std = float((result[:, 8:24] - image[:, 8:24]).std())
    dim_std = float((result[:, 40:56] - image[:, 40:56]).std())
    bright_std = float((result[:, 72:88] - image[:, 72:88]).std())

    assert dim_std > black_std * 2.5
    assert dim_std > bright_std * 2.5


def test_shadow_recovery_chroma_noise_can_bias_blue_channel() -> None:
    pipeline = _deterministic_sensor_pipeline(
        auto_exposure={"enabled": False},
        sensor_overrides={
            "shadow_recovery_noise": {
                "enabled": True,
                "luminance_threshold": 0.5,
                "luminance_softness": 0.3,
                "gamma": 1.0,
                "luma_sigma": 0.0,
                "chroma_sigma": 0.02,
                "chroma_mode": "balanced",
                "red_chroma_gain": 0.65,
                "blue_chroma_gain": 2.8,
                "chroma_axis_correlation": 0.15,
                "chroma_luminance_preservation": 1.0,
                "blotch_sigma": 0.0,
                "smooth_sigma": 0.0,
            },
        },
    )
    image = np.full((96, 128, 3), 0.25, dtype=np.float32)

    result = pipeline.apply_np(
        image,
        CaptureContext(rng=np.random.default_rng(53)),
    )

    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    delta = result - image
    luminance_delta = np.sum(delta * weights.reshape(1, 1, 3), axis=-1)
    red_std = float(delta[..., 0].std())
    blue_std = float(delta[..., 2].std())

    assert blue_std > red_std * 2.5
    assert float(luminance_delta.std()) < 1e-6


def test_sensor_noise_modulation_is_stronger_in_dark_foggy_regions() -> None:
    pipeline = CaptureArtifactPipeline.from_config(
        {
            "capture": {
                "stages": [
                    {
                        "type": "sensor",
                        "input_space": "linear",
                        "exposure_gain": 1.0,
                        "white_balance": [1.0, 1.0, 1.0],
                        "white_balance_jitter": 0.0,
                        "channel_gain_sigma": 0.0,
                        "camera_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "bayer_pattern": "RGGB",
                        "shot_noise_electrons": 0.0,
                        "read_noise_sigma": 0.01,
                        "fixed_pattern_sigma": 0.0,
                        "row_noise_sigma": 0.0,
                        "column_noise_sigma": 0.0,
                        "black_level": [0.0, 0.0, 0.0],
                        "black_level_jitter": 0.0,
                        "white_level": [1.0, 1.0, 1.0],
                        "white_level_jitter": 0.0,
                        "adc_bit_depth": 0,
                        "demosaic": False,
                        "noise_modulation": {
                            "enabled": True,
                            "dark_gain": 1.0,
                            "depth_gain": 0.6,
                            "fog_gain": 1.0,
                            "max_gain": 3.0,
                        },
                    }
                ]
            }
        }
    )
    image = np.full((48, 64, 3), 0.75, dtype=np.float32)
    image[:, 32:] = 0.2
    depth = np.full((48, 64), 5.0, dtype=np.float32)
    depth[:, 32:] = 80.0
    k_map = np.full((48, 64), 0.08, dtype=np.float32)

    result = pipeline.apply_np(
        image,
        CaptureContext(
            rng=np.random.default_rng(17),
            depth_m=depth,
            k_map=k_map,
        ),
    )

    left_std = float(result[:, :32, 0].std())
    right_std = float(result[:, 32:, 0].std())
    assert right_std > left_std * 1.35


def test_sensor_noise_modulation_can_drop_in_pitch_black_regions() -> None:
    pipeline = CaptureArtifactPipeline.from_config(
        {
            "capture": {
                "stages": [
                    {
                        "type": "sensor",
                        "input_space": "linear",
                        "exposure_gain": 1.0,
                        "white_balance": [1.0, 1.0, 1.0],
                        "white_balance_jitter": 0.0,
                        "channel_gain_sigma": 0.0,
                        "camera_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "bayer_pattern": "RGGB",
                        "shot_noise_electrons": 0.0,
                        "read_noise_sigma": 0.012,
                        "fixed_pattern_sigma": 0.0,
                        "row_noise_sigma": 0.0,
                        "column_noise_sigma": 0.0,
                        "black_level": [0.0, 0.0, 0.0],
                        "black_level_jitter": 0.0,
                        "white_level": [1.0, 1.0, 1.0],
                        "white_level_jitter": 0.0,
                        "adc_bit_depth": 0,
                        "demosaic": False,
                        "noise_modulation": {
                            "enabled": True,
                            "dark_gain": 2.0,
                            "gamma": 1.0,
                            "max_gain": 3.0,
                            "black_noise_floor": 0.0,
                            "black_suppression_luminance": 0.02,
                            "black_suppression_softness": 0.06,
                        },
                    }
                ]
            }
        }
    )
    image = np.full((64, 96, 3), 0.4, dtype=np.float32)
    image[:, :32] = 0.0
    image[:, 32:64] = 0.08

    result = pipeline.apply_np(
        image,
        CaptureContext(rng=np.random.default_rng(47)),
    )

    black_std = float(result[:, 8:24, 0].std())
    dim_std = float(result[:, 40:56, 0].std())
    assert dim_std > black_std * 1.35


def test_sensor_row_banding_is_smoothed_by_correlation_length() -> None:
    pipeline = CaptureArtifactPipeline.from_config(
        {
            "capture": {
                "stages": [
                    {
                        "type": "sensor",
                        "input_space": "linear",
                        "exposure_gain": 1.0,
                        "white_balance": [1.0, 1.0, 1.0],
                        "white_balance_jitter": 0.0,
                        "channel_gain_sigma": 0.0,
                        "camera_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "bayer_pattern": "RGGB",
                        "shot_noise_electrons": 0.0,
                        "read_noise_sigma": 0.0,
                        "fixed_pattern_sigma": 0.0,
                        "row_noise_sigma": 0.01,
                        "row_banding_correlation_px": 48.0,
                        "column_noise_sigma": 0.0,
                        "black_level": [0.0, 0.0, 0.0],
                        "black_level_jitter": 0.0,
                        "white_level": [1.0, 1.0, 1.0],
                        "white_level_jitter": 0.0,
                        "adc_bit_depth": 0,
                        "demosaic": False,
                    }
                ]
            }
        }
    )
    image = np.full((96, 64, 3), 0.5, dtype=np.float32)

    result = pipeline.apply_np(
        image,
        CaptureContext(rng=np.random.default_rng(19)),
    )

    row_mean = result[..., 0].mean(axis=1)
    assert float(np.diff(row_mean).std()) < float(row_mean.std()) * 0.45


def test_depth_chromatic_fringing_is_weighted_by_depth_and_fog() -> None:
    pipeline = CaptureArtifactPipeline.from_config(
        {
            "capture": {
                "stages": [
                    {
                        "type": "optics",
                        "lens_distortion": 0.0,
                        "chromatic_aberration_px": 0.0,
                        "blur_sigma": 0.0,
                        "motion_blur": {"enabled": False},
                        "bloom": {"enabled": False},
                        "vignetting_strength": 0.0,
                        "veiling_glare_strength": 0.0,
                        "windshield_haze": {"enabled": False},
                        "droplets": {"enabled": False},
                        "depth_chromatic_fringing": {
                            "enabled": True,
                            "strength_px": 2.0,
                            "depth_weight": 0.5,
                            "fog_weight": 0.5,
                            "dark_weight": 0.0,
                            "gamma": 1.0,
                            "max_alpha": 1.0,
                            "blur_sigma": 0.0,
                        },
                    }
                ]
            }
        }
    )
    x = np.linspace(0.0, 1.0, 80, dtype=np.float32)
    image = np.dstack(
        [
            np.broadcast_to(x, (40, 80)),
            np.full((40, 80), 0.5, dtype=np.float32),
            np.broadcast_to(1.0 - x, (40, 80)),
        ]
    )
    depth = np.full((40, 80), 5.0, dtype=np.float32)
    depth[:, 40:] = 90.0
    k_map = np.full((40, 80), 0.08, dtype=np.float32)

    result = pipeline.apply_np(
        image,
        CaptureContext(
            rng=np.random.default_rng(23),
            depth_m=depth,
            k_map=k_map,
        ),
    )
    delta = np.abs(result - image).mean(axis=-1)

    assert float(delta[:, 40:].mean()) > float(delta[:, :40].mean()) * 2.0


def test_apply_model_returns_full_size_maps_for_uniform() -> None:
    """Sanity-check the broadcast logic on the model layer."""
    from euler_preprocess.fog.models import apply_model

    rng = np.random.default_rng(0)
    rgb = np.full((10, 12, 3), 0.5, dtype=np.float32)
    depth = np.full((10, 12), 30.0, dtype=np.float32)
    estimated = np.array([0.9, 0.85, 0.8], dtype=np.float32)
    cfg = {
        "visibility_m": {"dist": "constant", "value": 100.0},
        "atmospheric_light": "from_sky",
    }

    foggy, k_mean, ls_base, k_map, ls_map = apply_model(
        rgb,
        depth,
        "uniform",
        cfg,
        rng,
        contrast_threshold_default=0.05,
        estimated_airlight=estimated,
    )

    assert foggy.shape == (10, 12, 3)
    assert k_map.shape == (10, 12)
    assert ls_map.shape == (10, 12, 3)
    np.testing.assert_allclose(k_map, k_mean)
    np.testing.assert_allclose(ls_map, np.broadcast_to(ls_base, ls_map.shape))


def test_estimated_airlight_is_dampened_by_scattering_strength() -> None:
    """Dense fog should not push DCP/sky-estimated airlight toward overbright white."""
    from euler_preprocess.fog.models import apply_model, visibility_to_k

    rgb = np.full((6, 8, 3), 0.35, dtype=np.float32)
    depth = np.full((6, 8), 30.0, dtype=np.float32)
    estimated = np.array([0.9, 0.9, 0.9], dtype=np.float32)

    light_cfg = {
        "visibility_m": {"dist": "constant", "value": 300.0},
        "atmospheric_light": "from_sky",
    }
    dense_cfg = {
        "visibility_m": {"dist": "constant", "value": 20.0},
        "atmospheric_light": "from_sky",
    }

    _, light_beta, light_airlight, _, _ = apply_model(
        rgb,
        depth,
        "uniform",
        light_cfg,
        np.random.default_rng(0),
        contrast_threshold_default=0.05,
        estimated_airlight=estimated,
    )
    _, dense_beta, dense_airlight, _, _ = apply_model(
        rgb,
        depth,
        "uniform",
        dense_cfg,
        np.random.default_rng(0),
        contrast_threshold_default=0.05,
        estimated_airlight=estimated,
    )

    ref_beta = visibility_to_k(80.0, 0.05)
    expected_dense_factor = 0.45 + 0.55 / (1.0 + dense_beta / ref_beta)

    assert dense_beta > light_beta
    assert float(dense_airlight.mean()) < float(light_airlight.mean())
    np.testing.assert_allclose(
        dense_airlight,
        estimated * expected_dense_factor,
        rtol=1e-6,
    )


def test_literal_atmospheric_light_is_not_dampened_by_default() -> None:
    """Hand-authored L_s values remain exact unless the config opts into all values."""
    from euler_preprocess.fog.models import apply_model

    rng = np.random.default_rng(0)
    rgb = np.full((4, 5, 3), 0.35, dtype=np.float32)
    depth = np.full((4, 5), 30.0, dtype=np.float32)
    estimated = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    literal = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    cfg = {
        "visibility_m": {"dist": "constant", "value": 20.0},
        "atmospheric_light": literal.tolist(),
    }

    _, _, airlight, _, ls_map = apply_model(
        rgb,
        depth,
        "uniform",
        cfg,
        rng,
        contrast_threshold_default=0.05,
        estimated_airlight=estimated,
    )

    np.testing.assert_allclose(airlight, literal, atol=1e-6)
    np.testing.assert_allclose(ls_map, np.broadcast_to(literal, ls_map.shape))


def test_airlight_dampening_alias_can_disable_default() -> None:
    from euler_preprocess.fog.models import apply_model

    rgb = np.full((4, 5, 3), 0.35, dtype=np.float32)
    depth = np.full((4, 5), 30.0, dtype=np.float32)
    estimated = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    cfg = {
        "visibility_m": {"dist": "constant", "value": 20.0},
        "atmospheric_light": "from_sky",
        "airlight_damping": {"enabled": False},
    }

    _, _, airlight, _, _ = apply_model(
        rgb,
        depth,
        "uniform",
        cfg,
        np.random.default_rng(0),
        contrast_threshold_default=0.05,
        estimated_airlight=estimated,
    )

    np.testing.assert_allclose(airlight, estimated, atol=1e-6)


def test_heterogeneous_ls_uses_dampened_airlight_base() -> None:
    """Perlin L_s modes should modulate around the already dampened base airlight."""
    from euler_preprocess.fog.models import apply_model

    rng = np.random.default_rng(123)
    rgb = np.full((16, 16, 3), 0.35, dtype=np.float32)
    depth = np.full((16, 16), 40.0, dtype=np.float32)
    estimated = np.array([0.9, 0.9, 0.9], dtype=np.float32)
    cfg = {
        "scattering_coefficient": {"dist": "constant", "value": 0.15},
        "atmospheric_light": "from_sky",
        "ls_hetero": {
            "scales": [4],
            "min_factor": 1.0,
            "max_factor": 1.0,
            "normalize_to_mean": False,
        },
    }

    _, _, airlight, _, ls_map = apply_model(
        rgb,
        depth,
        "heterogeneous_ls",
        cfg,
        rng,
        contrast_threshold_default=0.05,
        estimated_airlight=estimated,
    )

    assert float(airlight.mean()) < float(estimated.mean())
    np.testing.assert_allclose(ls_map, np.broadcast_to(airlight, ls_map.shape))


def test_heterogeneous_ls_gradient_brightens_top_airlight() -> None:
    """Optional L_s gradients should act on the atmospheric-light field."""
    from euler_preprocess.fog.models import apply_model

    rng = np.random.default_rng(123)
    rgb = np.zeros((20, 16, 3), dtype=np.float32)
    depth = np.full((20, 16), 80.0, dtype=np.float32)
    estimated = np.array([0.6, 0.65, 0.7], dtype=np.float32)
    cfg = {
        "scattering_coefficient": {"dist": "constant", "value": 0.08},
        "atmospheric_light": "from_sky",
        "airlight_dampening": {"enabled": False},
        "ls_hetero": {
            "scales": [4],
            "min_factor": 1.0,
            "max_factor": 1.0,
            "normalize_to_mean": False,
            "ls_gradient": {
                "enabled": True,
                "top_factor": 1.18,
                "bottom_factor": 0.82,
                "gamma": 1.0,
                "normalize_to_mean": False,
                "fog_opacity_weight": 0.0,
            },
        },
    }

    _, _, _, _, ls_map = apply_model(
        rgb,
        depth,
        "heterogeneous_ls",
        cfg,
        rng,
        contrast_threshold_default=0.05,
        estimated_airlight=estimated,
    )

    assert float(ls_map[:4].mean()) > float(ls_map[-4:].mean()) * 1.3


def test_apply_model_returns_spatial_fields_for_heterogeneous() -> None:
    """Heterogeneous models should return the actual non-constant maps used."""
    from euler_preprocess.fog.models import apply_model

    rng = np.random.default_rng(123)
    rgb = np.full((16, 16, 3), 0.5, dtype=np.float32)
    depth = np.full((16, 16), 50.0, dtype=np.float32)
    estimated = np.array([0.8, 0.8, 0.9], dtype=np.float32)
    cfg = {
        "visibility_m": {"dist": "constant", "value": 80.0},
        "atmospheric_light": "from_sky",
        "k_hetero": {
            "scales": [4],
            "min_factor": 0.0,
            "max_factor": 1.0,
            "normalize_to_mean": False,
        },
    }

    _, k_mean, _, k_map, _ = apply_model(
        rgb,
        depth,
        "heterogeneous_k",
        cfg,
        rng,
        contrast_threshold_default=0.05,
        estimated_airlight=estimated,
    )

    assert k_map.shape == (16, 16)
    # Spatially varying — there should be actual variance in the field.
    assert float(k_map.std()) > 0.0


def test_smooth_auto_scales_are_image_relative_low_frequency() -> None:
    """smooth_auto should avoid the pixel-scale octaves that make fog speckly."""
    from euler_preprocess.fog.models import resolve_scales

    rng = np.random.default_rng(0)
    cfg = {
        "scales": "smooth_auto",
        "correlation_length_fraction": 0.25,
        "octaves": 4,
        "max_scale_fraction": 1.0,
    }

    assert resolve_scales(cfg, height=100, width=200, rng=rng) == [25, 50, 100, 200]


def test_smooth_noise_contrast_keeps_heterogeneous_beta_near_mean() -> None:
    """Low noise contrast keeps spatial fog gradients subtle around the base beta."""
    from euler_preprocess.fog.models import apply_model

    rng = np.random.default_rng(123)
    rgb = np.full((80, 120, 3), 0.5, dtype=np.float32)
    depth = np.full((80, 120), 50.0, dtype=np.float32)
    estimated = np.array([0.8, 0.8, 0.9], dtype=np.float32)
    cfg = {
        "visibility_m": {"dist": "constant", "value": 80.0},
        "atmospheric_light": "from_sky",
        "k_hetero": {
            "scales": "smooth_auto",
            "correlation_length_fraction": 0.25,
            "octaves": 3,
            "min_factor": 0.5,
            "max_factor": 1.5,
            "contrast": 0.2,
            "normalize_to_mean": True,
        },
    }

    _, k_mean, _, k_map, _ = apply_model(
        rgb,
        depth,
        "heterogeneous_k",
        cfg,
        rng,
        contrast_threshold_default=0.05,
        estimated_airlight=estimated,
    )

    factors = k_map / k_mean
    assert float(factors.std()) > 0.0
    assert float(factors.min()) >= 0.75
    assert float(factors.max()) <= 1.25
    np.testing.assert_allclose(float(factors.mean()), 1.0, rtol=1e-6)


def test_heterogeneous_k_normal_mor_sample_anchors_mean_beta() -> None:
    """A sampled MOR value is converted once, then used as the β-field mean."""
    from euler_preprocess.common.sampling import sample_value
    from euler_preprocess.fog.models import apply_model, visibility_to_k

    seed = 987
    visibility_spec = {
        "dist": "normal",
        "mean": 180.0,
        "std": 30.0,
        "min": 80.0,
    }
    expected_visibility = sample_value(visibility_spec, np.random.default_rng(seed))
    expected_beta = visibility_to_k(expected_visibility, 0.05)

    rgb = np.full((64, 64, 3), 0.5, dtype=np.float32)
    depth = np.full((64, 64), 50.0, dtype=np.float32)
    estimated = np.array([0.8, 0.8, 0.9], dtype=np.float32)
    cfg = {
        "visibility_m": visibility_spec,
        "atmospheric_light": "from_sky",
        "k_hetero": {
            "scales": "smooth_auto",
            "correlation_length_fraction": 0.25,
            "octaves": 3,
            "min_factor": 0.65,
            "max_factor": 1.45,
            "contrast": 0.65,
            "normalize_to_mean": True,
        },
    }

    _, k_mean, _, k_map, _ = apply_model(
        rgb,
        depth,
        "heterogeneous_k",
        cfg,
        np.random.default_rng(seed),
        contrast_threshold_default=0.05,
        estimated_airlight=estimated,
    )

    assert float(k_map.std()) > 0.0
    np.testing.assert_allclose(k_mean, expected_beta, rtol=1e-7)
    np.testing.assert_allclose(float(k_map.mean()), expected_beta, rtol=1e-6)


def test_apply_model_accepts_direct_scattering_coefficient() -> None:
    """Stepped configs may specify beta directly instead of MOR/visibility."""
    from euler_preprocess.fog.models import apply_model

    rng = np.random.default_rng(0)
    rgb = np.full((4, 5, 3), 0.5, dtype=np.float32)
    depth = np.full((4, 5), 10.0, dtype=np.float32)
    estimated = np.array([0.8, 0.8, 0.9], dtype=np.float32)
    cfg = {
        "scattering_coefficient": {"dist": "constant", "value": 0.123},
        "visibility_m": {"dist": "constant", "value": 999.0},
        "atmospheric_light": "from_sky",
    }

    _, k_mean, _, k_map, _ = apply_model(
        rgb,
        depth,
        "uniform",
        cfg,
        rng,
        contrast_threshold_default=0.05,
        estimated_airlight=estimated,
    )

    assert k_mean == 0.123
    np.testing.assert_allclose(k_map, 0.123)
