"""Real Drive Sim scenario-profile smoke and heuristic tests.

By default these tests use the reusable archive in ``tests/fixtures``. Override
that fixture with one of:

- ``EULER_PREPROCESS_RDS_SAMPLE_ROOT=/path/to/extracted/rds-samples``
- ``EULER_PREPROCESS_RDS_SAMPLE_ZIP=/path/to/rds-samples.zip``
- ``EULER_PREPROCESS_RDS_DOWNLOAD=1`` to download the default temp.sh archive
"""

from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from euler_loading.loaders.cpu import real_drive_sim
from euler_preprocess.common.io import load_json
from euler_preprocess.fog.transform import FogTransform


RDS_SAMPLE_URL = "https://temp.sh/fmuVe/rds-samples.zip"
DEFAULT_RDS_SAMPLE_ZIP = (
    Path(__file__).resolve().parent / "fixtures" / "rds-samples.zip"
)
RDS_CONFIG_PATH = Path("configs/dense_gloomy_daylight_fog_camera.json")
SCENARIO_ORDER = (
    "clear_weather_camera_reference",
    "light_haze_soft_overcast",
    "moderate_gloomy_fog_nominal_camera",
    "underexposed_dense_gloom",
    "severe_low_contrast_sensor_stress",
)
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@pytest.fixture(scope="session")
def rds_sample_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root_env = os.environ.get("EULER_PREPROCESS_RDS_SAMPLE_ROOT")
    if root_env:
        root = Path(root_env)
        if not root.exists():
            pytest.fail(f"EULER_PREPROCESS_RDS_SAMPLE_ROOT does not exist: {root}")
        return _resolve_sample_root(root)

    zip_env = os.environ.get("EULER_PREPROCESS_RDS_SAMPLE_ZIP")
    if zip_env:
        zip_path = Path(zip_env)
        if not zip_path.exists():
            pytest.fail(f"EULER_PREPROCESS_RDS_SAMPLE_ZIP does not exist: {zip_path}")
    elif DEFAULT_RDS_SAMPLE_ZIP.exists():
        zip_path = DEFAULT_RDS_SAMPLE_ZIP
    elif _truthy_env("EULER_PREPROCESS_RDS_DOWNLOAD"):
        zip_path = tmp_path_factory.mktemp("rds-download") / "rds-samples.zip"
        _download_rds_samples(zip_path)
    else:
        pytest.skip(
            "Real Drive Sim samples not configured. Set "
            "tests/fixtures/rds-samples.zip, EULER_PREPROCESS_RDS_SAMPLE_ROOT, "
            "EULER_PREPROCESS_RDS_SAMPLE_ZIP, or EULER_PREPROCESS_RDS_DOWNLOAD=1."
        )

    extract_root = tmp_path_factory.mktemp("rds-samples")
    _extract_sample_zip(zip_path, extract_root)
    return _resolve_sample_root(extract_root)


@pytest.fixture(scope="session")
def rds_samples(rds_sample_root: Path) -> list[dict[str, Any]]:
    file_ids = _sample_file_ids(rds_sample_root)
    if not file_ids:
        pytest.fail(f"No Real Drive Sim sample IDs found under {rds_sample_root}")
    return [_load_rds_sample(rds_sample_root, file_id) for file_id in file_ids[:2]]


@pytest.fixture(scope="session")
def dense_gloomy_transform(tmp_path_factory: pytest.TempPathFactory) -> FogTransform:
    cfg = load_json(RDS_CONFIG_PATH)
    cfg["device"] = "cpu"
    cfg["gpu_batching"] = {"scenario_scope": "sample"}
    cfg_path = tmp_path_factory.mktemp("rds-config") / "dense_gloomy_cpu.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    return FogTransform(str(cfg_path), str(tmp_path_factory.mktemp("rds-output")))


@pytest.mark.integration
def test_real_drive_sim_cpu_loaders_read_rgb_depth_segmentation(
    rds_samples: list[dict[str, Any]],
) -> None:
    sample = rds_samples[0]

    rgb = sample["rgb"]
    depth = sample["depth"]
    sky_mask = sample["semantic_segmentation"]
    class_segmentation = sample["class_segmentation"]
    intrinsics = sample["intrinsics"]["intrinsics"]

    assert rgb.ndim == 3 and rgb.shape[-1] == 3
    assert rgb.dtype == np.float32
    assert 0.0 <= float(rgb.min()) <= float(rgb.max()) <= 1.0

    assert depth.shape == rgb.shape[:2]
    assert depth.dtype == np.float32
    assert np.isfinite(depth).all()
    assert float(depth.max()) > float(depth.min()) >= 0.0

    assert sky_mask.shape == rgb.shape[:2]
    assert sky_mask.dtype == np.bool_
    assert 0.0 < float(sky_mask.mean()) < 0.8
    np.testing.assert_array_equal(sky_mask, class_segmentation == 29)

    assert intrinsics.shape == (3, 3)
    assert np.isfinite(intrinsics).all()
    assert float(intrinsics[0, 0]) > 0.0
    assert float(intrinsics[1, 1]) > 0.0


@pytest.mark.integration
def test_dense_gloomy_scenario_profiles_render_real_drive_sim_samples(
    dense_gloomy_transform: FogTransform,
    rds_samples: list[dict[str, Any]],
) -> None:
    transform = dense_gloomy_transform

    profile_names = [
        transform._scenario_name(profile) for profile in transform.scenario_profiles
    ]
    assert tuple(profile_names) == SCENARIO_ORDER

    for sample in rds_samples:
        metrics = _render_scenario_metrics(transform, sample)

        opacity_by_scenario = [
            metrics[name]["opacity_mean"] for name in SCENARIO_ORDER
        ]
        assert opacity_by_scenario == sorted(opacity_by_scenario)
        assert metrics["severe_low_contrast_sensor_stress"]["opacity_mean"] > (
            metrics["clear_weather_camera_reference"]["opacity_mean"] + 0.25
        )

        assert metrics["underexposed_dense_gloom"]["luma_mean"] < (
            metrics["moderate_gloomy_fog_nominal_camera"]["luma_mean"] * 0.75
        )
        assert metrics["severe_low_contrast_sensor_stress"]["luma_mean"] < (
            metrics["clear_weather_camera_reference"]["luma_mean"] * 0.65
        )
        chromatically_bounded_profiles = SCENARIO_ORDER[:3]
        assert all(
            0.75 <= metrics[name]["blue_red_ratio"] <= 1.65
            for name in chromatically_bounded_profiles
        ), {sample["id"]: metrics}

        # The underexposed profiles can drive red near the black/noise floor,
        # so a blue/red ratio is not stable for the severe case. Bound
        # absolute blue bias instead to keep the stress profile from becoming
        # a blue wash.
        assert metrics["underexposed_dense_gloom"]["blue_red_ratio"] <= 2.5, {
            sample["id"]: metrics
        }
        stress_profiles = SCENARIO_ORDER[3:]
        assert all(
            metrics[name]["blue_minus_red_mean"] <= 0.1 for name in stress_profiles
        ), {sample["id"]: metrics}
        assert all(
            metrics[name]["blue_minus_red_p95"] <= 0.45 for name in stress_profiles
        ), {sample["id"]: metrics}


def _download_rds_samples(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(RDS_SAMPLE_URL, method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    target.write_bytes(payload)
    if not zipfile.is_zipfile(target):
        raise AssertionError(
            f"Downloaded Real Drive Sim sample payload is not a zip: {target}"
        )


def _extract_sample_zip(zip_path: Path, extract_root: Path) -> None:
    if not zipfile.is_zipfile(zip_path):
        pytest.fail(f"Real Drive Sim sample archive is not a zip: {zip_path}")
    extract_root = extract_root.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            name = member.filename
            if (
                name.startswith("__MACOSX/")
                or "/._" in name
                or name.endswith("/.DS_Store")
            ):
                continue
            target = (extract_root / name).resolve()
            if not target.is_relative_to(extract_root):
                pytest.fail(f"Unsafe path in Real Drive Sim sample archive: {name}")
            archive.extract(member, extract_root)


def _resolve_sample_root(root: Path) -> Path:
    root = root.resolve()
    if (root / "rds-samples").is_dir():
        root = root / "rds-samples"
    required = (
        root / "real-drive-sim+rgb",
        root / "real-drive-sim+depth",
        root / "real-drive-sim+semantic_segmentation",
        root / "calib.json",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.fail(
            f"Real Drive Sim sample root is missing required entries: {missing}"
        )
    return root


def _sample_file_ids(root: Path) -> list[str]:
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        file_ids = manifest.get("fileIds")
        if isinstance(file_ids, list):
            return [str(file_id) for file_id in file_ids]

    file_ids = []
    rgb_root = root / "real-drive-sim+rgb"
    for rgb_path in sorted(rgb_root.glob("*/*/*.png")):
        scene = rgb_path.parent.parent.name
        camera = rgb_path.parent.name
        frame = rgb_path.stem
        file_ids.append(f"{scene}~{camera}~{frame}")
    return file_ids


def _load_rds_sample(root: Path, file_id: str) -> dict[str, Any]:
    scene, camera, frame = file_id.split("~")
    rgb_path = root / "real-drive-sim+rgb" / scene / camera / f"{frame}.png"
    depth_path = root / "real-drive-sim+depth" / scene / camera / f"{frame}.npz"
    segmentation_path = (
        root
        / "real-drive-sim+semantic_segmentation"
        / scene
        / camera
        / f"{frame}.png"
    )
    rgb = real_drive_sim.rgb(str(rgb_path))
    depth = real_drive_sim.depth(str(depth_path))
    sky_mask = real_drive_sim.sky_mask(str(segmentation_path))
    class_segmentation = real_drive_sim.class_segmentation(str(segmentation_path))
    intrinsics = real_drive_sim.read_intrinsics(
        str(root / "calib.json"),
        attributes={"sensor": camera},
    )

    rgb, depth, sky_mask, class_segmentation, intrinsics = _downsample_sample(
        rgb,
        depth,
        sky_mask,
        class_segmentation,
        intrinsics,
        target_width=256,
    )
    return {
        "id": file_id,
        "full_id": f"/{scene}/{camera}/{frame}",
        "rgb": rgb,
        "depth": depth,
        "semantic_segmentation": sky_mask,
        "class_segmentation": class_segmentation,
        "intrinsics": {"intrinsics": intrinsics},
        "meta": {"rgb": {"path": str(rgb_path)}},
    }


def _downsample_sample(
    rgb: np.ndarray,
    depth: np.ndarray,
    sky_mask: np.ndarray,
    class_segmentation: np.ndarray,
    intrinsics: np.ndarray,
    *,
    target_width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = rgb.shape[:2]
    target_height = max(16, int(round(height * float(target_width) / float(width))))
    size = (target_width, target_height)

    rgb_u8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    rgb_small = np.asarray(
        Image.fromarray(rgb_u8).resize(size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    depth_small = np.asarray(
        Image.fromarray(depth).resize(size, Image.Resampling.NEAREST),
        dtype=np.float32,
    )
    class_small = np.asarray(
        Image.fromarray(class_segmentation.astype(np.uint8)).resize(
            size,
            Image.Resampling.NEAREST,
        ),
        dtype=np.int64,
    )
    mask_small = np.asarray(
        Image.fromarray(sky_mask.astype(np.uint8) * 255).resize(
            size,
            Image.Resampling.NEAREST,
        )
    ) > 0

    scaled_intrinsics = intrinsics.astype(np.float32, copy=True)
    scaled_intrinsics[0, :] *= float(target_width) / float(width)
    scaled_intrinsics[1, :] *= float(target_height) / float(height)
    return rgb_small, depth_small, mask_small, class_small, scaled_intrinsics


def _render_profile(
    transform: FogTransform,
    sample: dict[str, Any],
    profile: dict[str, Any],
    scenario_index: int,
):
    rng = np.random.default_rng(np.random.SeedSequence([1337, scenario_index, 1]))
    plan = transform._resolve_render_plan(
        rng,
        scenario=profile,
        sample_scenario=False,
        freeze_sampled_parameters=True,
    )
    return transform.pipeline.process_np(
        rgb=sample["rgb"],
        depth_m=sample["depth"],
        sky_mask=sample["semantic_segmentation"],
        model_name=plan.model_name,
        model_cfg=plan.model_cfg,
        rng=rng,
        sample_id=sample["id"],
        intrinsics=sample["intrinsics"]["intrinsics"],
        airlight_method=plan.airlight_method,
        capture_artifacts=plan.capture_artifacts,
    )


def _render_scenario_metrics(
    transform: FogTransform,
    sample: dict[str, Any],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for scenario_index, profile in enumerate(transform.scenario_profiles):
        name = transform._scenario_name(profile)
        assert name is not None
        result = _render_profile(transform, sample, profile, scenario_index)
        image = np.asarray(result.rgb, dtype=np.float32)
        k_map = np.asarray(result.k_map, dtype=np.float32)
        depth = np.asarray(sample["depth"], dtype=np.float32)
        opacity = 1.0 - np.exp(-np.maximum(k_map, 0.0) * np.maximum(depth, 0.0))

        assert image.shape == sample["rgb"].shape
        assert image.dtype == np.float32
        assert np.isfinite(image).all()
        assert float(image.min()) >= 0.0
        assert float(image.max()) <= 1.0
        assert np.isfinite(k_map).all()
        assert np.isfinite(result.airlight).all()
        assert float(result.beta) >= 0.0

        metrics[name] = {
            "beta": float(result.beta),
            "opacity_mean": float(opacity.mean()),
            "opacity_p95": float(np.percentile(opacity, 95.0)),
            "luma_mean": _mean_luminance(image),
            "blue_red_ratio": _blue_red_ratio(image),
            "blue_minus_red_mean": _blue_minus_red_mean(image),
            "blue_minus_red_p95": _blue_minus_red_p95(image),
        }
    return metrics


def _mean_luminance(image: np.ndarray) -> float:
    return float(
        (np.asarray(image, dtype=np.float32) * LUMA_WEIGHTS).sum(axis=-1).mean()
    )


def _blue_red_ratio(image: np.ndarray) -> float:
    arr = np.asarray(image, dtype=np.float32)
    red = float(arr[..., 0].mean())
    blue = float(arr[..., 2].mean())
    return blue / max(red, 1e-6)


def _blue_minus_red_mean(image: np.ndarray) -> float:
    arr = np.asarray(image, dtype=np.float32)
    return float((arr[..., 2] - arr[..., 0]).mean())


def _blue_minus_red_p95(image: np.ndarray) -> float:
    arr = np.asarray(image, dtype=np.float32)
    return float(np.percentile(arr[..., 2] - arr[..., 0], 95.0))


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
