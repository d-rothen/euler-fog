from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from euler_preprocess.common.color import srgb_to_linear
from euler_preprocess.common.device import torch_generator_for_index
from euler_preprocess.common.intrinsics import (
    planar_to_radial_depth,
    planar_to_radial_depth_torch,
)
from euler_preprocess.fog.pipeline import FogPipelineResult
from euler_preprocess.fog.transform import FogTransform


@pytest.fixture
def torch_runtime():
    torch = pytest.importorskip("torch", reason="Torch fog pathway is unavailable")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    try:
        torch.zeros(1, device=device)
    except Exception as exc:  # pragma: no cover - backend-specific availability
        pytest.skip(f"Torch device {device} is unavailable: {exc}")
    return torch, device


def _transform(
    tmp_path: Path,
    config: dict[str, Any],
    torch_runtime,
    *,
    name: str,
) -> FogTransform:
    _torch, device = torch_runtime
    config = dict(config)
    config["device"] = "cpu"
    config_path = tmp_path / f"{name}.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    transform = FogTransform(
        config_path=str(config_path),
        out_path=str(tmp_path / f"{name}_out"),
    )
    transform.torch_device = device
    return transform


def _synthetic_scene(
    height: int = 18,
    width: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.08, 0.92, width, dtype=np.float32)
    y = np.linspace(0.12, 0.82, height, dtype=np.float32)
    rgb = np.dstack(
        [
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y[:, None], (height, width)),
            0.25
            + 0.45
            * np.broadcast_to(x, (height, width))
            * np.broadcast_to(y[:, None], (height, width)),
        ]
    ).astype(np.float32)
    rgb[: height // 3, :, 0] = 0.72
    rgb[: height // 3, :, 1] = 0.82
    rgb[: height // 3, :, 2] = 0.94
    depth = np.linspace(3.0, 70.0, height * width, dtype=np.float32).reshape(
        height,
        width,
    )
    sky_mask = np.zeros((height, width), dtype=bool)
    sky_mask[: height // 3, :] = True
    return rgb, depth, sky_mask


def _render_cpu_and_torch(
    transform: FogTransform,
    torch_runtime,
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    sky_mask: np.ndarray,
    model_name: str,
    model_cfg: dict[str, Any],
    seed: int,
    sample_id: str,
    intrinsics: np.ndarray | None = None,
) -> tuple[FogPipelineResult, FogPipelineResult]:
    torch, device = torch_runtime
    cpu = transform.pipeline.process_np(
        rgb=rgb,
        depth_m=depth_m,
        sky_mask=sky_mask,
        model_name=model_name,
        model_cfg=model_cfg,
        rng=np.random.default_rng(seed),
        sample_id=sample_id,
        intrinsics=intrinsics,
    )

    rgb_t = torch.from_numpy(rgb).to(device=device, dtype=torch.float32)
    depth_t = torch.from_numpy(depth_m).to(device=device, dtype=torch.float32)
    sky_t = torch.from_numpy(sky_mask).to(device=device, dtype=torch.bool)
    torch_gen = torch.Generator(device=device).manual_seed(seed + 10_000)
    rgb_out, beta, airlight, k_map, ls_map = transform._process_torch_pipeline(
        rgb_t,
        depth_t,
        sky_t,
        model_name,
        model_cfg,
        np.random.default_rng(seed),
        torch_gen,
        sample_id=sample_id,
        intrinsics=intrinsics,
    )
    gpu = FogPipelineResult(
        rgb=rgb_out.detach().cpu().numpy(),
        beta=beta,
        airlight=airlight.detach().cpu().numpy(),
        k_map=k_map.detach().cpu().numpy(),
        ls_map=ls_map.detach().cpu().numpy(),
    )
    return cpu, gpu


def _assert_results_close(
    cpu: FogPipelineResult,
    gpu: FogPipelineResult,
    *,
    atol: float = 2e-5,
) -> None:
    assert gpu.beta == pytest.approx(cpu.beta, abs=1e-8)
    np.testing.assert_allclose(gpu.airlight, cpu.airlight, atol=atol, rtol=0)
    np.testing.assert_allclose(gpu.k_map, cpu.k_map, atol=atol, rtol=0)
    np.testing.assert_allclose(gpu.ls_map, cpu.ls_map, atol=atol, rtol=0)
    np.testing.assert_allclose(gpu.rgb, cpu.rgb, atol=atol, rtol=0)


def test_uniform_srgb_fog_matches_cpu_end_to_end(
    tmp_path: Path,
    torch_runtime,
) -> None:
    config = {
        "airlight": "from_sky",
        "seed": 101,
        "render_input_space": "srgb",
    }
    transform = _transform(tmp_path, config, torch_runtime, name="uniform_srgb")
    rgb, depth, sky_mask = _synthetic_scene()
    model_cfg = {
        "visibility_m": 65.0,
        "atmospheric_light": "from_sky",
        "airlight_dampening": {"enabled": False},
    }

    cpu, gpu = _render_cpu_and_torch(
        transform,
        torch_runtime,
        rgb=rgb,
        depth_m=depth,
        sky_mask=sky_mask,
        model_name="uniform",
        model_cfg=model_cfg,
        seed=17,
        sample_id="uniform_srgb",
    )

    expected_airlight = srgb_to_linear(rgb)[sky_mask].mean(axis=0)
    np.testing.assert_allclose(cpu.airlight, expected_airlight, atol=2e-6)
    _assert_results_close(cpu, gpu)


def test_scene_illumination_with_sky_mask_matches_cpu(
    tmp_path: Path,
    torch_runtime,
) -> None:
    config = {"airlight": "from_sky", "seed": 102}
    transform = _transform(tmp_path, config, torch_runtime, name="illumination")
    rgb, depth, sky_mask = _synthetic_scene()
    model_cfg = {
        "visibility_m": 48.0,
        "atmospheric_light": [0.68, 0.72, 0.78],
        "airlight_dampening": {"enabled": False},
        "scene_illumination": {
            "enabled": True,
            "global_ev": 0.65,
            "near_ev": 0.8,
            "near_decay_depth_m": 11.0,
            "fog_coupled_ev": 0.35,
            "airlight_ev_ratio": 0.55,
            "sky_weight": 0.0,
            "min_radiance_scale": 0.04,
        },
    }

    cpu, gpu = _render_cpu_and_torch(
        transform,
        torch_runtime,
        rgb=rgb,
        depth_m=depth,
        sky_mask=sky_mask,
        model_name="uniform",
        model_cfg=model_cfg,
        seed=23,
        sample_id="illumination",
    )

    _assert_results_close(cpu, gpu)
    sky_airlight = cpu.ls_map[sky_mask]
    np.testing.assert_allclose(
        sky_airlight,
        np.broadcast_to(cpu.airlight, sky_airlight.shape),
        atol=2e-6,
    )
    assert float(cpu.ls_map[~sky_mask].mean()) < float(cpu.airlight.mean())


def test_fog_aware_capture_context_matches_cpu(
    tmp_path: Path,
    torch_runtime,
) -> None:
    config = {
        "airlight": "from_sky",
        "seed": 103,
        "render_input_space": "srgb",
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
                    "fog_coupled_glare": {
                        "enabled": True,
                        "base_strength": 0.01,
                        "fog_strength": 0.08,
                        "highlight_strength": 0.0,
                        "airlight_strength": 0.12,
                        "smooth_sigma": 0.0,
                        "color": [0.9, 0.94, 1.0],
                    },
                    "vignetting_strength": 0.0,
                    "windshield_haze": {"enabled": False},
                    "droplets": {"enabled": False},
                },
                {
                    "type": "sensor",
                    "input_space": "srgb",
                    "exposure_gain": 1.0,
                    "auto_exposure": {
                        "enabled": True,
                        "metering": "fog_aware_center_weighted",
                        "target_luminance": 0.22,
                        "sky_suppression": 0.95,
                        "fog_meter_suppression": 0.7,
                        "depth_meter_decay_m": 30.0,
                        "highlight_protection": 0.0,
                        "min_gain": 0.2,
                        "max_gain": 3.0,
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
                    "sensor_identity": {"enabled": False},
                    "shadow_recovery_noise": {"enabled": False},
                    "demosaic": True,
                },
            ]
        },
    }
    transform = _transform(tmp_path, config, torch_runtime, name="capture_context")
    rgb, depth, sky_mask = _synthetic_scene(height=20, width=26)
    model_cfg = {
        "visibility_m": 38.0,
        "atmospheric_light": "from_sky",
        "airlight_dampening": {"enabled": False},
    }

    cpu, gpu = _render_cpu_and_torch(
        transform,
        torch_runtime,
        rgb=rgb,
        depth_m=depth,
        sky_mask=sky_mask,
        model_name="uniform",
        model_cfg=model_cfg,
        seed=29,
        sample_id="capture_context",
    )

    _assert_results_close(cpu, gpu, atol=5e-5)


def test_uniform_gpu_batch_matches_gpu_single_and_cpu_with_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    torch_runtime,
) -> None:
    torch, device = torch_runtime
    config = {
        "airlight": "from_sky",
        "seed": 211,
        "render_input_space": "srgb",
        "gpu_batch_size": 2,
        "gpu_batching": {
            "scenario_scope": "batch",
            "condition_parameter_scope": "batch",
        },
        "selection": {"mode": "fixed", "model": "uniform"},
        "models": {
            "uniform": {
                "visibility_m": 58.0,
                "atmospheric_light": "from_sky",
                "airlight_dampening": {"enabled": False},
                "scene_illumination": {
                    "enabled": True,
                    "global_ev": 0.25,
                    "near_ev": 0.35,
                    "near_decay_depth_m": 14.0,
                    "fog_coupled_ev": 0.15,
                    "airlight_ev_ratio": 0.2,
                    "sky_weight": 0.0,
                },
            }
        },
        "capture": {
            "stages": [
                {
                    "type": "exposure",
                    "gain": 0.82,
                    "white_balance": [1.02, 0.98, 1.0],
                    "white_balance_jitter": 0.0,
                }
            ]
        },
    }
    transform = _transform(tmp_path, config, torch_runtime, name="uniform_batch")
    rgb_a, depth_a, sky_a = _synthetic_scene(height=16, width=22)
    rgb_b = np.flip(rgb_a, axis=1).copy()
    depth_b = np.flip(depth_a, axis=0).copy()
    sky_b = sky_a.copy()
    intrinsics = np.array(
        [[20.0, 0.0, 10.5], [0.0, 19.0, 7.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    samples = [
        {
            "id": "batch_a",
            "rgb": rgb_a,
            "depth": depth_a,
            "semantic_segmentation": sky_a,
            "intrinsics": {"intrinsics": intrinsics},
        },
        {
            "id": "batch_b",
            "rgb": rgb_b,
            "depth": depth_b,
            "semantic_segmentation": sky_b,
            "intrinsics": {"intrinsics": intrinsics},
        },
    ]

    captured: dict[str, np.ndarray] = {}

    def capture_write(sample, value, **_kwargs):
        captured[sample["id"]] = np.asarray(value, dtype=np.float32).copy()
        return tmp_path / f"{sample['id']}.png"

    monkeypatch.setattr(transform.output_backend, "write", capture_write)
    transform._generate_fog_gpu(samples)
    assert set(captured) == {"batch_a", "batch_b"}

    plan = transform._resolve_gpu_batch_render_plan(0)
    assert plan is not None
    for index, sample in enumerate(samples):
        rng_cpu = transform._rng_for(index)
        radial_cpu = planar_to_radial_depth(sample["depth"], intrinsics)
        cpu = transform.pipeline.process_np(
            rgb=sample["rgb"],
            depth_m=radial_cpu,
            sky_mask=sample["semantic_segmentation"],
            model_name=plan.model_name,
            model_cfg=plan.model_cfg,
            rng=rng_cpu,
            sample_id=sample["id"],
            intrinsics=intrinsics,
            airlight_method=plan.airlight_method,
            capture_artifacts=plan.capture_artifacts,
        )

        depth_t = torch.from_numpy(sample["depth"]).to(
            device=device,
            dtype=torch.float32,
        )
        intrinsics_t = torch.from_numpy(intrinsics).to(
            device=device,
            dtype=torch.float32,
        )
        radial_t = planar_to_radial_depth_torch(depth_t, intrinsics_t)
        single = transform._process_torch_pipeline(
            torch.from_numpy(sample["rgb"]).to(device=device, dtype=torch.float32),
            radial_t,
            torch.from_numpy(sample["semantic_segmentation"]).to(
                device=device,
                dtype=torch.bool,
            ),
            plan.model_name,
            plan.model_cfg,
            transform._rng_for(index),
            torch_generator_for_index(
                device,
                transform.seed,
                transform.base_rng,
                index,
            ),
            sample_id=sample["id"],
            intrinsics=intrinsics,
            airlight_method=plan.airlight_method,
            capture_artifacts=plan.capture_artifacts,
        )[0]
        single_np = single.detach().cpu().numpy()

        np.testing.assert_allclose(captured[sample["id"]], single_np, atol=2e-5)
        np.testing.assert_allclose(captured[sample["id"]], cpu.rgb, atol=3e-5)


def test_heterogeneous_cpu_gpu_behavioral_invariants(
    tmp_path: Path,
    torch_runtime,
) -> None:
    config = {
        "airlight": "from_sky",
        "seed": 104,
        "render_input_space": "srgb",
    }
    transform = _transform(tmp_path, config, torch_runtime, name="heterogeneous")
    rgb, depth, sky_mask = _synthetic_scene(height=32, width=40)
    model_cfg = {
        "visibility_m": 52.0,
        "atmospheric_light": "from_sky",
        "airlight_dampening": {"enabled": False},
        "k_hetero": {
            "scales": [24, 12, 6],
            "min_factor": 0.65,
            "max_factor": 1.35,
            "normalize_to_mean": True,
            "contrast": 0.7,
        },
        "ls_hetero": {
            "scales": [30, 15, 7],
            "min_factor": 0.82,
            "max_factor": 1.08,
            "normalize_to_mean": True,
            "contrast": 0.6,
        },
        "scene_illumination": {
            "enabled": True,
            "global_ev": 0.2,
            "near_ev": 0.25,
            "near_decay_depth_m": 12.0,
            "fog_coupled_ev": 0.1,
            "airlight_ev_ratio": 0.15,
            "sky_weight": 0.0,
        },
    }

    cpu, gpu = _render_cpu_and_torch(
        transform,
        torch_runtime,
        rgb=rgb,
        depth_m=depth,
        sky_mask=sky_mask,
        model_name="heterogeneous_k_ls",
        model_cfg=model_cfg,
        seed=41,
        sample_id="heterogeneous",
    )

    assert np.isfinite(gpu.rgb).all()
    assert np.isfinite(gpu.k_map).all()
    assert np.isfinite(gpu.ls_map).all()
    assert 0.0 <= float(gpu.rgb.min()) <= float(gpu.rgb.max()) <= 1.0
    assert float(gpu.k_map.min()) >= 0.0
    assert gpu.beta == pytest.approx(cpu.beta, abs=1e-8)
    np.testing.assert_allclose(gpu.airlight, cpu.airlight, atol=2e-5, rtol=0)
    assert float(gpu.k_map.mean()) == pytest.approx(gpu.beta, rel=2e-4)
    assert float(cpu.k_map.mean()) == pytest.approx(cpu.beta, rel=2e-4)
    assert abs(float(gpu.ls_map.mean()) - float(cpu.ls_map.mean())) < 0.08
    assert abs(float(gpu.rgb.mean()) - float(cpu.rgb.mean())) < 0.08
