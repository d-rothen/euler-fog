from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from euler_preprocess.common.io import load_json
from euler_preprocess.fog.capture import CaptureContext
from euler_preprocess.fog.transform import FogTransform


DEFAULT_CONFIG = Path("configs/dense_gloomy_daylight_fog_camera.json")
DEFAULT_PROFILES = (
    "moderate_gloomy_fog_nominal_camera",
    "underexposed_dense_gloom",
    "severe_low_contrast_sensor_stress",
)
LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    if args.device is not None:
        config["device"] = args.device

    temp_config = _write_temp_config(config)
    try:
        transform = FogTransform(str(temp_config), args.output_dir)
    finally:
        temp_config.unlink(missing_ok=True)
    rgb, depth, sky_mask = _synthetic_dark_road_scene(args.height, args.width)

    requested_profiles = tuple(args.profile or DEFAULT_PROFILES)
    found_profiles: set[str] = set()
    for index, profile in enumerate(transform.scenario_profiles):
        name = transform._scenario_name(profile) or f"scenario_{index:03d}"
        if requested_profiles and name not in requested_profiles:
            continue
        found_profiles.add(name)
        _profile_scenario(
            transform,
            profile,
            name=name,
            index=index,
            rgb=rgb,
            depth=depth,
            sky_mask=sky_mask,
            seed=args.seed,
            cprofile_rows=args.cprofile_rows,
        )

    missing = set(requested_profiles) - found_profiles
    if missing:
        raise SystemExit(f"Scenario profile(s) not found: {', '.join(sorted(missing))}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile dark fog scenarios for blue tint and blue-channel speckle."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Fog config to profile. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--profile",
        action="append",
        help=(
            "Scenario profile name to run. Repeat to profile multiple scenarios. "
            "Defaults to the authored dark/gloom profiles."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Override config device for profiling. Default: cpu. Use an empty "
            "string to preserve the config value."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Base seed used for deterministic scenario profiling.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Synthetic image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=180,
        help="Synthetic image height.",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/fog-dark-blue-profile",
        help="Throwaway output directory required by FogTransform.",
    )
    parser.add_argument(
        "--cprofile-rows",
        type=int,
        default=8,
        help="Number of cumulative cProfile rows to print for each scenario.",
    )
    args = parser.parse_args()
    if args.device == "":
        args.device = None
    if args.width < 16 or args.height < 16:
        raise SystemExit("--width and --height must be >= 16")
    return args


def _write_temp_config(config: dict[str, Any]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w",
        suffix=".json",
        prefix="fog-dark-blue-profile-",
        delete=False,
    )
    with handle:
        json.dump(config, handle)
    return Path(handle.name)


def _synthetic_dark_road_scene(
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    sky = y < 0.33

    rgb = np.zeros((height, width, 3), dtype=np.float32)
    rgb[:] = np.array([0.28, 0.30, 0.30], dtype=np.float32)
    rgb[sky[:, 0]] = np.array([0.55, 0.62, 0.72], dtype=np.float32)
    rgb[..., 0] += 0.08 * x
    rgb[..., 1] += 0.05 * (1.0 - y)
    rgb[..., 2] += 0.04 * y

    # Two dark objects create low-luminance regions where shadow-recovery noise
    # and fog/depth weighting become visible.
    y0 = int(round(height * 0.44))
    y1 = int(round(height * 0.83))
    rgb[y0:y1, int(width * 0.12) : int(width * 0.36)] *= np.array(
        [0.35, 0.34, 0.33],
        dtype=np.float32,
    )
    rgb[y0 + 12 : y1 + 10, int(width * 0.66) : int(width * 0.91)] *= np.array(
        [0.30, 0.31, 0.34],
        dtype=np.float32,
    )

    depth = (5.0 + 95.0 * y + 10.0 * np.sin(2.0 * np.pi * x)).astype(np.float32)
    sky_mask = np.broadcast_to(sky, (height, width)).copy()
    return np.clip(rgb, 0.0, 1.0), depth, sky_mask


def _profile_scenario(
    transform: FogTransform,
    profile: dict[str, Any],
    *,
    name: str,
    index: int,
    rgb: np.ndarray,
    depth: np.ndarray,
    sky_mask: np.ndarray,
    seed: int,
    cprofile_rows: int,
) -> None:
    rng = np.random.default_rng(np.random.SeedSequence([seed, index, 999]))
    plan = transform._resolve_render_plan(
        rng,
        scenario=profile,
        sample_scenario=False,
        freeze_sampled_parameters=True,
    )

    profiler = cProfile.Profile()
    profiler.enable()
    render_start = time.perf_counter()
    render = transform.pipeline.render_scene_np(
        rgb=rgb,
        depth_m=depth,
        sky_mask=sky_mask,
        model_name=plan.model_name,
        model_cfg=plan.model_cfg,
        rng=rng,
        sample_id="synthetic",
        airlight_method=plan.airlight_method,
    )
    render_seconds = time.perf_counter() - render_start

    context = CaptureContext(
        sample_id="synthetic",
        rng=rng,
        depth_m=depth,
        k_map=render.k_map,
        attributes={
            "sky_mask": sky_mask,
            "airlight": render.airlight,
            "render_input_space": plan.model_cfg.get(
                "render_input_space",
                transform.pipeline.render_input_space,
            ),
        },
    )

    stage_rows = []
    image = render.rgb
    for stage in plan.capture_artifacts.stages if plan.capture_artifacts else ():
        before = _image_metrics(image)
        stage_start = time.perf_counter()
        next_image = stage.apply_np(image, context)
        stage_seconds = time.perf_counter() - stage_start
        after = _image_metrics(next_image)
        stage_rows.append((stage.name, stage_seconds, before, after))
        image = next_image
    profiler.disable()

    k_map = np.asarray(render.k_map, dtype=np.float32)
    opacity = 1.0 - np.exp(-np.maximum(k_map, 0.0) * np.maximum(depth, 0.0))

    print()
    print(f"Scenario: {name}")
    print(f"  model={plan.model_name} beta={render.beta:.5f}")
    print(
        "  airlight=%s ls_mean=%s opacity_mean=%.3f opacity_p95=%.3f"
        % (
            _format_triplet(render.airlight),
            _format_triplet(np.asarray(render.ls_map).mean(axis=(0, 1))),
            float(opacity.mean()),
            float(np.percentile(opacity, 95)),
        )
    )
    print(f"  input:  {_format_metrics(_image_metrics(rgb))}")
    print(f"  render: {_format_metrics(_image_metrics(render.rgb))}")
    print(f"          render_seconds={render_seconds:.4f}")
    for stage_name, seconds, before, after in stage_rows:
        print(
            "  %-9s %7.4fs  b/r %.3f -> %.3f  "
            "lowq_b/r %.3f -> %.3f  lowq_b-r_p95 %.4f -> %.4f"
            % (
                stage_name,
                seconds,
                before["b_over_r"],
                after["b_over_r"],
                before["lowq_b_over_r"],
                after["lowq_b_over_r"],
                before["lowq_b_minus_r_p95"],
                after["lowq_b_minus_r_p95"],
            )
        )
    print(f"  final:  {_format_metrics(_image_metrics(image))}")

    if cprofile_rows > 0:
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("cumtime").print_stats(
            cprofile_rows
        )
        print("  cProfile:")
        for line in stream.getvalue().splitlines()[4: 5 + cprofile_rows]:
            if line.strip():
                print(f"    {line}")


def _image_metrics(image: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(image, dtype=np.float32)
    pixels = arr.reshape(-1, 3)
    luma = pixels @ LUMA_WEIGHTS
    lowq = luma <= np.percentile(luma, 25)
    low_pixels = pixels[lowq]

    b_minus_r = pixels[:, 2] - pixels[:, 0]
    b_minus_g = pixels[:, 2] - pixels[:, 1]
    low_b_minus_r = low_pixels[:, 2] - low_pixels[:, 0]
    low_b_minus_g = low_pixels[:, 2] - low_pixels[:, 1]

    return {
        "mean": pixels.mean(axis=0),
        "b_over_r": _channel_ratio(pixels, 2, 0),
        "b_minus_r_mean": float(b_minus_r.mean()),
        "b_minus_r_p95": float(np.percentile(b_minus_r, 95)),
        "b_minus_r_std": float(b_minus_r.std()),
        "blue_over_red_fraction": float(np.mean(b_minus_r > 0.03)),
        "blue_over_green_fraction": float(np.mean(b_minus_g > 0.03)),
        "luma": float(luma.mean()),
        "lowq_b_over_r": _channel_ratio(low_pixels, 2, 0),
        "lowq_b_minus_r_mean": float(low_b_minus_r.mean()),
        "lowq_b_minus_r_p95": float(np.percentile(low_b_minus_r, 95)),
        "lowq_b_minus_r_std": float(low_b_minus_r.std()),
        "lowq_blue_over_red_fraction": float(np.mean(low_b_minus_r > 0.03)),
        "lowq_blue_over_green_fraction": float(np.mean(low_b_minus_g > 0.03)),
    }


def _channel_ratio(pixels: np.ndarray, numerator: int, denominator: int) -> float:
    return float(
        pixels[:, numerator].mean() / max(float(pixels[:, denominator].mean()), 1e-6)
    )


def _format_metrics(metrics: dict[str, Any]) -> str:
    return (
        f"mean={_format_triplet(metrics['mean'])} "
        f"b/r={metrics['b_over_r']:.3f} "
        f"b-r_mean={metrics['b_minus_r_mean']:.4f} "
        f"b-r_p95={metrics['b_minus_r_p95']:.4f} "
        f"b-r_std={metrics['b_minus_r_std']:.4f} "
        f"blueR%={metrics['blue_over_red_fraction']:.3f} "
        f"lowq_b/r={metrics['lowq_b_over_r']:.3f} "
        f"lowq_b-r_p95={metrics['lowq_b_minus_r_p95']:.4f} "
        f"lowq_blueR%={metrics['lowq_blue_over_red_fraction']:.3f} "
        f"luma={metrics['luma']:.3f}"
    )


def _format_triplet(value: Any) -> str:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)[:3]
    return "[" + ", ".join(f"{float(component):.4f}" for component in arr) + "]"


if __name__ == "__main__":
    main()
