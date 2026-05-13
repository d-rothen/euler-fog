from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from euler_preprocess.common.io import load_json
from euler_preprocess.fog.transform import FogTransform


DEFAULT_INPUT_ROOT = Path("/Volumes/Volume/Datasets/tmp/test_samples")
DEFAULT_CONFIG = Path("configs/dense_gloomy_daylight_fog_camera.json")
DEFAULT_OUTPUT_DIR = Path(".qualitative_fog_outputs/dense_gloomy_daylight")
DEFAULT_SKY_CLASS = (29, 0, 0)


def main() -> None:
    args = _parse_args()
    input_root = args.input_root.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()

    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = list(
        load_structured_samples(
            input_root,
            sky_class=args.sky_class,
            camera_filter=args.camera,
            depth_key=args.depth_key,
        )
    )
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(f"No samples found under {input_root}")

    effective_config = _write_effective_config(
        config_path,
        output_dir,
        device=args.device,
        seed=args.seed,
        depth_scale=args.depth_scale,
    )

    transform = FogTransform(
        config_path=str(effective_config),
        out_path=str(output_dir / "fog_outputs"),
    )
    saved_paths = transform.run(samples)

    if args.review_panels:
        write_review_panels(samples, saved_paths, output_dir / "review")

    manifest = {
        "input_root": str(input_root),
        "config_path": str(config_path),
        "effective_config_path": str(effective_config),
        "output_paths": [str(path) for path in saved_paths],
        "sample_ids": [str(sample["id"]) for sample in samples],
        "sky_class": list(args.sky_class),
    }
    (output_dir / "qualitative_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(saved_paths)} fog outputs to {output_dir / 'fog_outputs'}")
    if args.review_panels:
        print(f"Wrote review panels to {output_dir / 'review'}")


def load_structured_samples(
    root: Path,
    *,
    sky_class: tuple[int, int, int] = DEFAULT_SKY_CLASS,
    camera_filter: str | None = None,
    depth_key: str | None = None,
) -> list[dict[str, Any]]:
    calibration = _load_calibration(root / "calibration.json")
    rgb_root = root / "rgb"
    if not rgb_root.exists():
        raise FileNotFoundError(f"Missing rgb directory: {rgb_root}")

    samples: list[dict[str, Any]] = []
    for rgb_path in sorted(rgb_root.glob("*/*/*.png")):
        scene = rgb_path.parent.parent.name
        camera = rgb_path.parent.name
        frame = rgb_path.stem
        if camera_filter and camera != camera_filter:
            continue

        rel_png = Path(scene) / camera / f"{frame}.png"
        rel_npz = Path(scene) / camera / f"{frame}.npz"
        depth_path = root / "depth" / rel_npz
        seg_path = _first_existing(
            root / "semantic_segmentation" / rel_png,
            root / "semantic_segmentation_2d" / rel_png,
        )
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth for {rgb_path}: {depth_path}")
        if seg_path is None:
            raise FileNotFoundError(
                f"Missing semantic segmentation for {rgb_path}: {rel_png}"
            )

        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        depth = _load_depth_npz(depth_path, depth_key)
        sky_mask = _load_sky_mask(seg_path, sky_class)
        intrinsics = calibration.get(camera)
        if intrinsics is None:
            raise KeyError(f"No intrinsics for camera '{camera}' in calibration.json")

        sample_id = f"{scene}~{camera}~{frame}"
        samples.append(
            {
                "id": sample_id,
                "full_id": f"/{scene}/{camera}/{frame}",
                "rgb": rgb,
                "depth": depth,
                "semantic_segmentation": sky_mask,
                "intrinsics": {"intrinsics": intrinsics},
                "meta": {"rgb": {"path": str(rgb_path)}},
            }
        )
    return samples


def write_review_panels(
    samples: list[dict[str, Any]],
    saved_paths: list[Path],
    review_dir: Path,
) -> None:
    review_dir.mkdir(parents=True, exist_ok=True)
    if len(samples) != len(saved_paths):
        raise ValueError(
            f"Expected one output per sample, got {len(saved_paths)} outputs "
            f"for {len(samples)} samples"
        )
    for sample, fog_path in zip(samples, saved_paths):
        rgb = Image.fromarray(np.asarray(sample["rgb"], dtype=np.uint8), mode="RGB")
        foggy = Image.open(fog_path).convert("RGB")
        if foggy.size != rgb.size:
            foggy = foggy.resize(rgb.size, resample=Image.Resampling.BILINEAR)

        panel = Image.new("RGB", (rgb.width * 2, rgb.height), color=(0, 0, 0))
        panel.paste(rgb, (0, 0))
        panel.paste(foggy, (rgb.width, 0))
        panel.save(review_dir / f"{sample['id']}_input_vs_fog.png")

        mask = np.asarray(sample["semantic_segmentation"], dtype=bool)
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        mask_img.save(review_dir / f"{sample['id']}_sky_mask.png")

        depth_preview = _depth_preview(np.asarray(sample["depth"], dtype=np.float32))
        Image.fromarray(depth_preview, mode="L").save(
            review_dir / f"{sample['id']}_depth_preview.png"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a structured rgb/depth/semantic_segmentation/calibration folder, "
            "run FogTransform, and write images for qualitative review."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--camera", default="CS_FRONT")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--depth-key", default=None)
    parser.add_argument("--device", default=None, help="Override config device.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="Override config depth_scale.",
    )
    parser.add_argument(
        "--sky-class",
        type=_parse_rgb_triplet,
        default=DEFAULT_SKY_CLASS,
        help="RGB semantic class used as sky, default: 29,0,0.",
    )
    parser.add_argument(
        "--no-review-panels",
        dest="review_panels",
        action="store_false",
        help="Only write transform outputs, not side-by-side review panels.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output directory before running.",
    )
    parser.set_defaults(review_panels=True)
    return parser.parse_args()


def _write_effective_config(
    config_path: Path,
    output_dir: Path,
    *,
    device: str | None,
    seed: int | None,
    depth_scale: float | None,
) -> Path:
    cfg = load_json(config_path)
    if device is not None:
        cfg["device"] = device
    if seed is not None:
        cfg["seed"] = seed
    if depth_scale is not None:
        cfg["depth_scale"] = depth_scale

    config_out = output_dir / "effective_fog_config.json"
    config_out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return config_out


def _load_calibration(path: Path) -> dict[str, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload.get("names", [])
    intrinsics = payload.get("intrinsics", [])
    if not isinstance(names, list) or not isinstance(intrinsics, list):
        raise ValueError("calibration.json must contain names and intrinsics lists")

    result: dict[str, np.ndarray] = {}
    for name, intr in zip(names, intrinsics):
        if not isinstance(intr, dict):
            continue
        fx = float(intr.get("fx", 0.0))
        fy = float(intr.get("fy", 0.0))
        if fx == 0.0 or fy == 0.0:
            continue
        result[str(name)] = np.array(
            [
                [fx, float(intr.get("skew", 0.0)), float(intr.get("cx", 0.0))],
                [0.0, fy, float(intr.get("cy", 0.0))],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    if not result:
        raise ValueError(f"No usable camera intrinsics found in {path}")
    return result


def _load_depth_npz(path: Path, depth_key: str | None) -> np.ndarray:
    with np.load(path) as data:
        key = depth_key or ("data" if "data" in data.files else data.files[0])
        if key not in data.files:
            raise KeyError(f"{path} does not contain depth key '{key}'")
        return np.asarray(data[key], dtype=np.float32)


def _load_sky_mask(
    path: Path,
    sky_class: tuple[int, int, int],
) -> np.ndarray:
    seg = np.asarray(Image.open(path).convert("RGB"))
    target = np.asarray(sky_class, dtype=np.uint8).reshape(1, 1, 3)
    return np.all(seg == target, axis=-1)


def _depth_preview(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0)
    if not valid.any():
        return np.zeros(depth.shape, dtype=np.uint8)
    clipped = np.clip(depth, 0.0, np.percentile(depth[valid], 98.0))
    clipped = np.log1p(clipped)
    denom = max(float(clipped[valid].max() - clipped[valid].min()), 1e-6)
    normalized = (clipped - float(clipped[valid].min())) / denom
    return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)


def _parse_rgb_triplet(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected R,G,B triplet")
    try:
        triplet = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected integer R,G,B triplet") from exc
    if any(component < 0 or component > 255 for component in triplet):
        raise argparse.ArgumentTypeError("RGB components must be in [0, 255]")
    return triplet  # type: ignore[return-value]


def _first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


if __name__ == "__main__":
    main()
