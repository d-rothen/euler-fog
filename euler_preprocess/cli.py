"""CLI entry point for euler-preprocess.

Used both by ``python main.py`` and by the installed ``euler-preprocess`` console
script.  Supports subcommands: ``fog``, ``sky-depth``, ``radial``.
"""
from __future__ import annotations

import argparse
import inspect
import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from euler_preprocess.common.dataset import build_dataset
from euler_preprocess.common.logging import get_logger, log_dataset_info
from euler_preprocess.common.output import prepare_output_backends


def _resolve(path_str: str, config_dir: Path) -> Path:
    """Resolve a path relative to the config file's directory."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (config_dir / p).resolve()


class _SelectedSamples(Sequence):
    """Lazy view over selected euler-loading dataset entries."""

    def __init__(self, dataset, indices: Iterable[int]) -> None:
        self.dataset = dataset
        self.indices = tuple(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[dict]:
        for index in self.indices:
            yield self.dataset[index]

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return [self.dataset[i] for i in self.indices[index]]
        return self.dataset[self.indices[index]]


def _validate_sample_index(value: Any, *, key: str, dataset_size: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be a non-negative integer index")
    if value < 0:
        raise ValueError(f"{key} must be a non-negative integer index")
    if value >= dataset_size:
        raise IndexError(
            f"{key} {value} out of range for dataset of length {dataset_size}"
        )
    return value


def _positive_int(value: Any, *, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _non_negative_int(value: Any, *, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _resolve_sample_indices(selection: Any, *, dataset_size: int) -> tuple[int, ...]:
    if isinstance(selection, list):
        indices = tuple(
            _validate_sample_index(value, key="samples[]", dataset_size=dataset_size)
            for value in selection
        )
        if not indices:
            raise ValueError("samples must select at least one dataset entry")
        return indices

    if not isinstance(selection, dict):
        raise ValueError("samples must be an object or a list of integer indices")

    allowed = {"start", "stop", "step", "count"}
    unknown = sorted(set(selection) - allowed)
    if unknown:
        raise ValueError(f"samples contains unknown keys: {', '.join(unknown)}")

    start = _non_negative_int(selection.get("start", 0), key="samples.start")
    stop_value = selection.get("stop")
    if stop_value is None:
        stop = dataset_size
    else:
        stop = _non_negative_int(stop_value, key="samples.stop")
    step = _positive_int(selection.get("step", 1), key="samples.step")

    if start >= dataset_size:
        raise IndexError(
            f"samples.start {start} out of range for dataset of length {dataset_size}"
        )

    indices = tuple(range(start, min(stop, dataset_size), step))
    if "count" in selection:
        count = _positive_int(selection["count"], key="samples.count")
        indices = indices[:count]

    if not indices:
        raise ValueError("samples must select at least one dataset entry")
    return indices


def _select_configured_samples(config: dict, dataset, logger):
    """Apply optional top-level sample selection from the dataset config."""
    has_sample = "sample" in config
    has_samples = "samples" in config
    if has_sample and has_samples:
        raise ValueError("Use either sample or samples, not both")
    if not has_sample and not has_samples:
        return dataset

    dataset_size = len(dataset)
    if has_sample:
        sample_index = _validate_sample_index(
            config["sample"],
            key="sample",
            dataset_size=dataset_size,
        )
        sample = dataset[sample_index]
        logger.info(
            "Sample selection: using sample=%d of %d (id=%s, full_id=%s)",
            sample_index,
            dataset_size,
            sample.get("id"),
            sample.get("full_id"),
        )
        return [sample]

    indices = _resolve_sample_indices(config["samples"], dataset_size=dataset_size)
    logger.info(
        "Sample selection: using %d/%d samples (first_index=%d, last_index=%d)",
        len(indices),
        dataset_size,
        indices[0],
        indices[-1],
    )
    return _SelectedSamples(dataset, indices)


def _run_transform(args: argparse.Namespace, transform_class: type) -> int:
    """Shared logic for all subcommands."""
    logger = get_logger()

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Resolve the transform-specific config path.
    # Support both ``transform_config_path`` and legacy ``fog_config_path``.
    transform_config_key = "transform_config_path"
    if transform_config_key not in config:
        transform_config_key = "fog_config_path"
    transform_config_path = _resolve(config[transform_config_key], config_dir)
    # Read the transform config to determine the device (for dataset logging)
    with open(transform_config_path, "r", encoding="utf-8") as f:
        transform_cfg = json.load(f)
    device = transform_cfg.get("device", "cpu").lower()
    use_gpu = device not in ("cpu",)

    logger.info("Config: %s", args.config)
    logger.info("Transform config: %s", transform_config_path)

    required_modalities = transform_class.REQUIRED_MODALITIES
    required_hierarchical = transform_class.REQUIRED_HIERARCHICAL_MODALITIES or None
    dataset = build_dataset(config, required_modalities, required_hierarchical)
    output_backends = prepare_output_backends(config, dataset, transform_class)
    primary_backend = next(iter(output_backends.values()))
    samples = _select_configured_samples(config, dataset, logger)
    dataset_name = config.get("dataset", "dataset")

    raw_modalities = {
        **config.get("modalities", {}),
        **config.get("hierarchical_modalities", {}),
    }
    modality_info: dict[str, dict] = {}
    for name, entry in raw_modalities.items():
        if isinstance(entry, str):
            modality_info[name] = {"path": entry}
        else:
            modality_info[name] = entry
    log_dataset_info(logger, dataset_name, len(samples), modality_info, use_gpu)
    for slot, backend in output_backends.items():
        logger.info("Output path [%s]: %s", slot, backend.root)

    transform_kwargs: dict = {
        "config_path": str(transform_config_path),
        "out_path": str(primary_backend.root),
    }
    init_params = inspect.signature(transform_class.__init__).parameters
    if "output_backends" in init_params:
        transform_kwargs["output_backends"] = output_backends
    else:
        transform_kwargs["output_backend"] = primary_backend
        if len(output_backends) > 1:
            extra = [s for s in output_backends if s != next(iter(output_backends))]
            logger.warning(
                "%s does not accept output_backends; ignoring auxiliary slots: %s",
                transform_class.__name__,
                extra,
            )
    if "strict" in init_params:
        transform_kwargs["strict"] = bool(getattr(args, "strict", False))
    elif getattr(args, "strict", False):
        logger.warning(
            "--strict was set but %s does not support strict mode; ignoring.",
            transform_class.__name__,
        )
    transform = transform_class(**transform_kwargs)

    saved_paths = transform.run(samples)

    logger.info("Transform complete. Generated %d outputs.", len(saved_paths))
    return 0


def _cmd_fog(args: argparse.Namespace) -> int:
    from euler_preprocess.fog.transform import FogTransform
    return _run_transform(args, FogTransform)


def _cmd_sky_depth(args: argparse.Namespace) -> int:
    from euler_preprocess.sky_depth.transform import SkyDepthTransform
    return _run_transform(args, SkyDepthTransform)


def _cmd_radial(args: argparse.Namespace) -> int:
    from euler_preprocess.radial.transform import RadialTransform
    return _run_transform(args, RadialTransform)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocessing transforms for RGB+depth datasets.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Shared flags available to every subcommand.  Currently only honoured by
    # subcommands whose transforms accept a ``strict`` argument (see
    # ``_run_transform``); others log a warning and ignore it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to the dataset configuration JSON file.",
    )
    common.add_argument(
        "--strict", action="store_true",
        help="Abort early if leading-sample sanity checks detect suspicious "
             "inputs (e.g. 0%% or ~100%% sky pixels, wrong mask shape). "
             "Currently honoured by: sky-depth.",
    )

    fog_parser = subparsers.add_parser(
        "fog", parents=[common], help="Apply synthetic fog to RGB images.",
    )
    fog_parser.set_defaults(func=_cmd_fog)

    sky_depth_parser = subparsers.add_parser(
        "sky-depth", parents=[common], help="Override sky-region depth values.",
    )
    sky_depth_parser.set_defaults(func=_cmd_sky_depth)

    radial_parser = subparsers.add_parser(
        "radial", parents=[common],
        help="Convert planar (z-buffer) depth to radial (Euclidean) depth.",
    )
    radial_parser.set_defaults(func=_cmd_radial)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return parser.parse_args(["--help"])
    return args


def main() -> int:
    args = parse_args()
    return args.func(args)
