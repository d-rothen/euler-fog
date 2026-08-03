"""Integration test for the fog pipeline against a real euler-loading dataset.

The dataset is not bundled. Point the environment at one to run this test::

    EULER_PREPROCESS_RGB_PATH=/data/vkitti_2.0.3_rgb \
    EULER_PREPROCESS_DEPTH_PATH=/data/vkitti_2.0.3_depth \
    EULER_PREPROCESS_SEGMENTATION_PATH=/data/vkitti_2.0.3_classSegmentation \
    pytest tests/test_foggify_integration.py -m integration
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from euler_loading import Modality, MultiModalDataset
from euler_preprocess.fog.transform import FogTransform

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FOG_CONFIG_PATH = PROJECT_ROOT / "configs" / "fog_config.json"
MAX_SAMPLES = 5


def _modality_paths() -> dict[str, str]:
    env_keys = {
        "rgb": "EULER_PREPROCESS_RGB_PATH",
        "depth": "EULER_PREPROCESS_DEPTH_PATH",
        "semantic_segmentation": "EULER_PREPROCESS_SEGMENTATION_PATH",
    }
    paths = {name: os.environ.get(key, "") for name, key in env_keys.items()}
    missing = [env_keys[name] for name, path in paths.items() if not path]
    if missing:
        pytest.skip(f"Dataset not configured; set {', '.join(missing)}")
    return paths


def _cpu_fog_config(dst: Path) -> Path:
    """Copy the example fog config, forcing device to 'cpu' for portability."""
    cfg = json.loads(FOG_CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["device"] = "cpu"
    dst.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return dst


@pytest.mark.integration
def test_fog_transform_with_real_data(tmp_path):
    paths = _modality_paths()
    dataset = MultiModalDataset(
        modalities={name: Modality(path) for name, path in paths.items()},
    )
    assert len(dataset) > 0, f"No matching files found across modalities: {paths}"

    n_samples = min(MAX_SAMPLES, len(dataset))
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    transform = FogTransform(
        config_path=str(_cpu_fog_config(tmp_path / "fog_config_cpu.json")),
        out_path=str(out_dir),
    )
    saved_paths = transform.run(dataset[i] for i in range(n_samples))

    assert len(saved_paths) == n_samples

    for path in saved_paths:
        path = Path(path)
        assert path.exists(), f"Output file missing: {path}"
        assert path.suffix == ".png"

        img = np.asarray(Image.open(path))
        assert img.ndim == 3 and img.shape[2] == 3, f"Expected RGB, got {img.shape}"
        assert img.dtype == np.uint8

    model_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert model_dirs, "No model output directories created"
    for model_dir in model_dirs:
        cfg_file = model_dir / "config.json"
        assert cfg_file.exists(), f"Missing config.json in {model_dir}"
        assert "size" in json.loads(cfg_file.read_text(encoding="utf-8"))
