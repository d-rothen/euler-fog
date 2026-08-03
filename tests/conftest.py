"""Pytest configuration and shared fixtures for fog generation tests."""

import numpy as np
import pytest


@pytest.fixture
def synthetic_samples():
    """Three synthetic transform samples.

    Each is a dict of the expected shape:
        {"rgb": ndarray, "depth": ndarray, "semantic_segmentation": ndarray, "id": str}
    """
    rng = np.random.default_rng(42)
    samples = []
    for i in range(3):
        h, w = 100, 100
        rgb = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        depth = rng.uniform(1.0, 50.0, (h, w)).astype(np.float32)
        sky_mask = np.zeros((h, w), dtype=bool)
        sky_mask[:30, :] = True
        samples.append(
            {
                "rgb": rgb,
                "depth": depth,
                "semantic_segmentation": sky_mask,
                "id": f"sample_{i:05d}",
            }
        )
    return samples
