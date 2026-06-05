from __future__ import annotations

import numpy as np

import euler_preprocess.fog.capture as capture_module
from euler_preprocess.fog.capture import CaptureContext


def test_sky_suppression_reduces_sky_influence_on_metering() -> None:
    image = np.full((8, 8, 3), 0.1, dtype=np.float32)
    image[:4, :, :] = 0.9
    sky_mask = np.zeros((8, 8), dtype=bool)
    sky_mask[:4, :] = True
    context = CaptureContext(attributes={"sky_mask": sky_mask})

    base = capture_module._auto_exposure_metrics_np(
        image,
        context,
        {"metering": "mean", "sky_suppression": 0.0},
        np.random.default_rng(1),
    )
    suppressed = capture_module._auto_exposure_metrics_np(
        image,
        context,
        {"metering": "mean", "sky_suppression": 0.9},
        np.random.default_rng(1),
    )

    assert suppressed["meter_luminance"] < base["meter_luminance"]


def test_fog_suppression_reduces_dense_far_field_metering() -> None:
    image = np.full((8, 8, 3), 0.1, dtype=np.float32)
    image[:, 4:, :] = 0.85
    depth = np.full((8, 8), 4.0, dtype=np.float32)
    depth[:, 4:] = 80.0
    k_map = np.full((8, 8), 0.08, dtype=np.float32)
    context = CaptureContext(depth_m=depth, k_map=k_map)

    base = capture_module._auto_exposure_metrics_np(
        image,
        context,
        {"metering": "mean", "fog_meter_suppression": 0.0},
        np.random.default_rng(2),
    )
    suppressed = capture_module._auto_exposure_metrics_np(
        image,
        context,
        {"metering": "mean", "fog_meter_suppression": 0.9},
        np.random.default_rng(2),
    )

    assert suppressed["meter_luminance"] < base["meter_luminance"]


def test_legacy_metering_ignores_context_maps_without_new_options() -> None:
    image = np.linspace(0.02, 0.9, 64, dtype=np.float32).reshape(8, 8, 1)
    image = np.repeat(image, 3, axis=-1)
    depth = np.linspace(1.0, 80.0, 64, dtype=np.float32).reshape(8, 8)
    sky_mask = np.zeros((8, 8), dtype=bool)
    sky_mask[:2, :] = True
    config = {
        "metering": "center_weighted",
        "center_sigma": 0.4,
        "center_weight": 0.55,
        "meter_percentile": 50.0,
        "highlight_percentile": 98.5,
    }

    empty_context = CaptureContext()
    rich_context = CaptureContext(
        depth_m=depth,
        k_map=np.full((8, 8), 0.08, dtype=np.float32),
        attributes={"sky_mask": sky_mask},
    )
    legacy = capture_module._auto_exposure_metrics_np(
        image,
        empty_context,
        config,
        np.random.default_rng(3),
    )
    with_maps = capture_module._auto_exposure_metrics_np(
        image,
        rich_context,
        config,
        np.random.default_rng(3),
    )

    np.testing.assert_allclose(
        with_maps["meter_luminance"],
        legacy["meter_luminance"],
    )
    np.testing.assert_allclose(
        with_maps["highlight_luminance"],
        legacy["highlight_luminance"],
    )
