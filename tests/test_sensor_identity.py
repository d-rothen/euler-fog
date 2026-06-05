from __future__ import annotations

import numpy as np

from euler_preprocess.fog.capture import (
    CaptureArtifactPipeline,
    CaptureContext,
    SensorStage,
)


def _identity_config(sensor_id: str = "cam_a", seed: int = 17) -> dict:
    return {
        "sensor_identity": {
            "enabled": True,
            "sensor_id": sensor_id,
            "seed": seed,
            "prnu_sigma": 0.01,
            "dsnu_sigma": 0.0002,
            "persistent_hot_pixel_probability": 0.04,
            "persistent_dead_pixel_probability": 0.03,
            "persistent_row_sigma": 0.0003,
            "persistent_column_sigma": 0.0002,
        }
    }


def test_persistent_sensor_maps_are_deterministic_and_cached() -> None:
    stage = SensorStage({})
    maps_a = stage._sensor_identity_maps_np((18, 22), "RGGB", _identity_config())
    maps_b = stage._sensor_identity_maps_np((18, 22), "RGGB", _identity_config())

    assert maps_a is maps_b
    for key in ("prnu", "dsnu", "row_bias", "column_bias", "hot_mask", "dead_mask"):
        np.testing.assert_array_equal(maps_a[key], maps_b[key])


def test_persistent_sensor_maps_change_with_sensor_id_or_seed() -> None:
    stage = SensorStage({})
    maps_a = stage._sensor_identity_maps_np((18, 22), "RGGB", _identity_config())
    maps_b = stage._sensor_identity_maps_np(
        (18, 22),
        "RGGB",
        _identity_config(sensor_id="cam_b"),
    )
    maps_c = stage._sensor_identity_maps_np(
        (18, 22),
        "RGGB",
        _identity_config(seed=18),
    )

    assert not np.array_equal(maps_a["prnu"], maps_b["prnu"])
    assert not np.array_equal(maps_a["prnu"], maps_c["prnu"])


def test_persistent_bad_pixels_remain_stable_across_images() -> None:
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
                        "demosaic": False,
                        "sensor_identity": {
                            "enabled": True,
                            "sensor_id": "stable_bad_pixels",
                            "seed": 41,
                            "persistent_hot_pixel_probability": 0.18,
                            "persistent_dead_pixel_probability": 0.16,
                        },
                    }
                ]
            }
        }
    )
    image = np.full((24, 24, 3), 0.5, dtype=np.float32)

    first = pipeline.apply_np(image, CaptureContext(rng=np.random.default_rng(1)))
    second = pipeline.apply_np(image, CaptureContext(rng=np.random.default_rng(2)))

    np.testing.assert_array_equal(first, second)
    assert np.count_nonzero(first == 0.0) > 0
    assert np.count_nonzero(first == 1.0) > 0
