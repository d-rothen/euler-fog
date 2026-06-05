from __future__ import annotations

import numpy as np

from euler_preprocess.fog.models import apply_scene_illumination_np


def test_disabled_scene_illumination_leaves_rgb_unchanged() -> None:
    rgb = np.random.default_rng(1).random((4, 5, 3), dtype=np.float32)
    depth = np.full((4, 5), 12.0, dtype=np.float32)

    out, ev_map = apply_scene_illumination_np(
        rgb,
        depth,
        0.03,
        {"scene_illumination": {"enabled": False}},
        np.random.default_rng(2),
    )

    np.testing.assert_allclose(out, rgb)
    np.testing.assert_allclose(ev_map, 0.0)


def test_scene_illumination_darkens_near_pixels_more_than_far_pixels() -> None:
    rgb = np.ones((1, 2, 3), dtype=np.float32)
    depth = np.asarray([[2.0, 40.0]], dtype=np.float32)

    out, ev_map = apply_scene_illumination_np(
        rgb,
        depth,
        0.04,
        {
            "scene_illumination": {
                "enabled": True,
                "global_ev": 0.0,
                "near_ev": 1.2,
                "near_decay_depth_m": 10.0,
                "fog_coupled_ev": 0.0,
                "min_radiance_scale": 0.0,
            }
        },
        np.random.default_rng(3),
    )

    assert ev_map[0, 0] > ev_map[0, 1]
    assert float(out[0, 0].mean()) < float(out[0, 1].mean())


def test_scene_illumination_preserves_sky_when_sky_weight_is_zero() -> None:
    rgb = np.full((2, 3, 3), 0.7, dtype=np.float32)
    depth = np.full((2, 3), 5.0, dtype=np.float32)
    sky_mask = np.zeros((2, 3), dtype=bool)
    sky_mask[0, :] = True

    out, ev_map = apply_scene_illumination_np(
        rgb,
        depth,
        0.08,
        {
            "scene_illumination": {
                "enabled": True,
                "global_ev": 0.8,
                "near_ev": 0.6,
                "sky_weight": 0.0,
            }
        },
        np.random.default_rng(4),
        sky_mask=sky_mask,
    )

    np.testing.assert_allclose(out[0], rgb[0])
    np.testing.assert_allclose(ev_map[0], 0.0)
    assert float(out[1].mean()) < float(rgb[1].mean())


def test_scene_illumination_introduces_no_nans_for_invalid_depth() -> None:
    rgb = np.full((2, 2, 3), 0.5, dtype=np.float32)
    depth = np.asarray([[np.nan, np.inf], [-np.inf, -3.0]], dtype=np.float32)

    out, ev_map = apply_scene_illumination_np(
        rgb,
        depth,
        0.05,
        {
            "scene_illumination": {
                "enabled": True,
                "global_ev": 0.2,
                "near_ev": 0.5,
                "fog_coupled_ev": 0.3,
            }
        },
        np.random.default_rng(5),
    )

    assert np.isfinite(out).all()
    assert np.isfinite(ev_map).all()
