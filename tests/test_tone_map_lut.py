from __future__ import annotations

import numpy as np
import pytest

from euler_preprocess.fog.capture import (
    CaptureArtifactPipeline,
    CaptureContext,
    _apply_tone_map_np,
)

try:
    import torch

    from euler_preprocess.fog.capture import _apply_tone_map_torch
except ImportError:
    torch = None

# A camera-response curve whose toe compresses shadows ~11x, matching the
# profiles in configs/dense_gloomy_daylight_fog_camera.json.
TOE_LUT = [
    0.0, 0.006, 0.014, 0.028, 0.052, 0.090, 0.145, 0.220,
    0.320, 0.450, 0.610, 0.780, 0.900, 0.965, 0.995, 1.0,
]
LUT_CONFIG = {"tone_map_lut": TOE_LUT, "tone_map_lut_domain": "linear"}
# Shadow values below the LUT's linear toe, where a linear strength blend used
# to extrapolate through zero at strength ~= 1/(1 - 0.09) = 1.0989.
SHADOWS = np.asarray(
    [[[0.005, 0.005, 0.005], [0.0145, 0.0145, 0.0145], [0.03, 0.03, 0.03]]],
    dtype=np.float32,
)


def _isp_pipeline(**overrides):
    stage = {
        "type": "isp",
        "denoise_sigma": 0.0,
        "color_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "tone_map": "lut",
        "tone_map_strength": 1.0,
        "tone_map_lut": TOE_LUT,
        "gamma": "linear",
        "local_contrast_strength": 0.0,
        "sharpen_amount": 0.0,
        "saturation": 1.0,
    }
    stage.update(overrides)
    return CaptureArtifactPipeline.from_config({"capture": {"stages": [stage]}})


def test_isp_tone_map_lut_interpolates_1d_curve() -> None:
    pipeline = _isp_pipeline(tone_map_lut=[0.0, 0.25, 1.0])
    image = np.asarray([[[0.0, 0.5, 1.0]]], dtype=np.float32)

    result = pipeline.apply_np(image, CaptureContext(rng=np.random.default_rng(9)))

    np.testing.assert_allclose(result, [[[0.0, 0.25, 1.0]]], atol=1e-6)


def test_tone_map_strength_zero_is_identity() -> None:
    result = _apply_tone_map_np(SHADOWS, "lut", 0.0, LUT_CONFIG)

    np.testing.assert_allclose(result, SHADOWS, atol=1e-7)


def test_tone_map_strength_one_reproduces_curve_exactly() -> None:
    result = _apply_tone_map_np(SHADOWS, "lut", 1.0, LUT_CONFIG)

    expected = np.interp(
        SHADOWS,
        np.linspace(0.0, 1.0, len(TOE_LUT)),
        np.asarray(TOE_LUT, dtype=np.float32),
    )
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("strength", [1.05, 1.0989, 1.119, 1.22, 2.0, 8.0])
def test_tone_map_strength_above_one_never_goes_negative(strength: float) -> None:
    """A linear blend crossed zero here and clipped whole frames to black."""
    result = _apply_tone_map_np(SHADOWS, "lut", strength, LUT_CONFIG)

    assert np.isfinite(result).all()
    assert float(result.min()) > 0.0


def test_isp_does_not_black_out_shadows_at_strength_above_one() -> None:
    """End-to-end guard: the ISP must not crush a dark frame to zero."""
    pipeline = _isp_pipeline(tone_map_strength=1.22, gamma="srgb")

    result = pipeline.apply_np(SHADOWS, CaptureContext(rng=np.random.default_rng(9)))

    assert float(result.min()) > 0.0
    assert float(np.median(result)) > 1e-4


def test_tone_map_strength_is_monotonic_and_bounded() -> None:
    previous = None
    for strength in np.linspace(0.0, 4.0, 41):
        result = _apply_tone_map_np(SHADOWS, "lut", float(strength), LUT_CONFIG)
        assert np.isfinite(result).all()
        assert float(result.min()) >= 0.0
        if previous is not None:
            # A compressive curve only darkens further as strength rises.
            assert (result <= previous + 1e-6).all()
        previous = result


def test_tone_map_preserves_black_and_survives_extreme_strength() -> None:
    image = np.zeros((1, 1, 3), dtype=np.float32)
    assert float(_apply_tone_map_np(image, "lut", 3.0, LUT_CONFIG).max()) == 0.0

    result = _apply_tone_map_np(SHADOWS, "lut", 1e6, LUT_CONFIG)
    assert np.isfinite(result).all()
    assert float(result.min()) >= 0.0


@pytest.mark.parametrize("strength", [0.0, 1.0, 1.3, 5.0])
def test_aces_tone_map_stays_non_negative(strength: float) -> None:
    result = _apply_tone_map_np(SHADOWS, "aces", strength, None)

    assert np.isfinite(result).all()
    assert float(result.min()) >= 0.0


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("mode", ["lut", "aces"])
@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0, 1.119, 2.0])
def test_torch_tone_map_matches_numpy(mode: str, strength: float) -> None:
    config = LUT_CONFIG if mode == "lut" else None
    expected = _apply_tone_map_np(SHADOWS, mode, strength, config)

    actual = _apply_tone_map_torch(
        torch.from_numpy(SHADOWS.copy()),
        mode,
        strength,
        config,
    )

    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6)
