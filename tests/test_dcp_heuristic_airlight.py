from __future__ import annotations

import json

import numpy as np
import pytest

from euler_preprocess.fog.dcp_heuristic_airlight import DCPHeuristicAirlight
from euler_preprocess.fog.foggify import Foggify
from euler_preprocess.fog.models import apply_model

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from euler_preprocess.fog.dcp_heuristic_airlight_torch import (
        DCPHeuristicAirlightTorch,
    )


def _make_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = np.full((4, 4, 3), 0.05, dtype=np.float32)
    sky_color = np.array([0.90, 0.94, 1.00], dtype=np.float32)
    median_pixel = np.array([0.76, 0.76, 0.78], dtype=np.float32)

    image[0, 0] = sky_color
    image[0, 1] = sky_color
    image[1, 0] = [0.72, 0.72, 0.74]
    image[1, 1] = [0.74, 0.74, 0.74]
    image[1, 2] = median_pixel

    sky_mask = np.zeros((4, 4), dtype=bool)
    sky_mask[0, 0] = True
    sky_mask[0, 1] = True
    return image, sky_mask, sky_color, median_pixel


def _make_warm_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.full((4, 4, 3), 0.05, dtype=np.float32)
    sky_color = np.array([0.92, 0.82, 0.72], dtype=np.float32)
    image[0, 0] = sky_color
    image[0, 1] = sky_color
    image[1, 0] = [0.72, 0.70, 0.68]
    image[1, 1] = [0.74, 0.71, 0.69]
    image[1, 2] = [0.77, 0.74, 0.72]

    sky_mask = np.zeros((4, 4), dtype=bool)
    sky_mask[0, 0] = True
    sky_mask[0, 1] = True
    return image, sky_mask, sky_color


class TestDCPHeuristicAirlight:
    def test_compute_pools_bright_candidates_instead_of_single_median_pixel(self):
        image, _, sky_color, median_pixel = _make_scene()
        estimator = DCPHeuristicAirlight(patch_size=1, top_percent=0.3)

        result = estimator.compute(image)

        assert estimator._luminance(result) > estimator._luminance(median_pixel)
        assert estimator._luminance(result) < estimator._luminance(sky_color)
        assert not np.allclose(result, median_pixel)

    def test_estimate_airlight_uses_bright_sky_chromaticity_when_available(self):
        image, sky_mask, sky_color, _ = _make_scene()
        estimator = DCPHeuristicAirlight(patch_size=1, top_percent=0.3)

        result = estimator.estimate_airlight(image, sky_mask)

        np.testing.assert_allclose(result, sky_color, atol=1e-6)

    def test_bias_controls_push_airlight_toward_white_and_cool(self):
        image, sky_mask, sky_color = _make_warm_scene()
        estimator = DCPHeuristicAirlight(
            patch_size=1,
            top_percent=0.3,
            white_bias=0.2,
            cool_bias=0.3,
        )

        result = estimator.estimate_airlight(image, sky_mask)

        assert estimator._luminance(result) == pytest.approx(
            estimator._luminance(sky_color),
            abs=1e-6,
        )
        assert (result[0] - result[2]) < (sky_color[0] - sky_color[2])
        assert result[2] > sky_color[2]

    def test_cool_target_is_derived_from_sky_color(self):
        estimator = DCPHeuristicAirlight(
            white_bias=0.0,
            cool_bias=0.3,
            cool_target=np.array([0.9, 0.96, 1.0], dtype=np.float32),
        )
        sky_color = np.array([0.8, 0.7, 0.6], dtype=np.float32)

        derived = estimator._correlated_cool_target(sky_color)

        np.testing.assert_allclose(
            derived,
            [0.85, 0.83, 0.8],
            atol=1e-6,
        )

    def test_foggify_reads_dcp_heuristic_bias_config(self, tmp_path):
        cfg = {
            "airlight": "dcp_heuristic",
            "device": "cpu",
            "seed": 7,
            "dcp_heuristic": {
                "white_bias": 0.2,
                "cool_bias": 0.3,
                "cool_target": [0.9, 0.96, 1.0],
            },
            "models": {
                "uniform": {
                    "visibility_m": {"dist": "constant", "value": 150.0},
                    "atmospheric_light": "from_sky",
                },
            },
            "selection": {"mode": "fixed", "model": "uniform"},
        }
        cfg_path = tmp_path / "fog_config.json"
        cfg_path.write_text(json.dumps(cfg))

        foggify = Foggify(config_path=str(cfg_path), out_path=str(tmp_path / "out"))

        assert foggify.airlight_estimator.white_bias == pytest.approx(0.2)
        assert foggify.airlight_estimator.cool_bias == pytest.approx(0.3)
        np.testing.assert_allclose(
            foggify.airlight_estimator.cool_target,
            [0.9, 0.96, 1.0],
            atol=1e-6,
        )

    def test_apply_model_accepts_dcp_heuristic_atmospheric_light_alias(self):
        rng = np.random.default_rng(0)
        rgb = np.full((2, 2, 3), 0.4, dtype=np.float32)
        depth = np.ones((2, 2), dtype=np.float32)
        estimated_airlight = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        model_cfg = {
            "visibility_m": {"dist": "constant", "value": 120.0},
            "atmospheric_light": "dcp_heuristic",
            "airlight_dampening": {"enabled": False},
        }

        _, _, airlight, _, _ = apply_model(
            rgb,
            depth,
            "uniform",
            model_cfg,
            rng,
            0.05,
            estimated_airlight,
        )

        np.testing.assert_allclose(airlight, estimated_airlight, atol=1e-6)


@pytest.mark.skipif(torch is None, reason="torch not installed")
class TestDCPHeuristicAirlightTorch:
    def test_estimate_airlight_matches_cpu_sky_guided_result(self):
        image, sky_mask, sky_color, _ = _make_scene()
        estimator = DCPHeuristicAirlightTorch(patch_size=1, top_percent=0.3)

        result = estimator.estimate_airlight(
            torch.from_numpy(image),
            torch.from_numpy(sky_mask),
        )

        torch.testing.assert_close(
            result,
            torch.from_numpy(sky_color),
            atol=1e-6,
            rtol=0,
        )

    def test_bias_controls_match_cpu_behavior(self):
        image, sky_mask, sky_color = _make_warm_scene()
        estimator = DCPHeuristicAirlightTorch(
            patch_size=1,
            top_percent=0.3,
            white_bias=0.2,
            cool_bias=0.3,
        )

        result = estimator.estimate_airlight(
            torch.from_numpy(image),
            torch.from_numpy(sky_mask),
        )

        weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32)
        sky_luminance = float((torch.from_numpy(sky_color) * weights).sum().item())
        result_luminance = float((result * weights).sum().item())

        assert result_luminance == pytest.approx(sky_luminance, abs=1e-6)
        assert float(result[2].item()) > float(sky_color[2])

    def test_cool_target_is_derived_from_sky_color(self):
        estimator = DCPHeuristicAirlightTorch(
            white_bias=0.0,
            cool_bias=0.3,
            cool_target=torch.tensor([0.9, 0.96, 1.0]),
        )
        sky_color = torch.tensor([0.8, 0.7, 0.6], dtype=torch.float32)

        derived = estimator._correlated_cool_target(sky_color)

        torch.testing.assert_close(
            derived,
            torch.tensor([0.85, 0.83, 0.8], dtype=torch.float32),
            atol=1e-6,
            rtol=0,
        )

    def test_apply_model_torch_accepts_dcp_heuristic_atmospheric_light_alias(
        self,
        tmp_path,
    ):
        cfg = {
            "airlight": "dcp_heuristic",
            "device": "cpu",
            "seed": 7,
            "models": {
                "uniform": {
                    "visibility_m": {"dist": "constant", "value": 120.0},
                    "atmospheric_light": "dcp_heuristic",
                    "airlight_dampening": {"enabled": False},
                },
            },
            "selection": {"mode": "fixed", "model": "uniform"},
        }
        cfg_path = tmp_path / "fog_config.json"
        cfg_path.write_text(json.dumps(cfg))
        foggify = Foggify(config_path=str(cfg_path), out_path=str(tmp_path / "out"))

        rgb_t = torch.full((2, 2, 3), 0.4, dtype=torch.float32)
        depth_t = torch.ones((2, 2), dtype=torch.float32)
        estimated_airlight = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32)

        _, _, airlight, _, _ = foggify._apply_model_torch(
            rgb_t,
            depth_t,
            "uniform",
            cfg["models"]["uniform"],
            np.random.default_rng(0),
            estimated_airlight,
            torch.Generator(),
        )

        torch.testing.assert_close(
            airlight,
            estimated_airlight,
            atol=1e-6,
            rtol=0,
        )
