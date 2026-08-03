"""Unit tests for airlight estimation fallback when no sky pixels are present."""

import json
import logging

import numpy as np
import pytest

from euler_preprocess.common.sampling import format_value
from euler_preprocess.fog.airlight_from_sky import AirlightFromSky, DEFAULT_AIRLIGHT_FALLBACK
from euler_preprocess.fog.models import normalize_atmospheric_light
from euler_preprocess.fog.transform import FogTransform

try:
    import torch
    from euler_preprocess.fog.models import estimate_airlight_torch
except ImportError:
    torch = None

FOG_LOGGER = "euler-preprocess.fog"


class TestAirlightFromSkyFallback:
    def _make_estimator(self):
        return AirlightFromSky(sky_depth_threshold=0.0)

    def test_empty_sky_mask_returns_fallback(self):
        estimator = self._make_estimator()
        image = np.random.rand(64, 64, 3).astype(np.float32)
        sky_mask = np.zeros((64, 64), dtype=bool)  # no sky

        result = estimator.estimate_airlight(image, sky_mask)

        np.testing.assert_array_equal(result, DEFAULT_AIRLIGHT_FALLBACK)
        assert np.all(np.isfinite(result))

    def test_empty_sky_mask_logs_warning(self, caplog):
        estimator = self._make_estimator()
        image = np.random.rand(32, 32, 3).astype(np.float32)
        sky_mask = np.zeros((32, 32), dtype=bool)

        with caplog.at_level(logging.WARNING, logger=FOG_LOGGER):
            estimator.estimate_airlight(image, sky_mask, sample_id="test_001")

        assert any("No sky pixels" in msg for msg in caplog.messages)
        assert any("test_001" in msg for msg in caplog.messages)

    def test_empty_sky_mask_logs_without_sample_id(self, caplog):
        estimator = self._make_estimator()
        image = np.random.rand(32, 32, 3).astype(np.float32)
        sky_mask = np.zeros((32, 32), dtype=bool)

        with caplog.at_level(logging.WARNING, logger=FOG_LOGGER):
            estimator.estimate_airlight(image, sky_mask)

        assert any("No sky pixels" in msg for msg in caplog.messages)

    def test_normal_sky_mask_returns_mean(self):
        estimator = self._make_estimator()
        # Create image where sky pixels are known color
        image = np.zeros((64, 64, 3), dtype=np.float32)
        sky_mask = np.zeros((64, 64), dtype=bool)
        sky_mask[0:10, :] = True
        image[0:10, :] = [0.5, 0.6, 0.7]

        result = estimator.estimate_airlight(image, sky_mask)

        np.testing.assert_allclose(result, [0.5, 0.6, 0.7], atol=1e-4)

    def test_nan_sky_pixels_returns_fallback(self):
        estimator = self._make_estimator()
        image = np.full((32, 32, 3), np.nan, dtype=np.float32)
        sky_mask = np.ones((32, 32), dtype=bool)

        result = estimator.estimate_airlight(image, sky_mask, sample_id="nan_test")

        np.testing.assert_array_equal(result, DEFAULT_AIRLIGHT_FALLBACK)
        assert np.all(np.isfinite(result))


@pytest.mark.skipif(torch is None, reason="torch not installed")
class TestEstimateAirlightTorchFallback:
    def test_empty_sky_mask_returns_fallback(self):
        image = torch.rand(64, 64, 3)
        sky_mask = torch.zeros(64, 64, dtype=torch.bool)

        result = estimate_airlight_torch(image, sky_mask)

        assert torch.all(result == 1.0)
        assert torch.all(torch.isfinite(result))
        assert result.shape == (3,)

    def test_empty_sky_mask_logs_warning(self, caplog):
        image = torch.rand(32, 32, 3)
        sky_mask = torch.zeros(32, 32, dtype=torch.bool)

        with caplog.at_level(logging.WARNING, logger=FOG_LOGGER):
            estimate_airlight_torch(image, sky_mask, sample_id="torch_001")

        assert any("No sky pixels" in msg for msg in caplog.messages)
        assert any("torch_001" in msg for msg in caplog.messages)

    def test_normal_sky_mask_returns_mean(self):
        image = torch.zeros(32, 32, 3)
        sky_mask = torch.zeros(32, 32, dtype=torch.bool)
        sky_mask[0:5, :] = True
        image[0:5, :] = torch.tensor([0.3, 0.4, 0.5])

        result = estimate_airlight_torch(image, sky_mask)

        torch.testing.assert_close(result, torch.tensor([0.3, 0.4, 0.5]), atol=1e-6, rtol=0)


class TestNormalizeAtmosphericLightNaN:
    def test_nan_input_clipped_to_bounds(self):
        """NaN fed to normalize_atmospheric_light should be caught upstream,
        but verify clip behavior so it doesn't silently pass through."""
        value = np.array([np.nan, 0.5, 0.5], dtype=np.float32)
        result = normalize_atmospheric_light(value)
        # np.clip with NaN still returns NaN – this confirms the fix must
        # happen *before* normalize_atmospheric_light is called.
        assert np.isnan(result[0])


class TestFormatValueNaN:
    def test_nan_formats_as_nan_string(self):
        """Confirm NaN produces 'nan' so filenames are identifiable."""
        assert format_value(float("nan")) == "nan"

    def test_normal_value(self):
        assert format_value(0.017) == "0.017"
        assert format_value(1.0) == "1"


class TestFogTransformEmptySkyMask:
    @pytest.fixture
    def fog_config(self, tmp_path):
        cfg = {
            "airlight": "from_sky",
            "device": "cpu",
            "models": {
                "uniform": {
                    "visibility_m": {"dist": "constant", "value": 200.0},
                    "atmospheric_light": "from_sky",
                },
            },
            "selection": {"mode": "fixed", "model": "uniform"},
            "seed": 42,
        }
        path = tmp_path / "fog_cfg.json"
        path.write_text(json.dumps(cfg))
        return path

    def test_no_sky_produces_valid_output(self, tmp_path, fog_config, caplog):
        out_dir = tmp_path / "output"
        transform = FogTransform(config_path=str(fog_config), out_path=str(out_dir))

        image = np.random.rand(64, 64, 3).astype(np.float32)
        depth = np.random.rand(64, 64).astype(np.float32) * 100.0
        sky_mask = np.zeros((64, 64), dtype=bool)  # no sky at all

        samples = [{"rgb": image, "depth": depth, "semantic_segmentation": sky_mask, "id": "nosky_001"}]

        with caplog.at_level(logging.WARNING, logger=FOG_LOGGER):
            paths = transform.run(samples)

        assert len(paths) == 1
        assert paths[0].exists()
        assert "nan" not in paths[0].name.lower()
        assert any("No sky pixels" in msg for msg in caplog.messages)
        assert any("nosky_001" in msg for msg in caplog.messages)


class TestFogTransformOutputHierarchy:
    @pytest.fixture
    def fog_config(self, tmp_path):
        cfg = {
            "airlight": "from_sky",
            "device": "cpu",
            "models": {
                "uniform": {
                    "visibility_m": {"dist": "constant", "value": 200.0},
                    "atmospheric_light": "from_sky",
                },
            },
            "selection": {"mode": "fixed", "model": "uniform"},
            "seed": 42,
        }
        path = tmp_path / "fog_cfg.json"
        path.write_text(json.dumps(cfg))
        return path

    def _make_sample(self, sample_id, full_id=None):
        rng = np.random.default_rng(0)
        h, w = 64, 64
        sample = {
            "rgb": rng.random((h, w, 3)).astype(np.float32),
            "depth": rng.uniform(1.0, 50.0, (h, w)).astype(np.float32),
            "semantic_segmentation": np.ones((h, w), dtype=bool),
            "id": sample_id,
        }
        if full_id is not None:
            sample["full_id"] = full_id
        return sample

    def test_full_id_creates_subdirectories(self, tmp_path, fog_config):
        out_dir = tmp_path / "output"
        transform = FogTransform(config_path=str(fog_config), out_path=str(out_dir))

        samples = [
            self._make_sample("00042", full_id="/Scene02/30-deg-right/Camera_0/00042"),
        ]
        paths = transform.run(samples)

        assert len(paths) == 1
        # The output should be under model_name/Scene02/30-deg-right/Camera_0/
        rel = paths[0].relative_to(out_dir / "uniform")
        assert rel.parts[:-1] == ("Scene02", "30-deg-right", "Camera_0")
        assert paths[0].exists()

    def test_no_full_id_stays_flat(self, tmp_path, fog_config):
        out_dir = tmp_path / "output"
        transform = FogTransform(config_path=str(fog_config), out_path=str(out_dir))

        samples = [self._make_sample("00001")]
        paths = transform.run(samples)

        assert len(paths) == 1
        # Should be directly under model_name/ with no extra subdirs
        rel = paths[0].relative_to(out_dir / "uniform")
        assert len(rel.parts) == 1  # just the filename
        assert paths[0].exists()

    def test_single_segment_full_id_stays_flat(self, tmp_path, fog_config):
        """full_id with only one segment (just the frame id) should not add subdirs."""
        out_dir = tmp_path / "output"
        transform = FogTransform(config_path=str(fog_config), out_path=str(out_dir))

        samples = [self._make_sample("00005", full_id="/00005")]
        paths = transform.run(samples)

        assert len(paths) == 1
        rel = paths[0].relative_to(out_dir / "uniform")
        assert len(rel.parts) == 1
        assert paths[0].exists()

    def test_multiple_samples_different_hierarchies(self, tmp_path, fog_config):
        out_dir = tmp_path / "output"
        transform = FogTransform(config_path=str(fog_config), out_path=str(out_dir))

        samples = [
            self._make_sample("00001", full_id="/SceneA/clone/Camera_0/00001"),
            self._make_sample("00002", full_id="/SceneB/fog/Camera_1/00002"),
        ]
        paths = transform.run(samples)

        assert len(paths) == 2
        rel_a = paths[0].relative_to(out_dir / "uniform")
        rel_b = paths[1].relative_to(out_dir / "uniform")
        assert rel_a.parts[:-1] == ("SceneA", "clone", "Camera_0")
        assert rel_b.parts[:-1] == ("SceneB", "fog", "Camera_1")
