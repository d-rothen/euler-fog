"""Unit tests for SkyDepthTransform."""

import json

import numpy as np
import pytest

from euler_preprocess.sky_depth.transform import SkyDepthTransform


@pytest.fixture
def sky_depth_config(tmp_path):
    cfg = {"sky_depth_value": 500.0}
    cfg_path = tmp_path / "sky_depth.json"
    cfg_path.write_text(json.dumps(cfg))
    return cfg_path


@pytest.fixture
def samples():
    depth = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    sky_mask = np.array(
        [[True, True, False], [False, False, False], [False, False, True]], dtype=bool
    )
    return [
        {"id": "frame_001", "depth": depth.copy(), "semantic_segmentation": sky_mask.copy()},
    ]


class TestSkyDepthTransform:
    def test_replaces_sky_pixels(self, tmp_path, sky_depth_config, samples):
        out_path = tmp_path / "output"
        transform = SkyDepthTransform(str(sky_depth_config), str(out_path))
        paths = transform.run(samples)

        assert len(paths) == 1
        result = np.load(paths[0])
        assert result.dtype == np.float32

        # Sky pixels should be overridden
        assert result[0, 0] == 500.0
        assert result[0, 1] == 500.0
        assert result[2, 2] == 500.0

        # Non-sky pixels should be unchanged
        assert result[0, 2] == 3.0
        assert result[1, 0] == 4.0
        assert result[1, 1] == 5.0

    def test_default_sky_depth_value(self, tmp_path, samples):
        cfg = {}
        cfg_path = tmp_path / "default.json"
        cfg_path.write_text(json.dumps(cfg))
        out_path = tmp_path / "output"
        transform = SkyDepthTransform(str(cfg_path), str(out_path))
        paths = transform.run(samples)

        result = np.load(paths[0])
        assert result[0, 0] == 1000.0  # default value

    def test_no_sky_pixels(self, tmp_path, sky_depth_config):
        depth = np.ones((4, 4), dtype=np.float32) * 10.0
        sky_mask = np.zeros((4, 4), dtype=bool)
        samples = [{"id": "no_sky", "depth": depth, "semantic_segmentation": sky_mask}]

        out_path = tmp_path / "output"
        transform = SkyDepthTransform(str(sky_depth_config), str(out_path))
        paths = transform.run(samples)

        result = np.load(paths[0])
        np.testing.assert_array_equal(result, 10.0)

    def test_full_id_creates_subdirectories(self, tmp_path, sky_depth_config):
        depth = np.ones((4, 4), dtype=np.float32)
        sky_mask = np.ones((4, 4), dtype=bool)
        samples = [
            {
                "id": "00001",
                "depth": depth,
                "semantic_segmentation": sky_mask,
                "full_id": "/Scene01/Camera_0/00001",
            }
        ]

        out_path = tmp_path / "output"
        transform = SkyDepthTransform(str(sky_depth_config), str(out_path))
        paths = transform.run(samples)

        assert len(paths) == 1
        assert "Scene01" in str(paths[0])
        assert "Camera_0" in str(paths[0])

    def test_multiple_samples(self, tmp_path, sky_depth_config):
        samples = []
        for i in range(5):
            depth = np.full((4, 4), float(i + 1), dtype=np.float32)
            sky_mask = np.zeros((4, 4), dtype=bool)
            sky_mask[0, 0] = True
            samples.append({"id": f"frame_{i:03d}", "depth": depth, "semantic_segmentation": sky_mask})

        out_path = tmp_path / "output"
        transform = SkyDepthTransform(str(sky_depth_config), str(out_path))
        paths = transform.run(samples)

        assert len(paths) == 5
        for i, path in enumerate(paths):
            result = np.load(path)
            assert result[0, 0] == 500.0  # sky pixel overridden
            assert result[1, 1] == float(i + 1)  # non-sky unchanged

    def test_chw_depth_input(self, tmp_path, sky_depth_config):
        """Test that (1, H, W) depth maps are handled correctly."""
        depth = np.ones((1, 4, 4), dtype=np.float32) * 5.0
        sky_mask = np.zeros((4, 4), dtype=bool)
        sky_mask[0, :] = True
        samples = [{"id": "chw_test", "depth": depth, "semantic_segmentation": sky_mask}]

        out_path = tmp_path / "output"
        transform = SkyDepthTransform(str(sky_depth_config), str(out_path))
        paths = transform.run(samples)

        result = np.load(paths[0])
        assert result.shape == (4, 4)
        assert result[0, 0] == 500.0
        assert result[1, 0] == 5.0

    def test_strict_aborts_on_bad_sanity_samples(self, tmp_path):
        """--strict raises when sanity-check samples look suspicious."""
        cfg = {"sky_depth_value": 500.0, "sanity_check_samples": 2}
        cfg_path = tmp_path / "strict.json"
        cfg_path.write_text(json.dumps(cfg))

        depth = np.ones((4, 4), dtype=np.float32)
        empty_mask = np.zeros((4, 4), dtype=bool)  # 0 sky pixels — bad
        samples = [
            {"id": "no_sky_1", "depth": depth.copy(), "semantic_segmentation": empty_mask.copy()},
            {"id": "no_sky_2", "depth": depth.copy(), "semantic_segmentation": empty_mask.copy()},
            {"id": "no_sky_3", "depth": depth.copy(), "semantic_segmentation": empty_mask.copy()},
        ]

        out_path = tmp_path / "output"
        transform = SkyDepthTransform(str(cfg_path), str(out_path), strict=True)
        with pytest.raises(RuntimeError, match="sanity-check samples failed"):
            transform.run(samples)

    def test_strict_passes_when_samples_look_good(self, tmp_path):
        """--strict does not raise when sanity samples have plausible sky pixels."""
        cfg = {"sky_depth_value": 500.0, "sanity_check_samples": 2}
        cfg_path = tmp_path / "strict_ok.json"
        cfg_path.write_text(json.dumps(cfg))

        depth = np.ones((10, 10), dtype=np.float32)
        sky_mask = np.zeros((10, 10), dtype=bool)
        sky_mask[:3, :] = True  # 30% sky — fine
        samples = [
            {"id": f"f_{i}", "depth": depth.copy(), "semantic_segmentation": sky_mask.copy()}
            for i in range(3)
        ]

        out_path = tmp_path / "output"
        transform = SkyDepthTransform(str(cfg_path), str(out_path), strict=True)
        paths = transform.run(samples)
        assert len(paths) == 3

    def test_non_strict_warns_but_continues(self, tmp_path):
        """Without --strict, bad sanity samples only warn — processing continues."""
        cfg = {"sky_depth_value": 500.0, "sanity_check_samples": 1}
        cfg_path = tmp_path / "lenient.json"
        cfg_path.write_text(json.dumps(cfg))

        depth = np.ones((4, 4), dtype=np.float32)
        empty_mask = np.zeros((4, 4), dtype=bool)
        samples = [
            {"id": f"f_{i}", "depth": depth.copy(), "semantic_segmentation": empty_mask.copy()}
            for i in range(2)
        ]

        out_path = tmp_path / "output"
        transform = SkyDepthTransform(str(cfg_path), str(out_path), strict=False)
        paths = transform.run(samples)
        assert len(paths) == 2
