"""Unit tests for RadialTransform."""

import json

import numpy as np
import pytest

from euler_preprocess.radial.transform import RadialTransform


@pytest.fixture
def radial_config(tmp_path):
    cfg = {}
    cfg_path = tmp_path / "radial.json"
    cfg_path.write_text(json.dumps(cfg))
    return cfg_path


def _make_intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


class TestRadialTransform:
    def test_center_pixel_unchanged(self, tmp_path, radial_config):
        """At the optical center, planar == radial (factor == 1)."""
        H, W = 480, 640
        K = _make_intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        depth = np.ones((H, W), dtype=np.float32) * 10.0
        samples = [
            {
                "id": "center_test",
                "depth": depth,
                "intrinsics": {"intrinsics": K},
            }
        ]

        out_path = tmp_path / "output"
        transform = RadialTransform(str(radial_config), str(out_path))
        paths = transform.run(samples)

        result = np.load(paths[0])
        # At (cx, cy) the factor is sqrt(0 + 0 + 1) = 1.0
        assert result[240, 320] == pytest.approx(10.0, abs=1e-5)

    def test_off_center_increases(self, tmp_path, radial_config):
        """Off-center pixels should have radial > planar depth."""
        H, W = 480, 640
        K = _make_intrinsics()
        depth = np.ones((H, W), dtype=np.float32) * 10.0
        samples = [
            {
                "id": "off_center",
                "depth": depth,
                "intrinsics": {"intrinsics": K},
            }
        ]

        out_path = tmp_path / "output"
        transform = RadialTransform(str(radial_config), str(out_path))
        paths = transform.run(samples)

        result = np.load(paths[0])
        # Corner pixel should be strictly greater
        assert result[0, 0] > 10.0
        assert result[H - 1, W - 1] > 10.0

    def test_zero_depth_stays_zero(self, tmp_path, radial_config):
        """Zero depth should remain zero after conversion."""
        H, W = 100, 100
        K = _make_intrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0)
        depth = np.zeros((H, W), dtype=np.float32)
        samples = [
            {
                "id": "zero_depth",
                "depth": depth,
                "intrinsics": {"intrinsics": K},
            }
        ]

        out_path = tmp_path / "output"
        transform = RadialTransform(str(radial_config), str(out_path))
        paths = transform.run(samples)

        result = np.load(paths[0])
        np.testing.assert_array_equal(result, 0.0)

    def test_missing_intrinsics_raises(self, tmp_path, radial_config):
        """Should raise ValueError when intrinsics are missing."""
        depth = np.ones((10, 10), dtype=np.float32)
        samples = [{"id": "no_intr", "depth": depth}]

        out_path = tmp_path / "output"
        transform = RadialTransform(str(radial_config), str(out_path))
        with pytest.raises(ValueError, match="no intrinsics"):
            transform.run(samples)

    def test_full_id_creates_subdirectories(self, tmp_path, radial_config):
        K = _make_intrinsics()
        depth = np.ones((4, 4), dtype=np.float32)
        samples = [
            {
                "id": "00001",
                "depth": depth,
                "intrinsics": {"intrinsics": K},
                "full_id": "/Scene01/Camera_0/00001",
            }
        ]

        out_path = tmp_path / "output"
        transform = RadialTransform(str(radial_config), str(out_path))
        paths = transform.run(samples)

        assert len(paths) == 1
        assert "Scene01" in str(paths[0])
        assert "Camera_0" in str(paths[0])

    def test_known_conversion(self, tmp_path, radial_config):
        """Verify against manually computed value."""
        K = _make_intrinsics(fx=100.0, fy=100.0, cx=50.0, cy=50.0)
        H, W = 100, 100
        depth = np.ones((H, W), dtype=np.float32) * 5.0
        samples = [
            {
                "id": "known",
                "depth": depth,
                "intrinsics": {"intrinsics": K},
            }
        ]

        out_path = tmp_path / "output"
        transform = RadialTransform(str(radial_config), str(out_path))
        paths = transform.run(samples)

        result = np.load(paths[0])
        # At pixel (0, 0): factor = sqrt(((0-50)/100)^2 + ((0-50)/100)^2 + 1)
        #                         = sqrt(0.25 + 0.25 + 1) = sqrt(1.5)
        expected = 5.0 * np.sqrt(1.5)
        assert result[0, 0] == pytest.approx(expected, rel=1e-5)
