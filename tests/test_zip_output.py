"""Tests for zip output mode via OutputWriter."""

import io
import json
import zipfile

import numpy as np
from PIL import Image

from euler_preprocess.common.io import OutputWriter


class TestOutputWriterDisk:
    """Sanity checks that disk mode still works through the writer."""

    def test_save_depth_npy(self, tmp_path):
        out = tmp_path / "output"
        writer = OutputWriter(str(out))
        writer.mkdir(out)
        path = out / "test.npy"
        depth = np.ones((4, 4), dtype=np.float32)
        writer.save_depth_npy(path, depth)
        writer.close()

        result = np.load(path)
        np.testing.assert_array_equal(result, 1.0)

    def test_save_image(self, tmp_path):
        out = tmp_path / "output"
        writer = OutputWriter(str(out))
        writer.mkdir(out)
        path = out / "test.png"
        rgb = np.full((4, 4, 3), 0.5, dtype=np.float32)
        writer.save_image(path, rgb)
        writer.close()

        img = Image.open(path)
        assert img.size == (4, 4)

    def test_write_json(self, tmp_path):
        out = tmp_path / "output"
        writer = OutputWriter(str(out))
        path = out / "sub" / "config.json"
        writer.write_json(path, {"key": "value"})
        writer.close()

        with open(path) as f:
            assert json.load(f) == {"key": "value"}


class TestOutputWriterZip:
    """Verify that .zip output produces a valid archive."""

    def test_save_depth_npy(self, tmp_path):
        zip_path = tmp_path / "output.zip"
        writer = OutputWriter(str(zip_path))
        path = writer.root / "test.npy"
        depth = np.arange(12, dtype=np.float32).reshape(3, 4)
        writer.save_depth_npy(path, depth)
        writer.close()

        with zipfile.ZipFile(zip_path, "r") as zf:
            assert "test.npy" in zf.namelist()
            buf = io.BytesIO(zf.read("test.npy"))
            result = np.load(buf)
            np.testing.assert_array_equal(result, depth)

    def test_save_image(self, tmp_path):
        zip_path = tmp_path / "output.zip"
        writer = OutputWriter(str(zip_path))
        path = writer.root / "img.png"
        rgb = np.full((8, 8, 3), 0.5, dtype=np.float32)
        writer.save_image(path, rgb)
        writer.close()

        with zipfile.ZipFile(zip_path, "r") as zf:
            assert "img.png" in zf.namelist()
            buf = io.BytesIO(zf.read("img.png"))
            img = Image.open(buf)
            assert img.size == (8, 8)

    def test_write_json(self, tmp_path):
        zip_path = tmp_path / "output.zip"
        writer = OutputWriter(str(zip_path))
        path = writer.root / "sub" / "config.json"
        writer.write_json(path, {"a": 1})
        writer.close()

        with zipfile.ZipFile(zip_path, "r") as zf:
            assert "sub/config.json" in zf.namelist()
            data = json.loads(zf.read("sub/config.json"))
            assert data == {"a": 1}

    def test_hierarchical_paths(self, tmp_path):
        zip_path = tmp_path / "output.zip"
        writer = OutputWriter(str(zip_path))
        path = writer.root / "Scene01" / "Camera_0" / "frame.npy"
        depth = np.ones((2, 2), dtype=np.float32)
        writer.save_depth_npy(path, depth)
        writer.close()

        with zipfile.ZipFile(zip_path, "r") as zf:
            assert "Scene01/Camera_0/frame.npy" in zf.namelist()

    def test_context_manager(self, tmp_path):
        zip_path = tmp_path / "output.zip"
        with OutputWriter(str(zip_path)) as writer:
            path = writer.root / "test.npy"
            writer.save_depth_npy(path, np.zeros((2, 2), dtype=np.float32))

        # Archive should be finalised after exiting context
        with zipfile.ZipFile(zip_path, "r") as zf:
            assert "test.npy" in zf.namelist()

    def test_mkdir_is_noop(self, tmp_path):
        zip_path = tmp_path / "output.zip"
        writer = OutputWriter(str(zip_path))
        # Should not raise or create filesystem directories
        writer.mkdir(writer.root / "some" / "deep" / "path")
        assert not (tmp_path / "output" / "some").exists()
        writer.close()


class TestSkyDepthZipOutput:
    """End-to-end: SkyDepthTransform writing to a zip."""

    def test_writes_to_zip(self, tmp_path):
        from euler_preprocess.sky_depth.transform import SkyDepthTransform

        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps({"sky_depth_value": 999.0}))

        zip_path = tmp_path / "output.zip"
        transform = SkyDepthTransform(str(cfg_path), str(zip_path))

        depth = np.ones((4, 4), dtype=np.float32) * 5.0
        sky_mask = np.zeros((4, 4), dtype=bool)
        sky_mask[0, :] = True
        samples = [{"id": "frame_001", "depth": depth, "semantic_segmentation": sky_mask}]

        paths = transform.run(samples)
        assert len(paths) == 1

        with zipfile.ZipFile(zip_path, "r") as zf:
            assert "frame_001.npy" in zf.namelist()
            buf = io.BytesIO(zf.read("frame_001.npy"))
            result = np.load(buf)
            assert result[0, 0] == 999.0  # sky pixel
            assert result[1, 0] == 5.0    # non-sky pixel


class TestRadialZipOutput:
    """End-to-end: RadialTransform writing to a zip."""

    def test_writes_to_zip(self, tmp_path):
        from euler_preprocess.radial.transform import RadialTransform

        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps({}))

        zip_path = tmp_path / "output.zip"
        transform = RadialTransform(str(cfg_path), str(zip_path))

        K = np.array([[500, 0, 2], [0, 500, 2], [0, 0, 1]], dtype=np.float64)
        depth = np.ones((4, 4), dtype=np.float32)
        samples = [
            {
                "id": "frame_001",
                "depth": depth,
                "intrinsics": {"intrinsics": K},
            }
        ]

        paths = transform.run(samples)
        assert len(paths) == 1

        with zipfile.ZipFile(zip_path, "r") as zf:
            assert "frame_001.npy" in zf.namelist()
            buf = io.BytesIO(zf.read("frame_001.npy"))
            result = np.load(buf)
            assert result.shape == (4, 4)
            assert result.dtype == np.float32
