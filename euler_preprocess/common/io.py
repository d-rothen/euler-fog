from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from types import TracebackType

import numpy as np
from PIL import Image


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_image(path: Path, rgb: np.ndarray) -> None:
    rgb = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    img.save(path)


def save_depth_npy(path: Path, depth: np.ndarray) -> None:
    """Save a float32 depth map as a ``.npy`` file."""
    np.save(path, depth.astype(np.float32))


class OutputWriter:
    """Writes output files to disk or into a zip archive.

    Auto-detects zip mode when *base_path* ends in ``.zip``.
    Use as a context manager to ensure the archive is finalised::

        writer = OutputWriter("/path/to/output.zip")
        with writer:
            writer.save_image(writer.root / "img.png", rgb)

    In **disk mode** the *root* is *base_path* itself and files are written
    normally.  In **zip mode** *root* is a virtual directory (the zip path
    with the ``.zip`` suffix stripped) so that transforms can build paths
    under it just like they do for disk mode — the writer converts them to
    archive-relative names automatically.
    """

    def __init__(self, base_path: str | Path) -> None:
        self.base_path = Path(base_path)
        self.is_zip = self.base_path.suffix == ".zip"
        self._zf: zipfile.ZipFile | None = None

        if self.is_zip:
            self.root = self.base_path.with_suffix("")
            self.base_path.parent.mkdir(parents=True, exist_ok=True)
            self._zf = zipfile.ZipFile(
                self.base_path, "w", zipfile.ZIP_DEFLATED,
            )
        else:
            self.root = self.base_path

    def __enter__(self) -> OutputWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._zf is not None:
            self._zf.close()
            self._zf = None

    def mkdir(self, path: Path) -> None:
        """Create directories on disk (no-op in zip mode)."""
        if not self.is_zip:
            path.mkdir(parents=True, exist_ok=True)

    def save_image(self, path: Path, rgb: np.ndarray) -> None:
        rgb = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
        img = Image.fromarray(rgb, mode="RGB")
        if self.is_zip:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            self._zf.writestr(self._arcname(path), buf.getvalue())
        else:
            img.save(path)

    def save_depth_npy(self, path: Path, depth: np.ndarray) -> None:
        depth = depth.astype(np.float32)
        if self.is_zip:
            buf = io.BytesIO()
            np.save(buf, depth)
            self._zf.writestr(self._arcname(path), buf.getvalue())
        else:
            np.save(path, depth)

    def write_json(self, path: Path, data: dict) -> None:
        text = json.dumps(data, indent=2, sort_keys=True)
        if self.is_zip:
            self._zf.writestr(self._arcname(path), text)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)

    def _arcname(self, path: Path) -> str:
        """Convert an absolute path to an archive-relative name."""
        return str(path.relative_to(self.root))
