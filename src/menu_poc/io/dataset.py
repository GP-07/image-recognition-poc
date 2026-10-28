"""Input reading: a .zip (read in place, never extracted) or a directory of images.

Expected layout: ``<restaurant_id>/<image>.jpg``, optionally under a top-level folder.
The image id is ``<restaurant_id>/<file stem>``.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from PIL import Image

from menu_poc.cache.store import sha256_bytes

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ImageRef:
    image_id: str
    restaurant_id: str
    filename: str
    data: bytes

    @property
    def sha256(self) -> str:
        return sha256_bytes(self.data)

    def open(self) -> Image.Image:
        image = Image.open(io.BytesIO(self.data))
        image.load()
        return image


def _is_image(path: PurePosixPath) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES and not any(
        part.startswith(("__MACOSX", ".")) for part in path.parts
    )


def _ref(path: PurePosixPath, data: bytes) -> ImageRef:
    restaurant_id = path.parent.name or "unknown"
    return ImageRef(
        image_id=f"{restaurant_id}/{path.stem}",
        restaurant_id=restaurant_id,
        filename=path.name,
        data=data,
    )


def iter_images(source: Path, image_ids: list[str] | None = None) -> Iterator[ImageRef]:
    """Yield images sorted by id. ``image_ids`` restricts the selection."""
    source = Path(source)
    wanted = set(image_ids) if image_ids else None

    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            names = sorted(n for n in zf.namelist() if _is_image(PurePosixPath(n)))
            for name in names:
                path = PurePosixPath(name)
                ref_id = f"{path.parent.name}/{path.stem}"
                if wanted is None or ref_id in wanted:
                    yield _ref(path, zf.read(name))
    elif source.is_dir():
        for file in sorted(p for p in source.rglob("*") if p.is_file()):
            path = PurePosixPath(file.relative_to(source).as_posix())
            if not _is_image(path):
                continue
            ref_id = f"{path.parent.name}/{path.stem}"
            if wanted is None or ref_id in wanted:
                yield _ref(path, file.read_bytes())
    else:
        raise FileNotFoundError(f"Input must be a .zip file or a directory: {source}")
