"""Input reading: a .zip (read in place, never extracted) or a directory of images.

Expected layout: ``<restaurant_id>/<image>.jpg``, optionally under a top-level folder.
The image id is ``<restaurant_id>/<file stem>``.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from menu_poc.constants import IMAGE_SUFFIXES
from menu_poc.data_classes import ImageRef


def _is_image(path: PurePosixPath) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES and not any(
        part.startswith(("__MACOSX", ".")) for part in path.parts
    )


def _image_id(path: PurePosixPath) -> str:
    return f"{path.parent.name or 'unknown'}/{path.stem}"


def _ref(path: PurePosixPath, data: bytes) -> ImageRef:
    image_id = _image_id(path)
    return ImageRef(
        image_id=image_id,
        restaurant_id=image_id.split("/")[0],
        filename=path.name,
        data=data,
    )


def _check_unique(paths: list[PurePosixPath]) -> None:
    seen: dict[str, PurePosixPath] = {}
    for path in paths:
        other = seen.setdefault(_image_id(path), path)
        if other != path:
            raise ValueError(f"Duplicate image id {_image_id(path)!r}: {other} and {path}")


def iter_images(source: Path) -> Iterator[ImageRef]:
    """Yield every image of the input, sorted by id."""
    source = Path(source)

    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            names = sorted(name for name in archive.namelist() if _is_image(PurePosixPath(name)))
            _check_unique([PurePosixPath(name) for name in names])
            for name in names:
                yield _ref(PurePosixPath(name), archive.read(name))
    elif source.is_dir():
        files = sorted(file for file in source.rglob("*") if file.is_file())
        paths = {file: PurePosixPath(file.relative_to(source).as_posix()) for file in files}
        images = {file: path for file, path in paths.items() if _is_image(path)}
        _check_unique(list(images.values()))
        for file, path in images.items():
            yield _ref(path, file.read_bytes())
    else:
        raise FileNotFoundError(f"Input must be a .zip file or a directory: {source}")
