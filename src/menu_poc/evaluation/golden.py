"""Golden set: one YAML per image at ``data/golden/<restaurant_id>__<file_stem>.yaml``.

Each dish is labelled once; its translations on the same menu go in ``aliases``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

Split = Literal["dev", "test"]


class GoldenError(ValueError):
    """A golden file is malformed; the message starts with the file name."""


class GoldenDish(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("dish name must not be empty")
        return v

    @property
    def variants(self) -> list[str]:
        """Name plus aliases (any of them counts as recognising the dish)."""
        return [self.name, *self.aliases]


class GoldenImage(BaseModel):
    image_id: str  # "<restaurant_id>/<file_stem>"
    split: Split
    reviewed: bool = False
    notes: str | None = None
    dishes: list[GoldenDish] = Field(default_factory=list)


def expected_filename(image_id: str) -> str:
    """File name a golden YAML must have for ``image_id``."""
    return image_id.replace("/", "__", 1) + ".yaml"


def load_golden_file(path: Path) -> GoldenImage:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise GoldenError(f"{path.name}: invalid YAML: {e}") from e
    if not isinstance(data, dict):
        raise GoldenError(f"{path.name}: expected a mapping at the top level")
    try:
        golden = GoldenImage.model_validate(data)
    except ValidationError as e:
        raise GoldenError(f"{path.name}: {e}") from e
    if path.name != expected_filename(golden.image_id):
        raise GoldenError(
            f"{path.name}: image_id {golden.image_id!r} implies file name "
            f"{expected_filename(golden.image_id)!r}"
        )
    return golden


def load_golden(golden_dir: Path, only_reviewed: bool = False) -> dict[str, GoldenImage]:
    """Load every ``*.yaml`` in ``golden_dir`` keyed by image_id.

    With ``only_reviewed`` the files not yet reviewed by a human are skipped (their images then
    count as unlabeled). A missing directory yields an empty golden set.
    """
    golden_dir = Path(golden_dir)
    out: dict[str, GoldenImage] = {}
    if not golden_dir.is_dir():
        return out
    for path in sorted([*golden_dir.glob("*.yaml"), *golden_dir.glob("*.yml")]):
        golden = load_golden_file(path)
        if golden.image_id in out:
            raise GoldenError(f"{path.name}: duplicate image_id {golden.image_id!r}")
        if only_reviewed and not golden.reviewed:
            continue
        out[golden.image_id] = golden
    return out
