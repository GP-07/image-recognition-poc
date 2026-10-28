"""Run configuration: one YAML file plus CLI overrides. One LLM model per run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from menu_poc.llm.engine import LLMEngine

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATH_FIELDS = ("input_path", "runs_dir", "artifacts_dir", "reuse_from")

Mode = Literal["ocr", "ocr+llm"]
LLMPolicy = Literal["fallback", "always"]


class RunConfig(BaseModel):
    input_path: Path
    runs_dir: Path = PROJECT_ROOT / "tmp"  # one sub-directory per run
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"  # stage cache, reused across runs

    mode: Mode = "ocr"
    llm_policy: LLMPolicy = "fallback"
    # Alias (haiku, sonnet, opus, ollama) or exact id; stored resolved to the exact id.
    llm_model: str = Field("haiku", validate_default=True)
    llm_offline: bool = False  # only use cached LLM answers; never call the API

    confidence_threshold: float = Field(0.6, ge=0.0, le=1.0)
    dedup: bool = True  # skip OCR/LLM for duplicate photos (same file or near-identical)
    dedup_max_distance: int = Field(10, ge=0)  # perceptual hash bits out of 256
    max_image_side: int = Field(2000, gt=0)

    # Image selection
    image_ids: list[str] | None = None  # restrict to these "<restaurant>/<file stem>" ids
    reuse_from: Path | None = None  # previous run dir; with only_low_confidence picks its images
    only_low_confidence: bool = False

    @field_validator("llm_model")
    @classmethod
    def _resolve_model(cls, value: str) -> str:
        return LLMEngine.resolve(value)[1]

    @model_validator(mode="after")
    def _check(self) -> RunConfig:
        if self.only_low_confidence != (self.reuse_from is not None):
            raise ValueError("only_low_confidence and reuse_from go together")
        if self.only_low_confidence and not self.uses_llm:
            raise ValueError("re-running low-confidence images only makes sense with mode ocr+llm")
        return self

    @property
    def uses_llm(self) -> bool:
        return self.mode == "ocr+llm"


def load_config(path: Path | None = None, **overrides: Any) -> RunConfig:
    """Load a YAML config (optional) and apply non-None overrides on top.

    Relative paths in the YAML file are relative to the project root; relative paths given as
    overrides (CLI flags) are relative to the current directory, as usual for a command line.
    """
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
        for field in PATH_FIELDS:
            if data.get(field) is not None:
                data[field] = PROJECT_ROOT / Path(data[field]).expanduser()
    for key, value in overrides.items():
        if value is None:
            continue
        data[key] = Path(value).expanduser().resolve() if key in PATH_FIELDS else value
    return RunConfig(**data)
