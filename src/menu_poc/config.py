"""Run configuration: one YAML file plus CLI overrides. One LLM model per run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Mode = Literal["ocr", "ocr+llm"]
LLMPolicy = Literal["fallback", "always"]


class RunConfig(BaseModel):
    input_path: Path
    runs_dir: Path = Path("runs")
    artifacts_dir: Path = Path("artifacts")

    mode: Mode = "ocr"
    llm_policy: LLMPolicy = "fallback"
    llm_model: str = "claude-haiku-5-5"
    llm_offline: bool = False  # only use cached LLM answers; never call the API

    confidence_threshold: float = Field(0.6, ge=0.0, le=1.0)
    max_image_side: int = Field(2000, gt=0)

    # Image selection
    image_ids: list[str] | None = None  # restrict to these "<restaurant>/<file stem>" ids
    reuse_from: Path | None = None  # previous run dir; with only_low_confidence picks its images
    only_low_confidence: bool = False

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
    """Load a YAML config (optional) and apply non-None overrides on top."""
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    return RunConfig(**data)
