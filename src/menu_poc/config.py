"""Run configuration: one YAML file plus CLI overrides. One LLM model per run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from menu_poc.constants import CONFIG_PATH_FIELDS, PROJECT_ROOT
from menu_poc.models import RunConfig


def load_config(path: Path, **overrides: Any) -> RunConfig:
    """Load the YAML config and apply the non-None overrides on top.

    Relative paths in the YAML file are relative to the project root; relative paths given as
    overrides (CLI flags) are relative to the current directory, as usual for a command line.
    """
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    for field in CONFIG_PATH_FIELDS:
        if data.get(field) is not None:
            data[field] = PROJECT_ROOT / Path(data[field]).expanduser()
    for key, value in overrides.items():
        if value is None:
            continue
        data[key] = Path(value).expanduser().resolve() if key in CONFIG_PATH_FIELDS else value
    return RunConfig(**data)
