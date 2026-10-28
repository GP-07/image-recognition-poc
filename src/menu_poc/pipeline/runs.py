"""Run directories: one per run, named after what actually ran."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from menu_poc.io.writers import write_yaml
from menu_poc.models import RunConfig


def new_run_dir(config: RunConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = "ocr" if not config.uses_llm else f"llm-{config.llm_policy}-{config.llm_model}"
    run_dir = config.runs_dir / f"{stamp}_{label}"
    suffix = 1
    while run_dir.exists():  # two runs started in the same second
        suffix += 1
        run_dir = config.runs_dir / f"{stamp}_{label}_{suffix}"
    run_dir.mkdir(parents=True)
    write_yaml(run_dir / "config.yaml", config.model_dump(mode="json"))
    return run_dir
