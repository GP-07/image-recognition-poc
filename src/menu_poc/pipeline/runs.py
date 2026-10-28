"""Run-level helpers: which images to process, where results go, carrying over a previous run."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from menu_poc.data_classes import ImageRef
from menu_poc.io.dataset import iter_images
from menu_poc.io.writers import read_jsonl, write_yaml
from menu_poc.models import ImageResult, RunConfig


def merge_previous(config: RunConfig, results: list[ImageResult]) -> list[ImageResult]:
    """With ``reuse_from``: carry over the previous run's results for the images not re-run, so
    the new run covers the same image set and stays comparable with other runs."""
    if config.reuse_from is None:
        return results
    redone = {result.image_id for result in results}
    previous = read_jsonl(config.reuse_from / "results.jsonl", ImageResult)
    merged = [result for result in previous if result.image_id not in redone] + results
    return sorted(merged, key=lambda result: result.image_id)


def select_images(config: RunConfig) -> list[ImageRef]:
    """Images for this run: explicit ids, or the low-confidence ones of a previous run."""
    ids = config.image_ids
    if config.reuse_from is not None and config.only_low_confidence:
        previous = read_jsonl(config.reuse_from / "results.jsonl", ImageResult)
        low = [
            result.image_id
            for result in previous
            if result.rules_confidence < config.confidence_threshold
        ]
        ids = [image_id for image_id in low if ids is None or image_id in ids]
        if not ids:
            return []
    return list(iter_images(config.input_path, ids))


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
