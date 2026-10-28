"""The pipeline only feeds images into the head of an already linked chain of steps."""

from __future__ import annotations

import io
import logging
from collections.abc import Iterable
from pathlib import Path

from PIL import Image

from menu_poc.constants import LLM_HELP
from menu_poc.data_classes import ImageContext, ImageRef, RunOutcome
from menu_poc.io.dataset import iter_images
from menu_poc.io.writers import write_jsonl
from menu_poc.models import ImageResult, RunConfig
from menu_poc.pipeline.runs import new_run_dir
from menu_poc.steps import InputValidationError, LlmStep, Step

log = logging.getLogger(__name__)


def _pixels(ref: ImageRef) -> int:
    width, height = Image.open(io.BytesIO(ref.data)).size  # lazy open: header only
    return width * height


class Pipeline:
    """Built by ``PipelineBuilder``; ``steps`` are linked in the system's fixed order."""

    def __init__(self, steps: list[Step]) -> None:
        self.steps = steps
        self.head = steps[0]

    @property
    def step_names(self) -> list[str]:
        return [step.name for step in self.steps]

    @property
    def uses_llm(self) -> bool:
        """True when the LLM step is in the chain and passed its validation."""
        return any(isinstance(step, LlmStep) and step.enabled for step in self.steps)

    def validate_inputs(self, input_path: Path) -> tuple[list[ImageRef], list[str]]:
        """Run-level checks, once, before any image: the input has images, and every step can do
        its job (``Step.validate``). Returns the images and the warnings; raises
        ``InputValidationError`` when the run cannot go on."""
        try:
            refs = list(iter_images(input_path))
        except (FileNotFoundError, ValueError) as exc:  # missing input, clashing image ids
            raise InputValidationError(str(exc)) from exc
        if not refs:
            raise InputValidationError(f"no images found in {input_path}")
        warnings = [warning for step in self.steps for warning in step.validate()]
        if not any(isinstance(step, LlmStep) for step in self.steps):
            warnings.append(f"No LLM configured: results come from OCR + rules only. {LLM_HELP}")
        return refs, warnings

    def process(self, ref: ImageRef) -> ImageResult:
        ctx = self.head.handle(ImageContext.for_image(ref))
        ctx.result.timings_s["total"] = sum(ctx.result.timings_s.values())
        return ctx.result

    def run(self, refs: Iterable[ImageRef], config: RunConfig) -> RunOutcome:
        """Process every image into a new run directory under ``config.runs_dir``; results come
        back (and are saved) in input order."""
        refs = list(refs)
        # The run directory records what actually runs: no LLM model when its step was disabled.
        effective = config if self.uses_llm else config.model_copy(update={"llm_model": None})
        run_dir = new_run_dir(effective)
        # Largest images first: the first photo of a duplicate group is kept as the original,
        # and the largest one gives the best OCR.
        order = sorted(refs, key=lambda ref: (-_pixels(ref), ref.image_id))
        done: dict[str, ImageResult] = {}
        for ref in order:
            log.info("processing %s", ref.image_id)
            done[ref.image_id] = self.process(ref)
            # Rewrite after every image so an interrupted run keeps its progress.
            finished = [done[ref.image_id] for ref in refs if ref.image_id in done]
            write_jsonl(run_dir / "results.jsonl", finished)
        results = [done[ref.image_id] for ref in refs]
        return RunOutcome(run_dir=run_dir, config=effective, results=results)
