"""The pipeline only feeds images into the head of an already linked chain of steps."""

from __future__ import annotations

import io
import logging
from collections.abc import Iterable
from pathlib import Path

from PIL import Image

from menu_poc.data_classes import ImageContext, ImageRef
from menu_poc.io.writers import write_jsonl
from menu_poc.models import ImageResult
from menu_poc.steps.base import Step

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

    def ensure_ready(self) -> None:
        """Raise when a required step cannot run (e.g. the OCR models cannot be loaded)."""
        for step in self.steps:
            step.ensure_ready()

    def process(self, ref: ImageRef) -> ImageResult:
        ctx = self.head.handle(ImageContext.for_image(ref))
        ctx.result.timings_s["total"] = sum(ctx.result.timings_s.values())
        return ctx.result

    def run(self, refs: Iterable[ImageRef], run_dir: Path) -> list[ImageResult]:
        """Process every image; results come back (and are saved) in input order."""
        refs = list(refs)
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
        return [done[ref.image_id] for ref in refs]
