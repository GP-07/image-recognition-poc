"""The pipeline only feeds images into the head of an already linked chain of steps."""

from __future__ import annotations

import io
import logging
from collections.abc import Iterable
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from PIL import Image

from menu_poc.constants import LLM_DEFERRED_RETRIES, LLM_HELP, MANIFEST_FILE, RESULTS_FILE
from menu_poc.data_classes import ImageContext, ImageRef, RunOutcome
from menu_poc.interfaces import PostProcessor
from menu_poc.io.dataset import iter_images
from menu_poc.io.writers import write_items_csv, write_json, write_jsonl
from menu_poc.models import ImageResult, RunConfig
from menu_poc.pipeline.runs import new_run_dir
from menu_poc.steps import InputValidationError, LlmStep, Step, require

log = logging.getLogger(__name__)


def _with_total_time(ctx: ImageContext) -> ImageContext:
    timings = ctx.result.timings_s
    timings["total"] = sum(seconds for step, seconds in timings.items() if step != "total")
    return ctx


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _pixels(ref: ImageRef) -> int:
    width, height = Image.open(io.BytesIO(ref.data)).size  # lazy open: header only
    return width * height


class Pipeline:
    """Built by ``PipelineBuilder``; ``steps`` are linked in the system's fixed order and the
    ``post_processors`` run once at the end of a run, in the order they were added."""

    def __init__(
        self,
        steps: list[Step],
        post_processors: list[PostProcessor] | None = None,
        llm_step: LlmStep | None = None,
    ) -> None:
        self.steps = steps
        self.head = steps[0]
        self.post_processors = post_processors or []
        # Also in ``steps``; kept apart because the run asks about it and retries from it.
        self.llm_step = llm_step
        self.warnings: list[str] = []  # from ``validate_inputs``, saved in the run manifest

    @property
    def step_names(self) -> list[str]:
        return [step.name for step in self.steps]

    @property
    def uses_llm(self) -> bool:
        """True when the LLM step is in the chain and passed its validation."""
        return self.llm_step is not None and self.llm_step.enabled

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
        if self.llm_step is None:
            warnings.append(f"No LLM configured: results come from OCR + rules only. {LLM_HELP}")
        self.warnings = warnings
        return refs, warnings

    def process(self, ref: ImageRef) -> ImageResult:
        return self._process(ref).result

    def _process(self, ref: ImageRef) -> ImageContext:
        return _with_total_time(self.head.handle(ImageContext.for_image(ref)))

    def run(self, refs: Iterable[ImageRef], config: RunConfig) -> RunOutcome:
        """Process every image into a new run directory under ``config.runs_dir`` and save its
        files: ``results.jsonl`` (kept up to date while running), then ``items.csv``, the
        post-processors' reports and ``manifest.json``. Transient LLM failures are retried at the
        end (``_retry_llm``). Results come back in input order."""
        refs = list(refs)
        started = _now()
        # The run directory records what actually runs: no LLM model when its step was disabled.
        effective = config if self.uses_llm else config.model_copy(update={"llm_model": None})
        run_dir = new_run_dir(effective)
        # Largest images first: the first photo of a duplicate group is kept as the original,
        # and the largest one gives the best OCR.
        order = sorted(refs, key=lambda ref: (-_pixels(ref), ref.image_id))
        done: dict[str, ImageContext] = {}
        to_retry: list[ImageContext] = []
        for ref in order:
            log.info("processing %s", ref.image_id)
            ctx = self._process(ref)
            done[ref.image_id] = ctx
            if ctx.should_be_retried:
                to_retry.append(ctx)
            # Rewrite after every image so an interrupted run keeps its progress.
            finished = [done[ref.image_id].result for ref in refs if ref.image_id in done]
            write_jsonl(run_dir / RESULTS_FILE, finished)
        if to_retry:
            self._retry_llm(to_retry, done)
            write_jsonl(run_dir / RESULTS_FILE, [done[ref.image_id].result for ref in refs])

        outcome = RunOutcome(
            run_dir=run_dir,
            requested_llm_model=config.llm_model,
            config=effective,
            results=[done[ref.image_id].result for ref in refs],
            warnings=self.warnings,
            started=started,
            finished=_now(),
        )
        write_items_csv(outcome.items_path, outcome.results)
        outcome = self._post_process(outcome)
        write_json(run_dir / MANIFEST_FILE, outcome.manifest())
        return outcome

    def _retry_llm(self, to_retry: list[ImageContext], done: dict[str, ImageContext]) -> None:
        """Retry, from the LLM step only (the OCR is already done), the images whose LLM call
        failed transiently (timeout, rate limit, 5xx) after the client's own immediate retries.
        Up to ``LLM_DEFERRED_RETRIES`` rounds; meanwhile those images keep the rules output."""
        llm_step = require(self.llm_step, "LLM step")
        pending = to_retry
        for round_number in range(1, LLM_DEFERRED_RETRIES + 1):
            log.info("retrying the LLM for %d images (round %d)", len(pending), round_number)
            for ctx in pending:
                ctx.should_be_retried, ctx.result.error = False, None
                ctx.result.llm_retries = round_number
                _with_total_time(llm_step.handle(ctx))
            pending = [ctx for ctx in pending if ctx.should_be_retried]
            if not pending:
                break
        retried = {ctx.ref.image_id for ctx in to_retry}
        for ctx in done.values():  # copies of a retried photo take its new items
            if ctx.result.duplicate_of in retried:
                ctx.result.dishes = done[ctx.result.duplicate_of].result.dishes

    def _post_process(self, outcome: RunOutcome) -> RunOutcome:
        """A failing post-processor never loses the run: its error becomes a warning."""
        reports: list[Path] = []
        warnings = list(outcome.warnings)
        for post_processor in self.post_processors:
            try:
                reports += post_processor.run(outcome)
            except Exception as exc:
                log.exception("post-processor %s failed", post_processor.name)
                warnings.append(f"Post-processor {post_processor.name} failed: {exc}")
        return replace(outcome, reports=reports, warnings=warnings)
