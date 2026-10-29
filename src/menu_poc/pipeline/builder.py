"""Builder: steps can be added in any order; ``build`` links them in the system's fixed order.

The order is not the caller's choice: duplicates are skipped first, the LLM never runs before
OCR and rules, and LLM output is always verified. Optional pieces (dedup, LLM) are simply not
added. Post-processors (e.g. evaluation against the golden set) run after the whole run, in the
order they are added; by default there are none.
"""

from __future__ import annotations

from menu_poc.cache.store import ArtifactStore
from menu_poc.evaluation.post_process import GoldenSetEvaluation
from menu_poc.interfaces import PostProcessor
from menu_poc.llm.engine import LLMEngine
from menu_poc.llm.grounding import GroundingVerifier
from menu_poc.llm.structurer import LLMStructurer
from menu_poc.models import RunConfig
from menu_poc.ocr.rapid import RapidOcrEngine
from menu_poc.parsing.rules import RuleBasedParser
from menu_poc.pipeline.core import Pipeline
from menu_poc.preprocess.image import ImagePreprocessor
from menu_poc.steps import DedupStep, LlmStep, OcrStep, RulesStep, Step, VerifyStep


class PipelineBuildError(ValueError):
    """The requested steps cannot form a valid pipeline."""


class PipelineBuilder:
    ORDER: tuple[type[Step], ...] = (DedupStep, OcrStep, RulesStep, LlmStep, VerifyStep)
    REQUIRES: dict[type[Step], tuple[type[Step], ...]] = {
        RulesStep: (OcrStep,),
        LlmStep: (OcrStep, RulesStep),  # the LLM reads the OCR text; rules give the fallback
        VerifyStep: (LlmStep,),
    }

    def __init__(self) -> None:
        self._steps: dict[type[Step], Step] = {}
        self._post_processors: list[PostProcessor] = []

    def add(self, step: Step) -> PipelineBuilder:
        """Add (or replace) the step of its kind."""
        kind = next((kind for kind in self.ORDER if kind.name == step.name), None)
        if kind is None:
            raise PipelineBuildError(f"Unknown step {step.name!r}")
        self._steps[kind] = step
        return self

    def add_post_processor(self, post_processor: PostProcessor) -> PipelineBuilder:
        self._post_processors.append(post_processor)
        return self

    def build(self) -> Pipeline:
        if LlmStep in self._steps and VerifyStep not in self._steps:
            self.add(VerifyStep(GroundingVerifier()))  # LLM output is never trusted unverified
        if OcrStep not in self._steps:
            raise PipelineBuildError("A pipeline needs at least the OCR step")
        for kind, needs in self.REQUIRES.items():
            missing = [
                need.name for need in needs if kind in self._steps and need not in self._steps
            ]
            if missing:
                raise PipelineBuildError(f"Step {kind.name!r} needs {', '.join(missing)}")

        steps = [self._steps[kind] for kind in self.ORDER if kind in self._steps]
        for step, following in zip(steps, steps[1:], strict=False):
            step.next = following
        steps[-1].next = None
        llm_step = self._steps.get(LlmStep)
        return Pipeline(steps, list(self._post_processors), llm_step)

    @classmethod
    def from_config(cls, config: RunConfig) -> PipelineBuilder:
        """The production pieces for a run configuration."""
        store = ArtifactStore(config.artifacts_dir)
        builder = cls()
        builder.add(OcrStep(ImagePreprocessor(config.max_image_side), RapidOcrEngine(), store))
        builder.add(RulesStep(RuleBasedParser()))
        if config.dedup:
            builder.add(DedupStep(config.dedup_max_distance))
        if config.uses_llm:
            builder.add(
                LlmStep(
                    LLMStructurer(LLMEngine.get_for_alias(config.llm_model)),
                    store,
                    config.llm_policy,
                    config.confidence_threshold,
                    config.llm_offline,
                )
            )
        if config.golden_dir is not None and config.golden_dir.is_dir():
            builder.add_post_processor(GoldenSetEvaluation(config.golden_dir))
        return builder
