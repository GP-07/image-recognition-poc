"""Builder: steps can be added in any order; ``build`` links them in the system's fixed order.

The order is not the caller's choice: duplicates are skipped first, the LLM never runs before
OCR and rules, and LLM output is always verified. Optional pieces (dedup, LLM) are simply not
added.
"""

from __future__ import annotations

from menu_poc.cache.store import ArtifactStore
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

    def add(self, step: Step) -> PipelineBuilder:
        """Add (or replace) the step of its kind."""
        kind = next((kind for kind in self.ORDER if isinstance(step, kind)), None)
        if kind is None:
            raise PipelineBuildError(f"Unknown step type {type(step).__name__}")
        self._steps[kind] = step
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
        return Pipeline(steps)

    @classmethod
    def from_config(cls, config: RunConfig, llm: LLMEngine | None = None) -> PipelineBuilder:
        """The production pieces for a run configuration (``llm``: an already checked engine)."""
        store = ArtifactStore(config.artifacts_dir)
        builder = cls()
        builder.add(OcrStep(ImagePreprocessor(config.max_image_side), RapidOcrEngine(), store))
        builder.add(RulesStep(RuleBasedParser()))
        if config.dedup:
            builder.add(DedupStep(config.dedup_max_distance))
        if config.uses_llm:
            engine = llm or LLMEngine.get_for_alias(config.llm_model)
            builder.add(
                LlmStep(
                    LLMStructurer(engine),
                    store,
                    config.llm_policy,
                    config.confidence_threshold,
                    config.llm_offline,
                )
            )
        return builder
