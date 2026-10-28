"""The LLM structures the OCR text, only when the policy asks for it. Answers are cached."""

from __future__ import annotations

from menu_poc.cache.store import ArtifactStore, make_key
from menu_poc.constants import DEFAULT_CONFIDENCE_THRESHOLD
from menu_poc.data_classes import ImageContext
from menu_poc.interfaces import Structurer
from menu_poc.models import StructureResult
from menu_poc.steps.base import Step, require
from menu_poc.typings import LLMPolicy


class LlmStep(Step):
    name = "llm"

    def __init__(
        self,
        structurer: Structurer,
        store: ArtifactStore,
        policy: LLMPolicy = "fallback",
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        offline: bool = False,
    ) -> None:
        super().__init__()
        self.structurer = structurer
        self.store = store
        self.policy = policy
        self.threshold = threshold
        self.offline = offline  # only cached answers, never call the API

    def needed(self, ctx: ImageContext) -> bool:
        """``always``: every image; ``fallback``: only when the rules are not confident."""
        parsed = require(ctx.parsed, "rules output")
        return self.policy == "always" or parsed.confidence < self.threshold

    def process(self, ctx: ImageContext) -> bool:
        if not self.needed(ctx):
            return False  # the rules output is the answer
        ocr = require(ctx.ocr, "OCR output")
        key = make_key("llm", ctx.ocr_key, self.structurer.fingerprint())
        cached = self.store.get("llm", key)
        if cached is not None:
            structured = StructureResult.model_validate(cached)
            structured.usage = structured.usage.model_copy(update={"cached": True})
        elif self.offline:
            ctx.result.error = "llm: cache miss in offline mode, kept rules output"
            return False
        else:
            try:
                structured = self.structurer.structure(ocr)
            except Exception as exc:
                # A billed but unusable answer (refusal, truncation) still counts as cost.
                ctx.result.llm_usage = getattr(exc, "usage", None)
                raise
            self.store.put("llm", key, structured.model_dump(mode="json"))

        ctx.structured = structured
        ctx.result.llm_usage = structured.usage
        ctx.result.llm_prompt_version = structured.prompt_version
        ctx.result.timings_s["llm"] = structured.usage.latency_s
        return True
