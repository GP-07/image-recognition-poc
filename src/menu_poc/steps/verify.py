"""Grounding: LLM items must appear in the OCR text; only then do they replace the rules output."""

from __future__ import annotations

from menu_poc.interfaces import Verifier
from menu_poc.steps.base import ImageContext, Step


class VerifyStep(Step):
    name = "verify"

    def __init__(self, verifier: Verifier) -> None:
        super().__init__()
        self.verifier = verifier

    def process(self, ctx: ImageContext) -> bool:
        assert ctx.structured is not None and ctx.ocr is not None and ctx.parsed is not None
        verified = self.verifier.verify(ctx.structured.dishes, ctx.ocr)
        result = ctx.result
        result.llm_used = True
        result.ungrounded_dropped = [d.name for d in verified.dropped]
        if verified.kept or not ctx.parsed.dishes:
            result.dishes = verified.kept
        else:  # an empty LLM answer should not wipe out what the rules found
            result.error = "llm: no grounded dishes, kept rules output"
        return True
