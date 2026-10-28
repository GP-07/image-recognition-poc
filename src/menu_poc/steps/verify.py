"""Grounding: LLM items must appear in the OCR text; only then do they replace the rules output."""

from __future__ import annotations

from menu_poc.data_classes import ImageContext
from menu_poc.interfaces import Verifier
from menu_poc.steps.base import Step, require


class VerifyStep(Step):
    name = "verify"

    def __init__(self, verifier: Verifier) -> None:
        super().__init__()
        self.verifier = verifier

    def process(self, ctx: ImageContext) -> bool:
        structured = require(ctx.structured, "LLM output")
        parsed = require(ctx.parsed, "rules output")
        verified = self.verifier.verify(structured.dishes, require(ctx.ocr, "OCR output"))
        result = ctx.result
        result.llm_used = True
        result.ungrounded_dropped = [dish.name for dish in verified.dropped]
        if verified.kept or not parsed.dishes:
            result.dishes = verified.kept
        else:  # an empty LLM answer should not wipe out what the rules found
            result.error = "llm: no grounded dishes, kept rules output"
        return True
