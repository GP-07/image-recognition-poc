"""Rule-based parsing of the OCR lines: items plus a confidence score."""

from __future__ import annotations

import time

from menu_poc.data_classes import ImageContext
from menu_poc.interfaces import Parser
from menu_poc.steps.base import Step, require


class RulesStep(Step):
    name = "rules"

    def __init__(self, parser: Parser) -> None:
        super().__init__()
        self.parser = parser

    def process(self, ctx: ImageContext) -> bool:
        ocr = require(ctx.ocr, "OCR output")
        start = time.perf_counter()
        parsed = self.parser.parse(ocr)
        result = ctx.result
        result.timings_s["rules"] = time.perf_counter() - start
        result.dishes = parsed.dishes
        result.rules_confidence = parsed.confidence
        result.rules_dish_count = len(parsed.dishes)
        result.rules_signals = parsed.signals
        ctx.parsed = parsed
        return True
