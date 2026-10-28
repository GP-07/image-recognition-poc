"""Evaluation as a pipeline post-processor: every run is scored against the golden set."""

from __future__ import annotations

from pathlib import Path

from menu_poc.constants import SUMMARY_FILE
from menu_poc.data_classes import RunOutcome
from menu_poc.evaluation.report import run_report


class GoldenSetEvaluation:
    """Writes ``summary.md`` and ``metrics.json``, the same report as ``menu-poc eval``."""

    name = "golden_set_evaluation"

    def __init__(self, golden_dir: Path) -> None:
        self.golden_dir = golden_dir

    def run(self, outcome: RunOutcome) -> list[Path]:
        summary = outcome.run_dir / SUMMARY_FILE
        summary.write_text(run_report(outcome.run_dir, self.golden_dir), encoding="utf-8")
        return [summary]
