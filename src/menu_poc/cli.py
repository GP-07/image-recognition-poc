"""Command line: ``menu-poc run | eval | compare``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from menu_poc.config import load_config
from menu_poc.constants import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_GOLDEN_DIR,
    PROJECT_ROOT,
    SUMMARY_FILE,
)
from menu_poc.evaluation.report import compare_runs, run_report
from menu_poc.pipeline import PipelineBuilder
from menu_poc.steps import InputValidationError


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(
        DEFAULT_CONFIG_PATH, input_path=args.input, runs_dir=args.output, llm_model=args.llm_model
    )
    pipeline = PipelineBuilder.from_config(config).build()
    try:
        refs, warnings = pipeline.validate_inputs(config.input_path)
    except InputValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    print(f"Processing {len(refs)} images, steps: {' → '.join(pipeline.step_names)}")
    outcome = pipeline.run(refs, config)

    print(
        f"Done: {outcome.n_items} items, {len(outcome.duplicates)} duplicate photos skipped, "
        f"{len(outcome.errors)} images with errors, {len(outcome.warnings)} warnings."
    )
    for image_id, error in outcome.errors.items():
        print(f"  {image_id}: {error}")
    print(f"Run: {outcome.run_dir}")
    print(f"Items: {outcome.items_path}")
    for report in outcome.reports:
        print(f"Report: {report}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    report = run_report(args.run, args.golden, only_reviewed=args.only_reviewed)
    (args.run / SUMMARY_FILE).write_text(report, encoding="utf-8")
    print(report)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    table = compare_runs(args.runs, args.golden, only_reviewed=args.only_reviewed)
    if args.out:
        args.out.write_text(table, encoding="utf-8")
    print(table)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="menu-poc", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="process images (results go to a new run directory)")
    run.add_argument("--input", type=Path, help="zip file or directory of images")
    run.add_argument("--output", type=Path, help="directory for run results (default: tmp/)")
    run.add_argument(
        "--llm-model", help="haiku | sonnet | opus | ollama (or an exact id); one model per run"
    )
    run.set_defaults(func=_cmd_run)

    ev = sub.add_parser("eval", help="evaluate a run against the golden set")
    ev.add_argument("run", type=Path)
    ev.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_DIR)
    ev.add_argument("--only-reviewed", action="store_true")
    ev.set_defaults(func=_cmd_eval)

    cmp_ = sub.add_parser("compare", help="compare several runs")
    cmp_.add_argument("runs", type=Path, nargs="+")
    cmp_.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_DIR)
    cmp_.add_argument("--only-reviewed", action="store_true")
    cmp_.add_argument("--out", type=Path)
    cmp_.set_defaults(func=_cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")  # API keys; real environment variables win
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
