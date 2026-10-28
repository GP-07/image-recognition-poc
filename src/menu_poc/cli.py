"""Command line: ``menu-poc run | eval | compare``."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from menu_poc.cache.store import ArtifactStore
from menu_poc.config import PROJECT_ROOT, RunConfig, load_config
from menu_poc.evaluation.report import compare_runs, run_report
from menu_poc.io.writers import read_jsonl, write_items_csv, write_json, write_jsonl
from menu_poc.llm.client import make_llm_client
from menu_poc.llm.grounding import GroundingVerifier
from menu_poc.llm.structurer import LLMStructurer
from menu_poc.models import ImageResult
from menu_poc.ocr.rapid import RapidOcrEngine
from menu_poc.parsing.rules import RuleBasedParser
from menu_poc.pipeline import Pipeline, merge_previous, new_run_dir, select_images
from menu_poc.preprocess.image import ImagePreprocessor

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_GOLDEN = PROJECT_ROOT / "data" / "golden"


def build_pipeline(config: RunConfig, llm_client: Any | None = None) -> Pipeline:
    structurer = verifier = None
    if config.uses_llm:
        structurer = LLMStructurer(llm_client or make_llm_client(config.llm_model))
        verifier = GroundingVerifier()
    return Pipeline(
        config,
        ArtifactStore(config.artifacts_dir),
        ImagePreprocessor(config.max_image_side),
        RapidOcrEngine(),
        RuleBasedParser(),
        structurer,
        verifier,
    )


def resolve_llm(config: RunConfig, warnings: list[str]) -> tuple[RunConfig, Any | None]:
    """The LLM client for this run, or degrade to OCR + rules with a warning when it cannot run
    (no API key, local server down): a CI job without the secret still produces results."""
    if not config.uses_llm:
        return config, None
    client = make_llm_client(config.llm_model)
    reason = None if config.llm_offline else client.unavailable_reason()
    if reason is None:
        return config, client
    warnings.append(f"LLM step disabled: {reason}. Results come from OCR + rules only.")
    return config.model_copy(update={"mode": "ocr"}), None


def _cmd_run(args: argparse.Namespace) -> int:
    if args.config != DEFAULT_CONFIG and not args.config.exists():
        print(f"Config file not found: {args.config}", file=sys.stderr)
        return 2
    requested = load_config(
        args.config if args.config.exists() else None,
        input_path=args.input,
        runs_dir=args.output,
        artifacts_dir=args.artifacts,
        mode=args.mode,
        llm_policy=args.llm_policy,
        llm_model=args.llm_model,
        llm_offline=True if args.offline else None,
        confidence_threshold=args.threshold,
        image_ids=args.image_ids.split(",") if args.image_ids else None,
        reuse_from=args.reuse_from,
        only_low_confidence=True if args.only_low_confidence else None,
    )
    # Check the LLM first, before reading any image: a bad or missing key is reported at once.
    warnings: list[str] = []
    config, client = resolve_llm(requested, warnings)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)

    refs = select_images(requested)
    if not refs:
        print("No images selected.", file=sys.stderr)
        return 1
    pipeline = build_pipeline(config, client)
    run_dir = new_run_dir(config)  # effective config: reports must show what actually ran
    print(f"Run {run_dir} — {len(refs)} images, mode={config.mode}")
    started = datetime.now().isoformat(timespec="seconds")
    results = pipeline.run(refs, run_dir)
    if config.reuse_from is not None:
        write_jsonl(run_dir / "results.jsonl", merge_previous(config, results))
        print(f"Re-ran {len(results)} images; the rest carried over from {config.reuse_from}.")
    n_rows = write_items_csv(
        run_dir / "items.csv", read_jsonl(run_dir / "results.jsonl", ImageResult)
    )

    errors = {r.image_id: r.error for r in results if r.error}
    duplicates = {r.image_id: r.duplicate_of for r in results if r.duplicate_of}
    write_json(
        run_dir / "manifest.json",
        {
            "run": run_dir.name,
            "started": started,
            "finished": datetime.now().isoformat(timespec="seconds"),
            "requested_mode": requested.mode,
            "effective_mode": config.mode,
            "llm_model": config.llm_model if config.uses_llm else None,
            "n_images": len(results),
            "n_dishes": sum(len(r.dishes) for r in results),
            "duplicates": duplicates,
            "warnings": warnings,
            "errors": errors,
        },
    )
    print(
        f"Done: {sum(len(r.dishes) for r in results)} dishes, {len(duplicates)} duplicate photos "
        f"skipped, {len(errors)} images with errors, {len(warnings)} warnings."
    )
    for image_id, error in errors.items():
        print(f"  {image_id}: {error}")
    print(f"Items: {run_dir / 'items.csv'} ({n_rows} rows)")
    if args.golden.exists():
        report = run_report(run_dir, args.golden)
        (run_dir / "summary.md").write_text(report, encoding="utf-8")
        print(f"Report: {run_dir / 'summary.md'}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    report = run_report(args.run, args.golden, only_reviewed=args.only_reviewed)
    (args.run / "summary.md").write_text(report, encoding="utf-8")
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
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--input", type=Path, help="zip file or directory of images")
    run.add_argument("--output", type=Path, help="directory for run results (default: tmp/)")
    run.add_argument("--artifacts", type=Path, help="stage cache directory (default: artifacts/)")
    run.add_argument("--mode", choices=["ocr", "ocr+llm"])
    run.add_argument("--llm-policy", choices=["fallback", "always"])
    run.add_argument(
        "--llm-model", help="haiku | sonnet | opus | ollama (or an exact id); one model per run"
    )
    run.add_argument("--offline", action="store_true", help="only cached LLM answers")
    run.add_argument("--threshold", type=float, help="rules confidence below this -> LLM")
    run.add_argument("--image-ids", help="comma-separated '<restaurant>/<stem>' ids")
    run.add_argument("--reuse-from", type=Path, help="previous run directory")
    run.add_argument(
        "--only-low-confidence",
        action="store_true",
        help="with --reuse-from: only images whose rules confidence was below the threshold",
    )
    run.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    run.set_defaults(func=_cmd_run)

    ev = sub.add_parser("eval", help="evaluate a run against the golden set")
    ev.add_argument("run", type=Path)
    ev.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    ev.add_argument("--only-reviewed", action="store_true")
    ev.set_defaults(func=_cmd_eval)

    cmp_ = sub.add_parser("compare", help="compare several runs")
    cmp_.add_argument("runs", type=Path, nargs="+")
    cmp_.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
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
