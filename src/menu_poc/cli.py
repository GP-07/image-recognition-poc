"""Command line: ``menu-poc run | eval | compare``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from menu_poc.cache.store import ArtifactStore
from menu_poc.config import RunConfig, load_config
from menu_poc.evaluation.report import compare_runs, run_report
from menu_poc.io.writers import write_jsonl
from menu_poc.llm.client import make_llm_client
from menu_poc.llm.grounding import GroundingVerifier
from menu_poc.llm.structurer import LLMStructurer
from menu_poc.ocr.rapid import RapidOcrEngine
from menu_poc.parsing.rules import RuleBasedParser
from menu_poc.pipeline import Pipeline, merge_previous, new_run_dir, select_images
from menu_poc.preprocess.image import ImagePreprocessor

DEFAULT_CONFIG = Path("config/default.yaml")
DEFAULT_GOLDEN = Path("data/golden")


def build_pipeline(config: RunConfig) -> Pipeline:
    structurer = verifier = None
    if config.uses_llm:
        structurer = LLMStructurer(make_llm_client(config.llm_model))
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


def _cmd_run(args: argparse.Namespace) -> int:
    if args.config != DEFAULT_CONFIG and not args.config.exists():
        print(f"Config file not found: {args.config}", file=sys.stderr)
        return 2
    config = load_config(
        args.config if args.config.exists() else None,
        input_path=args.input,
        mode=args.mode,
        llm_policy=args.llm_policy,
        llm_model=args.llm_model,
        llm_offline=True if args.offline else None,
        confidence_threshold=args.threshold,
        image_ids=args.image_ids.split(",") if args.image_ids else None,
        reuse_from=args.reuse_from,
        only_low_confidence=True if args.only_low_confidence else None,
    )
    refs = select_images(config)
    if not refs:
        print("No images selected.", file=sys.stderr)
        return 1
    pipeline = build_pipeline(config)
    run_dir = new_run_dir(config)
    print(f"Run {run_dir} — {len(refs)} images, mode={config.mode}")
    results = pipeline.run(refs, run_dir)
    if config.reuse_from is not None:
        write_jsonl(run_dir / "results.jsonl", merge_previous(config, results))
        print(f"Re-ran {len(results)} images; the rest carried over from {config.reuse_from}.")

    errors = [r for r in results if r.error]
    print(f"Done: {sum(len(r.dishes) for r in results)} dishes, {len(errors)} images with errors.")
    for r in errors:
        print(f"  {r.image_id}: {r.error}")
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
    run.add_argument("--mode", choices=["ocr", "ocr+llm"])
    run.add_argument("--llm-policy", choices=["fallback", "always"])
    run.add_argument("--llm-model", help="one model per run, e.g. claude-haiku-5-5")
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
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
