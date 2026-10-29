"""Markdown reports: one run (summary, per image, error analysis) and a comparison of runs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from menu_poc.constants import (
    IMAGE_SET_LABELS,
    MATCH_THRESHOLD,
    SPLIT_LABELS,
    SPLITS,
    SUMMARY_LEGEND,
    TOP_ERRORS,
)
from menu_poc.evaluation.metrics import evaluate_run
from menu_poc.models import ImageMetrics, RunMetrics


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def _usd(value: float) -> str:
    return f"${value:.4f}"


def _sec(value: float) -> str:
    return f"{value:.2f}s"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows]
    return lines


def _llm_model(metrics: RunMetrics) -> str:
    return metrics.llm_model if metrics.mode == "ocr+llm" and metrics.llm_model else "-"


def render_run(metrics: RunMetrics) -> str:
    """Markdown for one evaluated run."""
    out = [f"# Run `{metrics.run}`", ""]
    reviewed_only = " · reviewed references only" if metrics.only_reviewed else ""
    out.append(
        f"Mode: `{metrics.mode or '?'}` · LLM: `{_llm_model(metrics)}` · "
        f"policy: `{metrics.llm_policy or '-'}` · "
        f"name match threshold: {metrics.threshold:g}{reviewed_only}"
    )
    out += ["", "## Summary", "", *SUMMARY_LEGEND, ""]
    rows = []
    for split in SPLITS:
        stats = metrics.accuracy.get(split)
        if stats is None or stats.n_images == 0:
            continue
        rows.append(
            [
                SPLIT_LABELS[split],
                str(stats.n_images),
                str(stats.n_gold),
                str(stats.tp),
                str(stats.fp),
                str(stats.fn),
                str(stats.duplicates),
                _pct(stats.micro_recall),
                _pct(stats.micro_precision),
            ]
        )
    header = ["photos", "count", "reference items", "found correctly", "wrong", "missed"]
    header += ["repeats", "recall", "precision"]
    out += _table(header, rows) if rows else ["No photo has reference answers."]
    out += [
        "",
        f"- Photos: {metrics.n_images} ({metrics.n_labeled} with reference answers, "
        f"{len(metrics.unlabeled)} without, "
        f"{metrics.n_duplicate_images} copies of another photo, not scored)",
        f"- LLM used on {_pct(metrics.llm_share)} of the photos "
        f"({_pct(metrics.llm_cached_share)} of them answered from the cache)",
        f"- Cost: {_usd(metrics.cost_total_usd)} total, "
        f"{_usd(metrics.cost_per_image_usd)} per photo",
        f"- Time: {_sec(metrics.time_total_s)} total, {_sec(metrics.time_mean_s)} per photo; "
        f"LLM {_sec(metrics.llm_latency_total_s)} total, "
        f"{_sec(metrics.llm_latency_mean_s)} per LLM call",
        "- LLM items discarded because they are not in the photo's text (likely invented): "
        f"{metrics.ungrounded_dropped}",
        f"- Photos with errors: {metrics.n_errors}",
    ]
    out += [f"  - `{image_id}`: {err}" for image_id, err in metrics.errors.items()]

    out += ["", "## Per photo", ""]
    rows = [
        [
            f"`{image.image_id}`",
            _image_set(image),
            str(image.n_pred),
            str(image.n_gold) if image.labeled else "-",
            str(image.tp) if image.labeled else "-",
            str(image.fp) if image.labeled else "-",
            str(image.fn) if image.labeled else "-",
            str(image.duplicates) if image.labeled else "-",
            _pct(image.recall),
            _pct(image.precision),
            "yes" if image.llm_used else "no",
            _usd(image.cost_usd),
            _sec(image.time_s),
            "yes" if image.error else "",
        ]
        for image in metrics.images
    ]
    header = ["photo", "set", "items found", "reference items", "correct", "wrong", "missed"]
    header += ["repeats", "recall", "precision", "LLM used", "cost", "time", "error"]
    out += _table(header, rows)

    errors = [
        image
        for image in metrics.images
        if image.labeled and (image.unmatched_gold or image.false_positives)
    ]
    if errors:
        out += ["", "## What went wrong, per photo", ""]
        for image in errors:
            out.append(f"### `{image.image_id}` (recall {_pct(image.recall)})")
            if image.unmatched_gold:
                out.append(f"- Missed: {_listing(image.unmatched_gold)}")
            if image.false_positives:
                out.append(f"- Wrong: {_listing(image.false_positives)}")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def _image_set(image: ImageMetrics) -> str:
    if image.duplicate_of is not None:
        return "copy, not scored"
    return IMAGE_SET_LABELS[image.split] if image.split else "no reference"


def _listing(items: list[str]) -> str:
    shown = ", ".join(f'"{item}"' for item in items[:TOP_ERRORS])
    extra = len(items) - TOP_ERRORS
    return shown + (f" (+{extra} more)" if extra > 0 else "")


def render_comparison(metrics: Sequence[RunMetrics]) -> str:
    """One row per run with the headline numbers."""
    header = ["run", "mode", "LLM model", "policy", "recall dev", "recall test", "recall all"]
    header += ["precision", "% LLM", "cost total", "cost/image", "time/image", "LLM latency"]
    rows = []
    for run in metrics:
        acc = run.accuracy
        rows.append(
            [
                f"`{run.run}`",
                run.mode or "?",
                _llm_model(run),
                run.llm_policy if run.mode == "ocr+llm" and run.llm_policy else "-",
                _pct(acc["dev"].micro_recall if "dev" in acc else None),
                _pct(acc["test"].micro_recall if "test" in acc else None),
                _pct(acc["all"].micro_recall if "all" in acc else None),
                _pct(acc["all"].micro_precision if "all" in acc else None),
                _pct(run.llm_share),
                _usd(run.cost_total_usd),
                _usd(run.cost_per_image_usd),
                _sec(run.time_mean_s),
                _sec(run.llm_latency_mean_s),
            ]
        )
    lines = _table(header, rows)
    lines += ["", "Recall and precision are micro-averaged over labeled images."]
    return "\n".join(lines) + "\n"


def run_report(
    run_dir: Path,
    golden_dir: Path,
    *,
    only_reviewed: bool = False,
    threshold: float = MATCH_THRESHOLD,
) -> str:
    """Evaluate a run (also writing ``metrics.json``) and return its Markdown report."""
    metrics = evaluate_run(run_dir, golden_dir, only_reviewed=only_reviewed, threshold=threshold)
    return render_run(metrics)


def compare_runs(
    run_dirs: Sequence[Path],
    golden_dir: Path,
    *,
    only_reviewed: bool = False,
    threshold: float = MATCH_THRESHOLD,
) -> str:
    """Markdown table comparing several runs against the same golden set."""
    metrics = [
        evaluate_run(
            run_dir, golden_dir, only_reviewed=only_reviewed, threshold=threshold, save=False
        )
        for run_dir in run_dirs
    ]
    return render_comparison(metrics)
