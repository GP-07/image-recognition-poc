"""Markdown reports: one run (summary, per image, error analysis) and a comparison of runs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from menu_poc.constants import MATCH_THRESHOLD, SPLITS, TOP_ERRORS
from menu_poc.evaluation.metrics import evaluate_run
from menu_poc.models import RunMetrics


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
    reviewed_only = " · reviewed golden only" if metrics.only_reviewed else ""
    out.append(
        f"Mode: `{metrics.mode or '?'}` · LLM: `{_llm_model(metrics)}` · "
        f"policy: `{metrics.llm_policy or '-'}` · "
        f"match threshold: {metrics.threshold:g}{reviewed_only}"
    )
    out += ["", "## Summary", ""]
    rows = []
    for split in SPLITS:
        stats = metrics.accuracy.get(split)
        if stats is None or stats.n_images == 0:
            continue
        rows.append(
            [
                split,
                str(stats.n_images),
                str(stats.n_gold),
                str(stats.tp),
                str(stats.fp),
                str(stats.fn),
                str(stats.duplicates),
                _pct(stats.micro_recall),
                _pct(stats.macro_recall),
                _pct(stats.micro_precision),
                _pct(stats.macro_precision),
            ]
        )
    header = ["split", "images", "gold", "TP", "FP", "FN", "dup", "recall (micro)"]
    header += ["recall (macro)", "precision (micro)", "precision (macro)"]
    out += _table(header, rows) if rows else ["No labeled images."]
    out += [
        "",
        f"- Images: {metrics.n_images} ({metrics.n_labeled} labeled, "
        f"{len(metrics.unlabeled)} unlabeled, "
        f"{metrics.n_duplicate_images} duplicate photos skipped and not scored)",
        f"- LLM used: {_pct(metrics.llm_share)} of images "
        f"({_pct(metrics.llm_cached_share)} of them cached)",
        f"- Cost: {_usd(metrics.cost_total_usd)} total, "
        f"{_usd(metrics.cost_per_image_usd)} per image",
        f"- Time: {_sec(metrics.time_total_s)} total, {_sec(metrics.time_mean_s)} per image; "
        f"LLM latency {_sec(metrics.llm_latency_total_s)} total, "
        f"{_sec(metrics.llm_latency_mean_s)} per LLM call",
        f"- Ungrounded dishes dropped: {metrics.ungrounded_dropped}",
        f"- Errors: {metrics.n_errors}",
    ]
    out += [f"  - `{image_id}`: {err}" for image_id, err in metrics.errors.items()]

    out += ["", "## Per image", ""]
    rows = [
        [
            f"`{image.image_id}`",
            image.split or "unlabeled",
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
            "error" if image.error else "",
        ]
        for image in metrics.images
    ]
    header = ["image", "split", "pred", "gold", "TP", "FP", "FN", "dup", "recall", "precision"]
    header += ["LLM", "cost", "time", ""]
    out += _table(header, rows)

    errors = [
        image
        for image in metrics.images
        if image.labeled and (image.unmatched_gold or image.false_positives)
    ]
    if errors:
        out += ["", "## Error analysis", ""]
        for image in errors:
            out.append(f"### `{image.image_id}` (recall {_pct(image.recall)})")
            if image.unmatched_gold:
                out.append(f"- Missed gold dishes: {_listing(image.unmatched_gold)}")
            if image.false_positives:
                out.append(f"- False positives: {_listing(image.false_positives)}")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


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
