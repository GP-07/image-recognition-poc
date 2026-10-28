"""Markdown reports: one run (summary, per image, error analysis) and a comparison of runs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from menu_poc.evaluation.matching import DEFAULT_THRESHOLD
from menu_poc.evaluation.metrics import SPLITS, RunMetrics, evaluate_run

TOP_ERRORS = 5  # unmatched gold dishes / false positives listed per image


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _usd(x: float) -> str:
    return f"${x:.4f}"


def _sec(x: float) -> str:
    return f"{x:.2f}s"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _llm_model(m: RunMetrics) -> str:
    return m.llm_model if m.mode == "ocr+llm" and m.llm_model else "-"


def render_run(m: RunMetrics) -> str:
    """Markdown for one evaluated run."""
    out = [f"# Run `{m.run}`", ""]
    out.append(
        f"Mode: `{m.mode or '?'}` · LLM: `{_llm_model(m)}` · policy: `{m.llm_policy or '-'}` · "
        f"match threshold: {m.threshold:g}" + (" · reviewed golden only" if m.only_reviewed else "")
    )
    out += ["", "## Summary", ""]
    rows = []
    for split in SPLITS:
        s = m.accuracy.get(split)
        if s is None or s.n_images == 0:
            continue
        rows.append(
            [
                split,
                str(s.n_images),
                str(s.n_gold),
                str(s.tp),
                str(s.fp),
                str(s.fn),
                str(s.duplicates),
                _pct(s.micro_recall),
                _pct(s.macro_recall),
                _pct(s.micro_precision),
                _pct(s.macro_precision),
            ]
        )
    header = ["split", "images", "gold", "TP", "FP", "FN", "dup", "recall (micro)"]
    header += ["recall (macro)", "precision (micro)", "precision (macro)"]
    out += _table(header, rows) if rows else ["No labeled images."]
    out += [
        "",
        f"- Images: {m.n_images} ({m.n_labeled} labeled, {len(m.unlabeled)} unlabeled)",
        f"- LLM used: {_pct(m.llm_share)} of images ({_pct(m.llm_cached_share)} of them cached)",
        f"- Cost: {_usd(m.cost_total_usd)} total, {_usd(m.cost_per_image_usd)} per image",
        f"- Time: {_sec(m.time_total_s)} total, {_sec(m.time_mean_s)} per image; "
        f"LLM latency {_sec(m.llm_latency_total_s)} total, "
        f"{_sec(m.llm_latency_mean_s)} per LLM call",
        f"- Ungrounded dishes dropped: {m.ungrounded_dropped}",
        f"- Errors: {m.n_errors}",
    ]
    out += [f"  - `{image_id}`: {err}" for image_id, err in m.errors.items()]

    out += ["", "## Per image", ""]
    rows = [
        [
            f"`{i.image_id}`",
            i.split or "unlabeled",
            str(i.n_pred),
            str(i.n_gold) if i.labeled else "-",
            str(i.tp) if i.labeled else "-",
            str(i.fp) if i.labeled else "-",
            str(i.fn) if i.labeled else "-",
            str(i.duplicates) if i.labeled else "-",
            _pct(i.recall),
            _pct(i.precision),
            "yes" if i.llm_used else "no",
            _usd(i.cost_usd),
            _sec(i.time_s),
            "error" if i.error else "",
        ]
        for i in m.images
    ]
    header = ["image", "split", "pred", "gold", "TP", "FP", "FN", "dup", "recall", "precision"]
    header += ["LLM", "cost", "time", ""]
    out += _table(header, rows)

    errors = [i for i in m.images if i.labeled and (i.unmatched_gold or i.false_positives)]
    if errors:
        out += ["", "## Error analysis", ""]
        for i in errors:
            out.append(f"### `{i.image_id}` (recall {_pct(i.recall)})")
            if i.unmatched_gold:
                out.append(f"- Missed gold dishes: {_listing(i.unmatched_gold)}")
            if i.false_positives:
                out.append(f"- False positives: {_listing(i.false_positives)}")
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def _listing(items: list[str]) -> str:
    shown = ", ".join(f'"{x}"' for x in items[:TOP_ERRORS])
    extra = len(items) - TOP_ERRORS
    return shown + (f" (+{extra} more)" if extra > 0 else "")


def render_comparison(metrics: Sequence[RunMetrics]) -> str:
    """One row per run with the headline numbers."""
    header = ["run", "mode", "LLM model", "policy", "recall dev", "recall test", "recall all"]
    header += ["precision", "% LLM", "cost total", "cost/image", "time/image", "LLM latency"]
    rows = []
    for m in metrics:
        acc = m.accuracy
        rows.append(
            [
                f"`{m.run}`",
                m.mode or "?",
                _llm_model(m),
                m.llm_policy if m.mode == "ocr+llm" and m.llm_policy else "-",
                _pct(acc["dev"].micro_recall if "dev" in acc else None),
                _pct(acc["test"].micro_recall if "test" in acc else None),
                _pct(acc["all"].micro_recall if "all" in acc else None),
                _pct(acc["all"].micro_precision if "all" in acc else None),
                _pct(m.llm_share),
                _usd(m.cost_total_usd),
                _usd(m.cost_per_image_usd),
                _sec(m.time_mean_s),
                _sec(m.llm_latency_mean_s),
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
    threshold: float = DEFAULT_THRESHOLD,
) -> str:
    """Evaluate a run (also writing ``metrics.json``) and return its Markdown report."""
    metrics = evaluate_run(run_dir, golden_dir, only_reviewed=only_reviewed, threshold=threshold)
    return render_run(metrics)


def compare_runs(
    run_dirs: Sequence[Path],
    golden_dir: Path,
    *,
    only_reviewed: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
) -> str:
    """Markdown table comparing several runs against the same golden set."""
    metrics = [
        evaluate_run(d, golden_dir, only_reviewed=only_reviewed, threshold=threshold, save=False)
        for d in run_dirs
    ]
    return render_comparison(metrics)
