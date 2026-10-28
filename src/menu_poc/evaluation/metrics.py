"""Per-image and per-run metrics: recognition rate (recall), precision as control, cost, latency.

Definitions:
- recall = TP / gold dishes (the recognition rate); precision = TP / (TP + FP). Duplicates
  (translations of an already matched dish) count in neither.
- micro = pooled counts over images; macro = mean of the per-image values (images with no gold
  dishes are skipped for macro recall, images with no TP+FP for macro precision).
- Images without a golden file are "unlabeled": excluded from accuracy, included in cost/latency.
- Cost = sum of ``llm_usage.cost_usd``. Image time = ``timings_s["total"]`` when present, else the
  sum of all ``timings_s`` values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from menu_poc.constants import MATCH_THRESHOLD, RESULTS_FILE
from menu_poc.evaluation.golden import load_golden
from menu_poc.evaluation.matching import match_dishes
from menu_poc.io.writers import read_jsonl, write_json
from menu_poc.models import AccuracyStats, GoldenImage, ImageMetrics, ImageResult, RunMetrics


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def image_time(result: ImageResult) -> float:
    timings = result.timings_s
    return timings["total"] if "total" in timings else sum(timings.values())


def evaluate_image(
    result: ImageResult, golden: GoldenImage | None, threshold: float = MATCH_THRESHOLD
) -> ImageMetrics:
    usage = result.llm_usage
    metrics = ImageMetrics(
        image_id=result.image_id,
        restaurant_id=result.restaurant_id,
        labeled=golden is not None,
        n_pred=len(result.dishes),
        cost_usd=usage.cost_usd if usage else 0.0,
        llm_used=result.llm_used,
        llm_cached=bool(usage and usage.cached),
        llm_latency_s=usage.latency_s if usage else 0.0,
        time_s=image_time(result),
        ungrounded_dropped=len(result.ungrounded_dropped),
        duplicate_of=result.duplicate_of,
        error=result.error,
    )
    if golden is None:
        return metrics
    match = match_dishes([dish.name for dish in result.dishes], golden.dishes, threshold)
    metrics.split = golden.split
    metrics.n_gold = len(golden.dishes)
    metrics.tp, metrics.fp, metrics.fn = match.tp, match.fp, match.fn
    metrics.duplicates = match.duplicates
    metrics.recall = _ratio(match.tp, metrics.n_gold)
    metrics.precision = _ratio(match.tp, match.tp + match.fp)
    metrics.matched, metrics.duplicate_pairs = match.matched, match.duplicate_pairs
    metrics.unmatched_gold = match.unmatched_gold
    metrics.false_positives = match.false_positives
    return metrics


def accuracy_stats(images: list[ImageMetrics]) -> AccuracyStats:
    stats = AccuracyStats(n_images=len(images))
    for image in images:
        stats.n_gold += image.n_gold
        stats.tp += image.tp
        stats.fp += image.fp
        stats.fn += image.fn
        stats.duplicates += image.duplicates
    stats.micro_recall = _ratio(stats.tp, stats.n_gold)
    stats.micro_precision = _ratio(stats.tp, stats.tp + stats.fp)
    stats.macro_recall = _mean([image.recall for image in images if image.recall is not None])
    stats.macro_precision = _mean(
        [image.precision for image in images if image.precision is not None]
    )
    return stats


def aggregate(
    run: str,
    results: list[ImageResult],
    golden: dict[str, GoldenImage],
    config: dict[str, Any] | None = None,
    threshold: float = MATCH_THRESHOLD,
    only_reviewed: bool = False,
) -> RunMetrics:
    """Compute run metrics from results and an already loaded golden set."""
    config = config or {}
    images = [evaluate_image(result, golden.get(result.image_id), threshold) for result in results]
    # Duplicate photos would weigh their menu twice: each menu is scored once.
    labeled = [image for image in images if image.labeled and image.duplicate_of is None]
    llm_images = [image for image in images if image.llm_used]
    n_images = len(images)
    cost = sum(image.cost_usd for image in images)
    time_total = sum(image.time_s for image in images)
    llm_latency = sum(image.llm_latency_s for image in images)
    return RunMetrics(
        run=run,
        # Older runs stored a "mode"; newer ones use the LLM step iff an LLM model is set.
        mode=config.get("mode") or ("ocr+llm" if config.get("llm_model") else "ocr"),
        llm_model=config.get("llm_model"),
        llm_policy=config.get("llm_policy"),
        threshold=threshold,
        only_reviewed=only_reviewed,
        n_images=n_images,
        n_labeled=len(labeled),
        n_duplicate_images=sum(image.duplicate_of is not None for image in images),
        unlabeled=[image.image_id for image in images if not image.labeled],
        accuracy={
            "all": accuracy_stats(labeled),
            "dev": accuracy_stats([image for image in labeled if image.split == "dev"]),
            "test": accuracy_stats([image for image in labeled if image.split == "test"]),
        },
        cost_total_usd=cost,
        cost_per_image_usd=cost / n_images if n_images else 0.0,
        llm_share=len(llm_images) / n_images if n_images else 0.0,
        llm_cached_share=(
            sum(image.llm_cached for image in llm_images) / len(llm_images) if llm_images else 0.0
        ),
        time_total_s=time_total,
        time_mean_s=time_total / n_images if n_images else 0.0,
        llm_latency_total_s=llm_latency,
        llm_latency_mean_s=llm_latency / len(llm_images) if llm_images else 0.0,
        ungrounded_dropped=sum(image.ungrounded_dropped for image in images),
        n_errors=sum(image.error is not None for image in images),
        errors={image.image_id: image.error for image in images if image.error is not None},
        images=images,
    )


def load_run_config(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / "config.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def evaluate_run(
    run_dir: Path,
    golden_dir: Path,
    *,
    only_reviewed: bool = False,
    threshold: float = MATCH_THRESHOLD,
    save: bool = True,
) -> RunMetrics:
    """Evaluate ``run_dir/results.jsonl`` against the golden set; writes ``metrics.json``."""
    run_dir = Path(run_dir)
    results = read_jsonl(run_dir / RESULTS_FILE, ImageResult)
    golden = load_golden(golden_dir, only_reviewed=only_reviewed)
    metrics = aggregate(
        run_dir.name, results, golden, load_run_config(run_dir), threshold, only_reviewed
    )
    if save:
        save_metrics(metrics, run_dir)
    return metrics


def save_metrics(metrics: RunMetrics, run_dir: Path) -> Path:
    path = Path(run_dir) / "metrics.json"
    write_json(path, metrics.model_dump(mode="json"))
    return path
