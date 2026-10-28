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
from pydantic import BaseModel, Field

from menu_poc.evaluation.golden import GoldenImage, Split, load_golden
from menu_poc.evaluation.matching import DEFAULT_THRESHOLD, MatchedPair, match_dishes
from menu_poc.io.writers import read_jsonl, write_json
from menu_poc.models import ImageResult

SPLITS: tuple[str, ...] = ("all", "dev", "test")


class ImageMetrics(BaseModel):
    image_id: str
    restaurant_id: str
    labeled: bool
    split: Split | None = None
    n_pred: int
    n_gold: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    duplicates: int = 0
    recall: float | None = None
    precision: float | None = None
    cost_usd: float = 0.0
    llm_used: bool = False
    llm_cached: bool = False
    llm_latency_s: float = 0.0
    time_s: float = 0.0
    ungrounded_dropped: int = 0
    duplicate_of: str | None = None  # duplicate photo: excluded from accuracy (counted once)
    error: str | None = None
    matched: list[MatchedPair] = Field(default_factory=list)
    duplicate_pairs: list[MatchedPair] = Field(default_factory=list)
    unmatched_gold: list[str] = Field(default_factory=list)
    false_positives: list[str] = Field(default_factory=list)


class AccuracyStats(BaseModel):
    n_images: int = 0
    n_gold: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    duplicates: int = 0
    micro_recall: float | None = None
    micro_precision: float | None = None
    macro_recall: float | None = None
    macro_precision: float | None = None


class RunMetrics(BaseModel):
    run: str
    mode: str | None = None
    llm_model: str | None = None
    llm_policy: str | None = None
    threshold: float = DEFAULT_THRESHOLD
    only_reviewed: bool = False
    n_images: int = 0
    n_labeled: int = 0
    n_duplicate_images: int = 0  # skipped photos (same menu as another image)
    unlabeled: list[str] = Field(default_factory=list)
    accuracy: dict[str, AccuracyStats] = Field(default_factory=dict)  # keys: all, dev, test
    cost_total_usd: float = 0.0
    cost_per_image_usd: float = 0.0
    llm_share: float = 0.0  # share of images where the LLM was used
    llm_cached_share: float = 0.0  # share of LLM images served from cache (no API call)
    time_total_s: float = 0.0
    time_mean_s: float = 0.0
    llm_latency_total_s: float = 0.0
    llm_latency_mean_s: float = 0.0  # over images where the LLM was used
    ungrounded_dropped: int = 0
    n_errors: int = 0
    errors: dict[str, str] = Field(default_factory=dict)
    images: list[ImageMetrics] = Field(default_factory=list)


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def image_time(result: ImageResult) -> float:
    t = result.timings_s
    return t["total"] if "total" in t else sum(t.values())


def evaluate_image(
    result: ImageResult, golden: GoldenImage | None, threshold: float = DEFAULT_THRESHOLD
) -> ImageMetrics:
    usage = result.llm_usage
    m = ImageMetrics(
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
        return m
    match = match_dishes([d.name for d in result.dishes], golden.dishes, threshold)
    m.split = golden.split
    m.n_gold = len(golden.dishes)
    m.tp, m.fp, m.fn, m.duplicates = match.tp, match.fp, match.fn, match.duplicates
    m.recall = _ratio(match.tp, m.n_gold)
    m.precision = _ratio(match.tp, match.tp + match.fp)
    m.matched, m.duplicate_pairs = match.matched, match.duplicate_pairs
    m.unmatched_gold, m.false_positives = match.unmatched_gold, match.false_positives
    return m


def accuracy_stats(images: list[ImageMetrics]) -> AccuracyStats:
    s = AccuracyStats(n_images=len(images))
    for m in images:
        s.n_gold += m.n_gold
        s.tp += m.tp
        s.fp += m.fp
        s.fn += m.fn
        s.duplicates += m.duplicates
    s.micro_recall = _ratio(s.tp, s.n_gold)
    s.micro_precision = _ratio(s.tp, s.tp + s.fp)
    s.macro_recall = _mean([m.recall for m in images if m.recall is not None])
    s.macro_precision = _mean([m.precision for m in images if m.precision is not None])
    return s


def aggregate(
    run: str,
    results: list[ImageResult],
    golden: dict[str, GoldenImage],
    config: dict[str, Any] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    only_reviewed: bool = False,
) -> RunMetrics:
    """Compute run metrics from results and an already loaded golden set."""
    config = config or {}
    images = [evaluate_image(r, golden.get(r.image_id), threshold) for r in results]
    # Duplicate photos would weigh their menu twice: each menu is scored once.
    labeled = [m for m in images if m.labeled and m.duplicate_of is None]
    llm_images = [m for m in images if m.llm_used]
    n = len(images)
    cost = sum(m.cost_usd for m in images)
    time_total = sum(m.time_s for m in images)
    llm_latency = sum(m.llm_latency_s for m in images)
    return RunMetrics(
        run=run,
        mode=config.get("mode"),
        llm_model=config.get("llm_model"),
        llm_policy=config.get("llm_policy"),
        threshold=threshold,
        only_reviewed=only_reviewed,
        n_images=n,
        n_labeled=len(labeled),
        n_duplicate_images=sum(m.duplicate_of is not None for m in images),
        unlabeled=[m.image_id for m in images if not m.labeled],
        accuracy={
            "all": accuracy_stats(labeled),
            "dev": accuracy_stats([m for m in labeled if m.split == "dev"]),
            "test": accuracy_stats([m for m in labeled if m.split == "test"]),
        },
        cost_total_usd=cost,
        cost_per_image_usd=cost / n if n else 0.0,
        llm_share=len(llm_images) / n if n else 0.0,
        llm_cached_share=(
            sum(m.llm_cached for m in llm_images) / len(llm_images) if llm_images else 0.0
        ),
        time_total_s=time_total,
        time_mean_s=time_total / n if n else 0.0,
        llm_latency_total_s=llm_latency,
        llm_latency_mean_s=llm_latency / len(llm_images) if llm_images else 0.0,
        ungrounded_dropped=sum(m.ungrounded_dropped for m in images),
        n_errors=sum(m.error is not None for m in images),
        errors={m.image_id: m.error for m in images if m.error is not None},
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
    threshold: float = DEFAULT_THRESHOLD,
    save: bool = True,
) -> RunMetrics:
    """Evaluate ``run_dir/results.jsonl`` against the golden set; writes ``metrics.json``."""
    run_dir = Path(run_dir)
    results = read_jsonl(run_dir / "results.jsonl", ImageResult)
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
