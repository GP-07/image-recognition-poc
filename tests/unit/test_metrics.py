import json

import pytest

from menu_poc.evaluation.metrics import evaluate_run
from menu_poc.io.writers import write_jsonl, write_yaml
from menu_poc.models import Dish, ImageResult, LLMUsage


def _golden(golden_dir, image_id, split, dishes, reviewed=True):
    write_yaml(
        golden_dir / (image_id.replace("/", "__") + ".yaml"),
        {"image_id": image_id, "split": split, "reviewed": reviewed, "dishes": dishes},
    )


def _result(image_id, names, cost=None, **kw):
    usage = LLMUsage(model="m", cost_usd=cost, latency_s=2.0) if cost is not None else None
    return ImageResult(
        image_id=image_id,
        restaurant_id=image_id.split("/")[0],
        dishes=[Dish(name=name, origin="rules") for name in names],
        llm_used=usage is not None,
        llm_usage=usage,
        **kw,
    )


@pytest.fixture
def run(tmp_path):
    golden = tmp_path / "golden"
    _golden(
        golden,
        "r1/a",
        "dev",
        [{"name": "Lamb Curry", "aliases": ["Cordero al curry"]}, {"name": "Naan"}],
    )
    _golden(
        golden,
        "r2/b",
        "test",
        [
            {"name": "Pizza Margherita"},
            {"name": "Tiramisu"},
            {"name": "Calzone"},
            {"name": "Lasagna"},
        ],
        reviewed=False,
    )
    run_dir = tmp_path / "run1"
    write_yaml(
        run_dir / "config.yaml", {"mode": "ocr+llm", "llm_model": "m", "llm_policy": "fallback"}
    )
    write_jsonl(
        run_dir / "results.jsonl",
        [
            _result(
                "r1/a",
                ["Lamb Curry", "Cordero al curry", "Naan"],
                cost=0.01,
                timings_s={"ocr": 1.0, "llm": 2.0},
            ),
            _result(
                "r2/b",
                ["Pizza Margherita", "Bebidas"],
                ungrounded_dropped=["Ghost"],
                timings_s={"ocr": 1.0, "parse": 0.5, "total": 2.0},
            ),
            _result("r3/c", ["Something"], cost=0.03, timings_s={"total": 3.0}, error="boom"),
        ],
    )
    return run_dir, golden


def test_per_image_and_split_metrics(run):
    run_dir, golden = run
    metrics = evaluate_run(run_dir, golden)
    first, second, _ = metrics.images
    assert (first.tp, first.fp, first.fn, first.duplicates) == (2, 0, 0, 1)
    assert (first.recall, first.precision) == (1.0, 1.0)
    assert (second.tp, second.fp, second.fn, second.recall, second.precision) == (
        1,
        1,
        3,
        0.25,
        0.5,
    )
    assert metrics.accuracy["dev"].micro_recall == 1.0
    assert metrics.accuracy["test"].micro_recall == 0.25
    assert metrics.accuracy["all"].micro_recall == pytest.approx(3 / 6)
    assert metrics.accuracy["all"].macro_recall == pytest.approx((1.0 + 0.25) / 2)
    assert metrics.accuracy["all"].micro_precision == pytest.approx(3 / 4)


def test_unlabeled_image_excluded_from_accuracy_but_counted_in_cost(run):
    run_dir, golden = run
    metrics = evaluate_run(run_dir, golden)
    assert metrics.unlabeled == ["r3/c"]
    assert metrics.n_images == 3 and metrics.n_labeled == 2
    assert metrics.accuracy["all"].n_images == 2
    assert not metrics.images[2].labeled and metrics.images[2].recall is None


def test_cost_latency_and_counters(run):
    run_dir, golden = run
    metrics = evaluate_run(run_dir, golden)
    assert metrics.cost_total_usd == pytest.approx(0.04)
    assert metrics.cost_per_image_usd == pytest.approx(0.04 / 3)
    assert metrics.llm_share == pytest.approx(2 / 3)
    assert metrics.time_total_s == pytest.approx(3.0 + 2.0 + 3.0)  # "total" wins over the sum
    assert metrics.llm_latency_total_s == 4.0 and metrics.llm_latency_mean_s == 2.0
    assert metrics.ungrounded_dropped == 1
    assert metrics.n_errors == 1 and metrics.errors == {"r3/c": "boom"}
    assert (metrics.mode, metrics.llm_model, metrics.llm_policy) == ("ocr+llm", "m", "fallback")


def test_only_reviewed_and_metrics_json(run):
    run_dir, golden = run
    metrics = evaluate_run(run_dir, golden, only_reviewed=True)
    assert metrics.unlabeled == ["r2/b", "r3/c"]
    assert metrics.accuracy["test"].n_images == 0 and metrics.accuracy["test"].micro_recall is None
    saved = json.loads((run_dir / "metrics.json").read_text())
    assert saved["n_labeled"] == 1 and saved["images"][0]["image_id"] == "r1/a"
