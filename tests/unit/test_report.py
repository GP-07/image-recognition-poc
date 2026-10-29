from menu_poc.evaluation.report import compare_runs, run_report
from menu_poc.io.writers import write_jsonl, write_yaml
from menu_poc.models import Dish, ImageResult, LLMUsage


def _make_run(run_dir, config, names, cost=None):
    write_yaml(run_dir / "config.yaml", config)
    usage = LLMUsage(model="claude-haiku-5-5", cost_usd=cost) if cost is not None else None
    write_jsonl(
        run_dir / "results.jsonl",
        [
            ImageResult(
                image_id="r1/a",
                restaurant_id="r1",
                dishes=[Dish(name=name, origin="rules") for name in names],
                llm_used=usage is not None,
                llm_usage=usage,
                timings_s={"total": 1.5},
            ),
            ImageResult(image_id="r9/x", restaurant_id="r9", timings_s={"total": 0.5}),
        ],
    )


def _golden(tmp_path):
    golden = tmp_path / "golden"
    write_yaml(
        golden / "r1__a.yaml",
        {
            "image_id": "r1/a",
            "split": "dev",
            "dishes": [{"name": "Lamb Curry"}, {"name": "Patatas | Bravas"}],
        },
    )
    return golden


def test_run_report_sections(tmp_path):
    golden = _golden(tmp_path)
    run_dir = tmp_path / "run_ocr"
    _make_run(run_dir, {"mode": "ocr"}, ["Lamb Curry", "Postres"])
    md = run_report(run_dir, golden)
    assert "# Run `run_ocr`" in md
    assert "| all | 1 | 2 | 1 | 1 | 1 | 0 | 50.0% | 50.0% |" in md
    assert "`r9/x` | no reference" in md
    assert 'Missed: "Patatas | Bravas"' in md
    assert 'Wrong: "Postres"' in md
    assert "| `r1/a` | tuning | 2 | 2 | 1 | 1 | 1 | 0 | 50.0% | 50.0% | no |" in md
    assert (run_dir / "metrics.json").exists()


def test_compare_runs_table(tmp_path):
    golden = _golden(tmp_path)
    ocr_run, llm_run = tmp_path / "a", tmp_path / "b"
    _make_run(ocr_run, {"mode": "ocr", "llm_model": "claude-haiku-5-5"}, ["Lamb Curry"])
    _make_run(
        llm_run,
        {"mode": "ocr+llm", "llm_model": "claude-haiku-5-5", "llm_policy": "always"},
        ["Lamb Curry", "Patatas bravas"],
        cost=0.02,
    )
    md = compare_runs([ocr_run, llm_run], golden)
    lines = md.splitlines()
    assert lines[0].startswith("| run | mode | LLM model | policy | recall dev | recall test")
    assert lines[2].startswith("| `a` | ocr | - | - | 50.0% | - | 50.0% | 100.0% | 0.0% | $0.0000")
    assert lines[3].startswith(
        "| `b` | ocr+llm | claude-haiku-5-5 | always | 100.0% | - | 100.0% | 100.0% | 50.0% "
        "| $0.0200 | $0.0100"
    )
    assert not (ocr_run / "metrics.json").exists()
