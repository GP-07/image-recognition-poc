"""items.csv output and path resolution (config file vs CLI flags)."""

import csv

from menu_poc.config import PROJECT_ROOT, RunConfig, load_config
from menu_poc.io.writers import ItemRow, write_items_csv
from menu_poc.models import Dish, ImageResult


def test_items_csv_has_one_row_per_item(tmp_path):
    results = [
        ImageResult(
            image_id="r/a",
            restaurant_id="r",
            dishes=[
                Dish(name="Margarita", prices=[8.5, 12.0], section="PIZZAS", origin="rules"),
                Dish(name="Agua", description="50 cl", prices=[1.5], origin="llm"),
            ],
        ),
        ImageResult(image_id="r/b", restaurant_id="r", duplicate_of="r/a"),
    ]
    assert write_items_csv(tmp_path / "items.csv", results) == 2
    with (tmp_path / "items.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["item"] == "Margarita" and rows[0]["prices"] == "8.5|12"
    assert rows[1]["description"] == "50 cl" and rows[1]["source"] == "llm"


def test_default_output_is_tmp_in_the_project():
    assert RunConfig(input_path="x.zip").runs_dir == PROJECT_ROOT / "tmp"


def test_config_paths_are_relative_to_project_root_and_flags_to_cwd(tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("input_path: data/menus.zip\nruns_dir: out\n")
    monkeypatch.chdir(tmp_path)
    config = load_config(cfg)
    assert config.input_path == PROJECT_ROOT / "data/menus.zip"
    assert config.runs_dir == PROJECT_ROOT / "out"
    config = load_config(cfg, runs_dir="my_output")
    assert config.runs_dir == tmp_path.resolve() / "my_output"


def test_item_row_from_dish():
    result = ImageResult(image_id="r/b", restaurant_id="r", duplicate_of="r/a")
    row = ItemRow.from_dish(Dish(name="Café", prices=[1.2], origin="rules"), result)
    assert row == ItemRow("r/b", "r", "Café", "", "1.2", "", "rules", "r/a")
    assert ItemRow.columns()[:3] == ["image_id", "restaurant_id", "item"]
