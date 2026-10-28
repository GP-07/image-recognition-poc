from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from menu_poc.constants import DEFAULT_PROMPT_PATH, DISHES_SCHEMA
from menu_poc.llm.engine import prompt_key
from menu_poc.llm.fake_engine import FakeEngine
from menu_poc.llm.grounding import GroundingVerifier
from menu_poc.llm.structurer import LLMStructurer, build_user_message
from menu_poc.llm.text import reading_order, render_lines
from menu_poc.models import OcrLine, OcrResult

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "llm"


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _line(line_id: int, text: str, x0: float, y0: float, x1: float, height: float = 20) -> OcrLine:
    return OcrLine(id=line_id, text=text, bbox=(x0, y0, x1, y0 + height), confidence=0.9)


def test_default_prompt_exists_and_states_key_rules():
    text = DEFAULT_PROMPT_PATH.read_text()
    for phrase in ("VERBATIM", "beverages", "source_line_ids", "Never invent"):
        assert phrase in text


def test_schema_shape_is_strict():
    item = DISHES_SCHEMA["properties"]["dishes"]["items"]
    assert DISHES_SCHEMA["additionalProperties"] is False
    assert item["additionalProperties"] is False
    assert set(item["required"]) == set(item["properties"])
    assert item["properties"]["prices"]["items"] == {"type": "number"}
    assert item["properties"]["source_line_ids"]["items"] == {"type": "integer"}


def test_reading_order_pairs_far_right_price_with_name():
    lines = [
        _line(0, "8,50 €", 800, 102, 860),  # price slightly lower, far right
        _line(1, "Margarita", 50, 100, 200),
        _line(2, "Tomate, mozzarella", 50, 125, 300),
        _line(3, "PIZZAS", 50, 60, 150),
    ]
    assert [ln.id for ln in reading_order(lines)] == [3, 1, 0, 2]


def test_reading_order_keeps_stacked_lines_top_down():
    lines = [_line(0, "Long description line", 40, 112, 900), _line(1, "Name", 45, 100, 200)]
    assert [ln.id for ln in reading_order(lines)] == [1, 0]


def test_render_lines_format():
    ocr = OcrResult(
        image_id="r/i",
        width=1000,
        height=1000,
        engine="t",
        lines=[_line(7, "Agua  ", 100, 490, 200)],
    )
    assert render_lines(ocr) == "[7] (10,50) Agua"


def test_end_to_end_with_recorded_fixture():
    fx = load_fixture("pizzeria_salads_desserts.json")
    ocr = OcrResult(**fx["ocr"])
    structurer = LLMStructurer(FakeEngine(lambda system, user, schema: fx["response"]))
    result = structurer.structure(ocr)

    assert [dish.name for dish in result.dishes] == fx["expected_names"]
    assert all(dish.origin == "llm" for dish in result.dishes)
    assert result.prompt_version.startswith("structure_v2@")
    assert result.usage.cost_usd == 0.0
    first = result.dishes[0]
    assert first.prices == [4.9] and first.section == "ENSALADAS" and first.source_line_ids
    assert not {"Agua", "San Miguel"} & {dish.name for dish in result.dishes}

    verified = GroundingVerifier().verify(result.dishes, ocr)
    assert len(verified.kept) == len(result.dishes) and not verified.dropped


def test_fake_client_keyed_by_prompt_hash():
    fx = load_fixture("indian_multilingual.json")
    ocr = OcrResult(**fx["ocr"])
    structurer = LLMStructurer(FakeEngine({}))
    key = prompt_key(structurer.system, build_user_message(ocr))
    structurer.client = FakeEngine({key: fx["response"]})
    result = structurer.structure(ocr)
    assert [dish.name for dish in result.dishes] == fx["expected_names"]
    verified = GroundingVerifier().verify(result.dishes, ocr)
    assert [dish.name for dish in verified.kept] == fx["expected_names"]


def test_user_message_contains_ids_and_text():
    fx = load_fixture("indian_multilingual.json")
    message = build_user_message(OcrResult(**fx["ocr"]))
    assert "] (14,23) 46. LAMB CURRY." in message
    assert message.count("\n[") == len(fx["ocr"]["lines"])


def test_tolerant_conversion():
    answer = {
        "dishes": [
            {
                "name": "  Pizza   Margarita ",
                "description": "",
                "prices": [8.5, -1, True],
                "section": None,
                "source_line_ids": [3, 3, 4],
            },
            {"name": "", "description": None, "prices": [], "section": None, "source_line_ids": []},
        ]
    }
    ocr = OcrResult(image_id="r/i", width=10, height=10, engine="t", lines=[_line(3, "x", 0, 0, 1)])
    result = LLMStructurer(FakeEngine(lambda *_: answer)).structure(ocr)
    assert len(result.dishes) == 1
    dish = result.dishes[0]
    assert dish.name == "Pizza Margarita" and dish.description is None
    assert dish.prices == [8.5] and dish.source_line_ids == [3, 4]


def test_empty_ocr_skips_the_call():
    fake = FakeEngine({})
    ocr = OcrResult(image_id="r/i", width=10, height=10, engine="t", lines=[])
    result = LLMStructurer(fake).structure(ocr)
    assert result.dishes == [] and fake.calls == []


def test_fingerprint_changes_with_model_and_prompt(tmp_path):
    prompt = tmp_path / "structure_v1.md"
    prompt.write_text("v1")
    original = LLMStructurer(FakeEngine({}, model="m1"), prompt).fingerprint()
    other_model = LLMStructurer(FakeEngine({}, model="m2"), prompt).fingerprint()
    prompt.write_text("v1 edited")
    edited = LLMStructurer(FakeEngine({}, model="m1"), prompt).fingerprint()
    assert original["client"]["model"] == "m1"
    assert original["prompt_version"].startswith("structure_v1@")
    assert original != other_model
    assert original["prompt_version"] != edited["prompt_version"]


@pytest.mark.parametrize("name", ["pizzeria_salads_desserts.json", "indian_multilingual.json"])
def test_fixture_responses_match_schema_keys(name):
    item_keys = set(DISHES_SCHEMA["properties"]["dishes"]["items"]["properties"])
    for dish in load_fixture(name)["response"]["dishes"]:
        assert set(dish) == item_keys
