"""Regression tests for the issues found in the code review."""

import io
import zipfile

import pytest
from PIL import Image

from menu_poc.evaluation.matching import match_dishes, normalize
from menu_poc.io.dataset import iter_images
from menu_poc.llm.fake_engine import FakeEngine
from menu_poc.llm.grounding import GroundingVerifier
from menu_poc.llm.structurer import LLMStructurer
from menu_poc.models import Dish, GoldenDish, OcrLine, OcrResult


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, "PNG")
    return buf.getvalue()


def test_grounding_keeps_non_latin_names():
    line = OcrLine(id=0, text="Ελληνική σαλάτα 8€", bbox=(0, 0, 10, 2), confidence=1.0)
    ocr = OcrResult(image_id="r/a", width=10, height=10, lines=[line], engine="t")
    dish = Dish(name="Ελληνική σαλάτα", source_line_ids=[0], origin="llm")
    assert GroundingVerifier().verify([dish], ocr).kept == [dish]


def test_matching_finds_the_optimal_assignment():
    gold = [GoldenDish(name="Pollo tikka"), GoldenDish(name="Pollo tikka masala")]
    result = match_dishes(["Pollo tikka masal", "Pollo tika"], gold)
    assert result.tp == 2 and result.fn == 0


def test_repeated_predictions_count_once():
    gold = [GoldenDish(name="Lamb curry"), GoldenDish(name="Lamb curry madras")]
    result = match_dishes(["Lamb curry madras", "Lamb curry madras"], gold)
    assert (result.tp, result.duplicates, result.fp) == (1, 1, 0)


def test_numbered_names_stay_distinct():
    assert normalize("Combo 1") != normalize("Combo 2")
    assert normalize("Lamb curry 12,50") == "lamb curry"


def test_root_level_images_get_an_id(tmp_path):
    (tmp_path / "root.png").write_bytes(_png())
    assert [ref.image_id for ref in iter_images(tmp_path)] == ["unknown/root"]


def test_duplicate_image_ids_are_rejected(tmp_path):
    path = tmp_path / "in.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("r1/x.jpg", _png())
        archive.writestr("r1/x.png", _png())
    with pytest.raises(ValueError, match="Duplicate image id"):
        list(iter_images(path))


def test_llm_cache_key_changes_with_client_settings():
    class Client(FakeEngine):
        def __init__(self, effort):
            super().__init__({}, model="m")
            self.effort = effort

        def fingerprint(self):
            return {"model": self.model, "effort": self.effort}

    assert (
        LLMStructurer(Client("low")).fingerprint() != LLMStructurer(Client("medium")).fingerprint()
    )
