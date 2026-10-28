"""Runs the real RapidOCR model on one menu photo: ``pytest -m slow``."""

from pathlib import Path

import pytest

from menu_poc.io.dataset import iter_images
from menu_poc.ocr.rapid import RapidOcrEngine
from menu_poc.parsing.rules import RuleBasedParser
from menu_poc.preprocess.image import ImagePreprocessor

DATASET = (
    Path(__file__).resolve().parents[3]
    / "_20260514 AI code test TO SEND"
    / "source"
    / "google_images.zip"
)
IMAGE_ID = "7571032721762097332/63bac2469809e9b1432e6f152996440d"  # single page, prices, tilted


@pytest.mark.slow
def test_rapidocr_reads_and_rules_parse_a_real_menu():
    if not DATASET.exists():
        pytest.skip(f"dataset not found: {DATASET}")
    (ref,) = [ref for ref in iter_images(DATASET) if ref.image_id == IMAGE_ID]
    image = ImagePreprocessor()(ref.open())

    engine = RapidOcrEngine()
    ocr = engine.run(image, ref.image_id)

    assert len(ocr.lines) > 30
    assert [line.id for line in ocr.lines] == list(range(len(ocr.lines)))
    assert all(0 <= line.x0 < line.x1 <= ocr.width for line in ocr.lines)
    assert ocr.rotation in (0, 358)  # slightly tilted photo; deskew may level it
    text = " ".join(line.text for line in ocr.lines).lower()
    assert "postres" in text

    fp = engine.fingerprint()
    assert fp["rapidocr_version"] and fp["models"]["Rec"]["ocr_version"]

    parsed = RuleBasedParser().parse(ocr)
    names = {dish.name.lower() for dish in parsed.dishes}
    assert {"coppa mascarpone", "brisa mediterránea"} <= names
    assert parsed.confidence >= 0.6
