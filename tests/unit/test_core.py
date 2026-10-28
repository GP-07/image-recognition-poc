import io
import zipfile

import pytest
from PIL import Image

from menu_poc.cache.store import ArtifactStore, make_key
from menu_poc.config import RunConfig, load_config
from menu_poc.io.dataset import iter_images
from menu_poc.io.writers import read_jsonl, write_jsonl
from menu_poc.models import Dish, ImageResult, OcrLine


def _jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="JPEG")
    return buf.getvalue()


def test_ocr_line_geometry():
    line = OcrLine(id=0, text="x", bbox=(10, 20, 30, 60), confidence=0.9)
    assert (line.width, line.height, line.x_center, line.y_center) == (20, 40, 20, 40)


def test_make_key_is_stable_and_order_independent():
    assert make_key({"a": 1, "b": 2}, "x") == make_key({"b": 2, "a": 1}, "x")
    assert make_key({"a": 1}) != make_key({"a": 2})


def test_artifact_store_roundtrip(tmp_path):
    store = ArtifactStore(tmp_path)
    key = make_key("img", {"engine": "fake"})
    assert store.get("ocr", key) is None
    store.put("ocr", key, {"lines": [1, 2]})
    assert store.get("ocr", key) == {"lines": [1, 2]}


def test_iter_images_from_zip_skips_macosx_and_filters(tmp_path):
    zpath = tmp_path / "menus.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("google_images/r1/a.jpg", _jpeg_bytes())
        zf.writestr("google_images/r2/b.JPG", _jpeg_bytes())
        zf.writestr("__MACOSX/google_images/r1/._a.jpg", b"junk")
        zf.writestr("google_images/r1/notes.txt", b"x")
    refs = list(iter_images(zpath))
    assert [r.image_id for r in refs] == ["r1/a", "r2/b"]
    assert refs[0].open().size == (8, 8)
    assert [r.image_id for r in iter_images(zpath, ["r2/b"])] == ["r2/b"]


def test_iter_images_from_dir(tmp_path):
    (tmp_path / "r1").mkdir()
    (tmp_path / "r1" / "a.jpg").write_bytes(_jpeg_bytes())
    assert [r.restaurant_id for r in iter_images(tmp_path)] == ["r1"]


def test_iter_images_rejects_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_images(tmp_path / "nope.zip"))


def test_config_overrides_and_validation(tmp_path):
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("input_path: x.zip\nmode: ocr\n")
    cfg = load_config(cfg_file, mode="ocr+llm", llm_model=None)
    assert cfg.uses_llm and cfg.llm_model == "claude-haiku-5-5"
    with pytest.raises(ValueError):
        RunConfig(input_path="x.zip", only_low_confidence=True)


def test_jsonl_roundtrip(tmp_path):
    result = ImageResult(
        image_id="r/a",
        restaurant_id="r",
        dishes=[Dish(name="Lamb curry", prices=[15.9], origin="rules")],
    )
    write_jsonl(tmp_path / "r.jsonl", [result])
    assert read_jsonl(tmp_path / "r.jsonl", ImageResult) == [result]
