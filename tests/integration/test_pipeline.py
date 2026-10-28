"""Pipeline cascade with fake stages: policy, caching, offline mode, failure isolation."""

import io
import zipfile

import pytest
from PIL import Image

from menu_poc.cache.store import ArtifactStore
from menu_poc.config import RunConfig
from menu_poc.io.writers import read_jsonl, write_jsonl
from menu_poc.models import (
    Dish,
    ImageResult,
    LLMUsage,
    OcrLine,
    OcrResult,
    ParseResult,
    StructureResult,
    VerificationResult,
)
from menu_poc.pipeline import Pipeline, new_run_dir, select_images


class FakePre:
    def fingerprint(self):
        return {"name": "fake-pre"}

    def __call__(self, image):
        return image


class FakeOcr:
    def __init__(self):
        self.calls = 0

    def fingerprint(self):
        return {"name": "fake-ocr"}

    def run(self, image, image_id):
        self.calls += 1
        if image_id.endswith("broken"):
            raise RuntimeError("boom")
        line = OcrLine(id=0, text="Lamb curry 15,90€", bbox=(0, 0, 100, 10), confidence=1.0)
        return OcrResult(image_id=image_id, width=8, height=8, lines=[line], engine="fake")


class FakeParser:
    """Confidence encoded in the image id: '.../low' -> 0.2, otherwise 0.9."""

    def fingerprint(self):
        return {"name": "fake-rules"}

    def parse(self, ocr):
        conf = 0.2 if ocr.image_id.endswith("low") else 0.9
        return ParseResult(dishes=[Dish(name="rules dish", origin="rules")], confidence=conf)


class FakeStructurer:
    def __init__(self, model="m1", fail=False):
        self.model, self.fail, self.calls = model, fail, 0

    def fingerprint(self):
        return {"model": self.model}

    def structure(self, ocr):
        self.calls += 1
        if self.fail:
            raise RuntimeError("api down")
        dishes = [Dish(name="Lamb curry", origin="llm"), Dish(name="Invented", origin="llm")]
        usage = LLMUsage(model=self.model, input_tokens=10, output_tokens=5, cost_usd=0.001)
        return StructureResult(dishes=dishes, usage=usage, prompt_version="v1")


class FakeVerifier:
    def verify(self, dishes, ocr):
        kept = [d for d in dishes if d.name != "Invented"]
        dropped = [d for d in dishes if d.name == "Invented"]
        return VerificationResult(kept=kept, dropped=dropped)


def _zip(tmp_path, names, identical=False):
    path = tmp_path / "in.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for i, name in enumerate(names):
            buf = io.BytesIO()
            Image.new("RGB", (8, 8), (0, 0, 0) if identical else (i * 40, 0, 0)).save(buf, "PNG")
            zf.writestr(f"imgs/r1/{name}.png", buf.getvalue())
    return path


def _pipeline(tmp_path, ocr=None, structurer=None, **cfg):
    config = RunConfig(input_path=_zip(tmp_path, ["high", "low"]), **cfg)
    return Pipeline(
        config,
        ArtifactStore(tmp_path / "artifacts"),
        FakePre(),
        ocr or FakeOcr(),
        FakeParser(),
        structurer or FakeStructurer(),
        FakeVerifier(),
    )


def _by_id(results):
    return {r.image_id: r for r in results}


def test_ocr_mode_never_calls_llm(tmp_path):
    structurer = FakeStructurer()
    pipe = _pipeline(tmp_path, structurer=structurer, mode="ocr")
    results = _by_id(pipe.run(select_images(pipe.config), tmp_path / "run"))
    assert structurer.calls == 0
    assert all(not r.llm_used for r in results.values())
    assert results["r1/low"].dishes[0].origin == "rules"


def test_fallback_policy_only_escalates_low_confidence_and_drops_ungrounded(tmp_path):
    pipe = _pipeline(tmp_path, mode="ocr+llm", llm_policy="fallback")
    results = _by_id(pipe.run(select_images(pipe.config), tmp_path / "run"))
    assert not results["r1/high"].llm_used
    low = results["r1/low"]
    assert low.llm_used and [d.name for d in low.dishes] == ["Lamb curry"]
    assert low.ungrounded_dropped == ["Invented"]
    assert low.llm_usage.cost_usd == pytest.approx(0.001)


def test_always_policy_escalates_every_image(tmp_path):
    pipe = _pipeline(tmp_path, mode="ocr+llm", llm_policy="always")
    results = pipe.run(select_images(pipe.config), tmp_path / "run")
    assert all(r.llm_used for r in results)


def test_rerun_reuses_ocr_and_llm_cache_and_new_model_only_calls_llm(tmp_path):
    ocr = FakeOcr()
    first = _pipeline(tmp_path, ocr=ocr, mode="ocr+llm", llm_policy="always")
    first.run(select_images(first.config), tmp_path / "run1")
    assert ocr.calls == 2

    same = FakeStructurer()
    second = _pipeline(tmp_path, ocr=ocr, structurer=same, mode="ocr+llm", llm_policy="always")
    results = second.run(select_images(second.config), tmp_path / "run2")
    assert ocr.calls == 2 and same.calls == 0
    assert all(r.llm_usage.cached for r in results)

    other = FakeStructurer(model="m2")
    third = _pipeline(tmp_path, ocr=ocr, structurer=other, mode="ocr+llm", llm_policy="always")
    third.run(select_images(third.config), tmp_path / "run3")
    assert ocr.calls == 2 and other.calls == 2


def test_offline_cache_miss_keeps_rules_output(tmp_path):
    pipe = _pipeline(tmp_path, mode="ocr+llm", llm_policy="always", llm_offline=True)
    results = pipe.run(select_images(pipe.config), tmp_path / "run")
    assert all(not r.llm_used and r.dishes[0].origin == "rules" for r in results)
    assert all("offline" in r.error for r in results)


def test_llm_failure_keeps_rules_output(tmp_path):
    pipe = _pipeline(tmp_path, structurer=FakeStructurer(fail=True), mode="ocr+llm")
    low = _by_id(pipe.run(select_images(pipe.config), tmp_path / "run"))["r1/low"]
    assert not low.llm_used and low.dishes[0].origin == "rules" and "api down" in low.error


def test_broken_image_is_isolated(tmp_path):
    config = RunConfig(input_path=_zip(tmp_path, ["ok", "broken"]))
    pipe = Pipeline(config, ArtifactStore(tmp_path / "a"), FakePre(), FakeOcr(), FakeParser())
    results = _by_id(pipe.run(select_images(config), tmp_path / "run"))
    assert results["r1/broken"].error and not results["r1/ok"].error
    assert len(read_jsonl(tmp_path / "run" / "results.jsonl", ImageResult)) == 2


def test_identical_images_share_ocr_but_keep_their_own_id(tmp_path):
    ocr = FakeOcr()
    config = RunConfig(input_path=_zip(tmp_path, ["high", "low"], identical=True))
    pipe = Pipeline(config, ArtifactStore(tmp_path / "a"), FakePre(), ocr, FakeParser())
    results = _by_id(pipe.run(select_images(config), tmp_path / "run"))
    assert ocr.calls == 1
    assert results["r1/low"].rules_confidence == pytest.approx(0.2)


def test_select_only_low_confidence_from_previous_run(tmp_path):
    zpath = _zip(tmp_path, ["high", "low"])
    prev = tmp_path / "prev"
    write_jsonl(
        prev / "results.jsonl",
        [
            ImageResult(image_id="r1/high", restaurant_id="r1", rules_confidence=0.9),
            ImageResult(image_id="r1/low", restaurant_id="r1", rules_confidence=0.1),
        ],
    )
    config = RunConfig(input_path=zpath, reuse_from=prev, only_low_confidence=True)
    assert [r.image_id for r in select_images(config)] == ["r1/low"]


def test_new_run_dir_writes_config(tmp_path):
    config = RunConfig(input_path="x.zip", runs_dir=tmp_path, mode="ocr+llm")
    run_dir = new_run_dir(config)
    assert "llm-fallback-claude-haiku-5-5" in run_dir.name
    assert (run_dir / "config.yaml").exists()
