"""Pipeline chain built with fake stages: policy, caching, offline mode, failure isolation."""

import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from menu_poc.cache.store import ArtifactStore
from menu_poc.io.dataset import iter_images
from menu_poc.io.writers import read_jsonl
from menu_poc.models import (
    Dish,
    ImageResult,
    LLMUsage,
    OcrLine,
    OcrResult,
    ParseResult,
    RunConfig,
    StructureResult,
    VerificationResult,
)
from menu_poc.pipeline import PipelineBuilder, new_run_dir
from menu_poc.steps import DedupStep, LlmStep, OcrStep, RulesStep, VerifyStep


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

    def unavailable_reason(self):
        return None

    def structure(self, ocr):
        self.calls += 1
        if self.fail:
            raise RuntimeError("api down")
        dishes = [Dish(name="Lamb curry", origin="llm"), Dish(name="Invented", origin="llm")]
        usage = LLMUsage(model=self.model, input_tokens=10, output_tokens=5, cost_usd=0.001)
        return StructureResult(dishes=dishes, usage=usage, prompt_version="v1")


class FakeVerifier:
    def verify(self, dishes, ocr):
        kept = [dish for dish in dishes if dish.name != "Invented"]
        dropped = [dish for dish in dishes if dish.name == "Invented"]
        return VerificationResult(kept=kept, dropped=dropped)


def _noise(seed: int, size: int = 64) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))


def _zip(tmp_path, names, identical=False):
    """One PNG per name; distinct random content unless ``identical``."""
    path = tmp_path / "in.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for seed, name in enumerate(names):
            buf = io.BytesIO()
            _noise(0 if identical else seed).save(buf, "PNG")
            archive.writestr(f"imgs/r1/{name}.png", buf.getvalue())
    return path


def _build(config, store, ocr=None, structurer=None, verifier=None):
    """Same pieces as ``PipelineBuilder.from_config``, with fake stages."""
    builder = PipelineBuilder()
    builder.add(OcrStep(FakePre(), ocr or FakeOcr(), store)).add(RulesStep(FakeParser()))
    if config.dedup:
        builder.add(DedupStep(config.dedup_max_distance))
    if config.uses_llm:
        policy, threshold = config.llm_policy, config.confidence_threshold
        llm = LlmStep(structurer or FakeStructurer(), store, policy, threshold, config.llm_offline)
        builder.add(llm).add(VerifyStep(verifier or FakeVerifier()))
    return builder.build()


def _pipeline(tmp_path, ocr=None, structurer=None, **cfg):
    config = RunConfig(input_path=_zip(tmp_path, ["high", "low"]), **cfg)
    return _build(config, ArtifactStore(tmp_path / "artifacts"), ocr, structurer), config


def _by_id(results):
    return {result.image_id: result for result in results}


def test_ocr_mode_never_calls_llm(tmp_path):
    structurer = FakeStructurer()
    pipe, config = _pipeline(tmp_path, structurer=structurer)
    results = _by_id(pipe.run(list(iter_images(config.input_path)), tmp_path / "run"))
    assert structurer.calls == 0
    assert all(not result.llm_used for result in results.values())
    assert results["r1/low"].dishes[0].origin == "rules"


def test_fallback_policy_only_escalates_low_confidence_and_drops_ungrounded(tmp_path):
    pipe, config = _pipeline(tmp_path, llm_model="haiku", llm_policy="fallback")
    results = _by_id(pipe.run(list(iter_images(config.input_path)), tmp_path / "run"))
    assert not results["r1/high"].llm_used
    low = results["r1/low"]
    assert low.llm_used and [dish.name for dish in low.dishes] == ["Lamb curry"]
    assert low.ungrounded_dropped == ["Invented"]
    assert low.llm_usage.cost_usd == pytest.approx(0.001)


def test_always_policy_escalates_every_image(tmp_path):
    pipe, config = _pipeline(tmp_path, llm_model="haiku", llm_policy="always")
    results = pipe.run(list(iter_images(config.input_path)), tmp_path / "run")
    assert all(result.llm_used for result in results)


def test_rerun_reuses_ocr_and_llm_cache_and_new_model_only_calls_llm(tmp_path):
    ocr = FakeOcr()
    first, cfg1 = _pipeline(tmp_path, ocr=ocr, llm_model="haiku", llm_policy="always")
    first.run(list(iter_images(cfg1.input_path)), tmp_path / "run1")
    assert ocr.calls == 2

    same = FakeStructurer()
    second, cfg2 = _pipeline(
        tmp_path, ocr=ocr, structurer=same, llm_model="haiku", llm_policy="always"
    )
    results = second.run(list(iter_images(cfg2.input_path)), tmp_path / "run2")
    assert ocr.calls == 2 and same.calls == 0
    assert all(result.llm_usage.cached for result in results)

    other = FakeStructurer(model="m2")
    third, cfg3 = _pipeline(
        tmp_path, ocr=ocr, structurer=other, llm_model="haiku", llm_policy="always"
    )
    third.run(list(iter_images(cfg3.input_path)), tmp_path / "run3")
    assert ocr.calls == 2 and other.calls == 2


def test_offline_cache_miss_keeps_rules_output(tmp_path):
    pipe, config = _pipeline(tmp_path, llm_model="haiku", llm_policy="always", llm_offline=True)
    results = pipe.run(list(iter_images(config.input_path)), tmp_path / "run")
    assert all(not result.llm_used and result.dishes[0].origin == "rules" for result in results)
    assert all("offline" in result.error for result in results)


def test_llm_failure_keeps_rules_output(tmp_path):
    pipe, config = _pipeline(tmp_path, structurer=FakeStructurer(fail=True), llm_model="haiku")
    low = _by_id(pipe.run(list(iter_images(config.input_path)), tmp_path / "run"))["r1/low"]
    assert not low.llm_used and low.dishes[0].origin == "rules" and "api down" in low.error


def test_broken_image_is_isolated(tmp_path):
    config = RunConfig(input_path=_zip(tmp_path, ["ok", "broken"]))
    pipe = _build(config, ArtifactStore(tmp_path / "a"))
    results = _by_id(pipe.run(list(iter_images(config.input_path)), tmp_path / "run"))
    assert results["r1/broken"].error and not results["r1/ok"].error
    assert len(read_jsonl(tmp_path / "run" / "results.jsonl", ImageResult)) == 2


def test_identical_images_share_ocr_but_keep_their_own_id(tmp_path):
    ocr = FakeOcr()
    config = RunConfig(input_path=_zip(tmp_path, ["high", "low"], identical=True), dedup=False)
    pipe = _build(config, ArtifactStore(tmp_path / "a"), ocr)
    results = _by_id(pipe.run(list(iter_images(config.input_path)), tmp_path / "run"))
    assert ocr.calls == 1
    assert results["r1/low"].rules_confidence == pytest.approx(0.2)


def test_new_run_dir_writes_config(tmp_path):
    config = RunConfig(input_path="x.zip", runs_dir=tmp_path, llm_model="haiku")
    run_dir = new_run_dir(config)
    assert "llm-fallback-claude-haiku-5-5" in run_dir.name
    assert (run_dir / "config.yaml").exists()


class RaisingVerifier:
    def fingerprint(self):
        return {}

    def verify(self, dishes, ocr):
        raise RuntimeError("verifier bug")


def test_verifier_failure_is_isolated(tmp_path):
    config = RunConfig(input_path=_zip(tmp_path, ["high", "low"]), llm_model="haiku")
    store = ArtifactStore(tmp_path / "a")
    pipe = _build(config, store, verifier=RaisingVerifier())
    low = _by_id(pipe.run(list(iter_images(config.input_path)), tmp_path / "run"))["r1/low"]
    assert "verifier bug" in low.error and low.dishes[0].origin == "rules"


def test_billed_failure_keeps_its_cost(tmp_path):
    from menu_poc.llm.engine import LLMOutputError

    class TruncatingStructurer(FakeStructurer):
        def structure(self, ocr):
            usage = LLMUsage(model="m", output_tokens=16000, cost_usd=0.32)
            raise LLMOutputError("truncated", usage=usage)

    pipe, config = _pipeline(tmp_path, structurer=TruncatingStructurer(), llm_model="haiku")
    low = _by_id(pipe.run(list(iter_images(config.input_path)), tmp_path / "run"))["r1/low"]
    assert not low.llm_used and low.llm_usage.cost_usd == pytest.approx(0.32)


def test_rules_signals_survive_llm_replacement(tmp_path):
    pipe, config = _pipeline(tmp_path, llm_model="haiku", llm_policy="always")
    low = _by_id(pipe.run(list(iter_images(config.input_path)), tmp_path / "run"))["r1/low"]
    assert low.llm_used and low.rules_dish_count == 1 and low.llm_prompt_version == "v1"


def test_duplicate_photos_are_skipped_before_ocr(tmp_path):
    path = tmp_path / "in.zip"
    big, small = io.BytesIO(), io.BytesIO()
    _noise(7, 128).save(big, "PNG")
    _noise(7, 128).resize((64, 64)).save(small, "JPEG", quality=90)  # resized + re-encoded
    other = io.BytesIO()
    _noise(8, 128).save(other, "PNG")
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("imgs/r1/a_small.jpg", small.getvalue())
        zf.writestr("imgs/r1/b_big.png", big.getvalue())
        zf.writestr("imgs/r1/c_other.png", other.getvalue())
    ocr = FakeOcr()
    config = RunConfig(input_path=path)
    pipe = _build(config, ArtifactStore(tmp_path / "a"), ocr)
    results = _by_id(pipe.run(list(iter_images(config.input_path)), tmp_path / "run"))
    assert ocr.calls == 2  # the small copy was never OCR'd
    dup = results["r1/a_small"]
    assert dup.duplicate_of == "r1/b_big"  # the larger image is kept as the original
    assert dup.dishes == results["r1/b_big"].dishes and "near-duplicate" in dup.duplicate_reason
    saved = read_jsonl(tmp_path / "run" / "results.jsonl", ImageResult)
    assert [result.image_id for result in saved] == [
        "r1/a_small",
        "r1/b_big",
        "r1/c_other",
    ]
