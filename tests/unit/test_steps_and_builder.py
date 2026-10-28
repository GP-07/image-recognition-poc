"""Each step on its own, and the builder's fixed order and dependency checks."""

import io

import pytest
from PIL import Image

from menu_poc.cache.store import ArtifactStore
from menu_poc.data_classes import ImageRef
from menu_poc.llm.grounding import GroundingVerifier
from menu_poc.models import (
    Dish,
    LLMUsage,
    OcrLine,
    OcrResult,
    ParseResult,
    RunConfig,
    StructureResult,
)
from menu_poc.pipeline import PipelineBuilder, PipelineBuildError
from menu_poc.steps import DedupStep, ImageContext, LlmStep, OcrStep, RulesStep, Step, VerifyStep


def _ref(image_id="r/a", color=(10, 20, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, "PNG")
    return ImageRef(image_id, image_id.split("/")[0], f"{image_id}.png", buf.getvalue())


def _ocr(image_id="r/a"):
    line = OcrLine(id=0, text="Lamb curry 15,90€", bbox=(0, 0, 100, 10), confidence=1.0)
    return OcrResult(image_id=image_id, width=16, height=16, lines=[line], engine="fake")


class _Parser:
    def __init__(self, confidence):
        self.confidence = confidence

    def fingerprint(self):
        return {}

    def parse(self, ocr):
        return ParseResult(
            dishes=[Dish(name="Lamb curry", origin="rules")], confidence=self.confidence
        )


class _Structurer:
    def __init__(self):
        self.calls = 0

    def fingerprint(self):
        return {"model": "m"}

    def structure(self, ocr):
        self.calls += 1
        usage = LLMUsage(model="m", cost_usd=0.001, latency_s=1.5)
        return StructureResult(
            dishes=[Dish(name="Lamb curry", origin="llm")], usage=usage, prompt_version="v"
        )


class _Ocr:
    def fingerprint(self):
        return {"name": "fake"}

    def run(self, image, image_id):
        return _ocr(image_id)


class _Pre:
    def fingerprint(self):
        return {}

    def __call__(self, image):
        return image


# --- steps in isolation -----------------------------------------------------------------------


def test_rules_step_fills_result_and_context():
    ctx = ImageContext.for_image(_ref())
    ctx.ocr = _ocr()
    assert RulesStep(_Parser(0.8)).process(ctx) is True
    assert ctx.result.rules_confidence == 0.8 and ctx.parsed.dishes[0].name == "Lamb curry"


@pytest.mark.parametrize(
    ("policy", "confidence", "needed"),
    [("fallback", 0.2, True), ("fallback", 0.9, False), ("always", 0.9, True)],
)
def test_llm_step_policy(tmp_path, policy, confidence, needed):
    ctx = ImageContext.for_image(_ref())
    ctx.ocr, ctx.ocr_key = _ocr(), "k"
    ctx.parsed = _Parser(confidence).parse(ctx.ocr)
    structurer = _Structurer()
    step = LlmStep(structurer, ArtifactStore(tmp_path), policy, threshold=0.6)
    assert step.process(ctx) is needed
    assert structurer.calls == int(needed)


def test_llm_step_caches_answers(tmp_path):
    structurer, store = _Structurer(), ArtifactStore(tmp_path)
    for _ in range(2):
        ctx = ImageContext.for_image(_ref())
        ctx.ocr, ctx.ocr_key, ctx.parsed = _ocr(), "k", _Parser(0.1).parse(_ocr())
        LlmStep(structurer, store).process(ctx)
    assert structurer.calls == 1 and ctx.result.llm_usage.cached


def test_verify_step_replaces_rules_with_grounded_items():
    ctx = ImageContext.for_image(_ref())
    ctx.ocr, ctx.parsed = _ocr(), _Parser(0.1).parse(_ocr())
    invented = Dish(name="Pizza Hawaiana", origin="llm")
    real = Dish(name="Lamb curry", source_line_ids=[0], origin="llm")
    ctx.structured = StructureResult(
        dishes=[real, invented], usage=LLMUsage(model="m"), prompt_version="v"
    )
    VerifyStep(GroundingVerifier()).process(ctx)
    assert [dish.name for dish in ctx.result.dishes] == ["Lamb curry"] and ctx.result.llm_used
    assert ctx.result.ungrounded_dropped == ["Pizza Hawaiana"]


def test_dedup_step_stops_the_chain_for_a_duplicate():
    step = DedupStep()
    first = ImageContext.for_image(_ref("r/a"))
    assert step.process(first) is True
    first.result.dishes = [Dish(name="Lamb curry", origin="rules")]
    second = ImageContext.for_image(_ref("r/b"))  # same pixels and bytes
    assert step.process(second) is False
    assert second.result.duplicate_of == "r/a" and second.result.dishes == first.result.dishes


def test_a_failing_step_stops_only_its_image():
    class Boom(Step):
        name = "boom"

        def process(self, ctx):
            raise RuntimeError("bad photo")

    ctx = Boom().handle(ImageContext.for_image(_ref()))
    assert ctx.result.error == "boom: bad photo"


# --- builder ----------------------------------------------------------------------------------


def _pieces(tmp_path):
    store = ArtifactStore(tmp_path)
    return {
        "dedup": DedupStep(),
        "ocr": OcrStep(_Pre(), _Ocr(), store),
        "rules": RulesStep(_Parser(0.1)),
        "llm": LlmStep(_Structurer(), store),
    }


def test_build_order_does_not_depend_on_insertion_order(tmp_path):
    pieces = _pieces(tmp_path)
    builder = PipelineBuilder()
    for name in ["llm", "rules", "dedup", "ocr"]:  # deliberately scrambled
        builder.add(pieces[name])
    pipeline = builder.build()
    assert pipeline.step_names == ["dedup", "ocr", "rules", "llm", "verify"]
    assert pieces["ocr"].next is pieces["rules"] and pipeline.steps[-1].next is None


def test_llm_output_is_always_verified(tmp_path):
    pieces = _pieces(tmp_path)
    pipeline = PipelineBuilder().add(pieces["ocr"]).add(pieces["rules"]).add(pieces["llm"]).build()
    assert isinstance(pipeline.steps[-1], VerifyStep)


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (["rules"], "needs at least the OCR step"),
        (["ocr", "llm"], "'llm' needs rules"),
    ],
)
def test_invalid_pipelines_are_rejected(tmp_path, names, message):
    pieces = _pieces(tmp_path)
    builder = PipelineBuilder()
    for name in names:
        builder.add(pieces[name])
    with pytest.raises(PipelineBuildError, match=message):
        builder.build()


def test_from_config_includes_only_the_requested_pieces(tmp_path):
    ocr_only = RunConfig(input_path="x.zip", artifacts_dir=tmp_path, dedup=False)
    assert PipelineBuilder.from_config(ocr_only).build().step_names == ["ocr", "rules"]
    cascade = RunConfig(input_path="x.zip", artifacts_dir=tmp_path, llm_model="haiku")
    names = PipelineBuilder.from_config(cascade).build().step_names
    assert names == ["dedup", "ocr", "rules", "llm", "verify"]


def test_pipeline_process_returns_a_result_with_total_time(tmp_path):
    pieces = _pieces(tmp_path)
    pipeline = PipelineBuilder().add(pieces["ocr"]).add(pieces["rules"]).build()
    result = pipeline.process(_ref())
    assert result.dishes[0].name == "Lamb curry" and "total" in result.timings_s


def test_pipeline_fails_fast_when_ocr_cannot_load(tmp_path):
    class BrokenOcr(_Ocr):
        def ensure_ready(self):
            raise RuntimeError("model file missing")

    store = ArtifactStore(tmp_path)
    pipeline = PipelineBuilder().add(OcrStep(_Pre(), BrokenOcr(), store)).build()
    with pytest.raises(RuntimeError, match="model file missing"):
        pipeline.ensure_ready()
