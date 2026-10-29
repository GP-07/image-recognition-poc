"""Each step on its own, and the builder's fixed order and dependency checks."""

import io

import pytest
from PIL import Image

from menu_poc.cache.store import ArtifactStore
from menu_poc.data_classes import ImageRef
from menu_poc.interfaces import OcrEngine, Parser, PostProcessor, Preprocessor, Structurer
from menu_poc.llm.engine import LLMOutputError, LLMRetryableError
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
from menu_poc.steps import (
    DedupStep,
    ImageContext,
    InputValidationError,
    LlmStep,
    OcrStep,
    RulesStep,
    Step,
    VerifyStep,
)


def _ref(image_id="r/a", color=(10, 20, 30)):
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color).save(buf, "PNG")
    return ImageRef(image_id, image_id.split("/")[0], f"{image_id}.png", buf.getvalue())


def _ocr(image_id="r/a"):
    line = OcrLine(id=0, text="Lamb curry 15,90€", bbox=(0, 0, 100, 10), confidence=1.0)
    return OcrResult(image_id=image_id, width=16, height=16, lines=[line], engine="fake")


class _Parser(Parser):
    def __init__(self, confidence):
        self.confidence = confidence

    def fingerprint(self):
        return {}

    def parse(self, ocr):
        return ParseResult(
            dishes=[Dish(name="Lamb curry", origin="rules")], confidence=self.confidence
        )


class _Structurer(Structurer):
    def __init__(self):
        self.calls = 0

    def fingerprint(self):
        return {"model": "m"}

    def unavailable_reason(self):
        return None

    def structure(self, ocr):
        self.calls += 1
        usage = LLMUsage(model="m", cost_usd=0.001, latency_s=1.5)
        return StructureResult(
            dishes=[Dish(name="Lamb curry", origin="llm")], usage=usage, prompt_version="v"
        )


def _rules():
    return RulesStep(_Parser(0.1))


class _Ocr(OcrEngine):
    def fingerprint(self):
        return {"name": "fake"}

    def ensure_ready(self):
        pass

    def run(self, image, image_id):
        return _ocr(image_id)


class _Pre(Preprocessor):
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


@pytest.mark.parametrize(
    ("failure", "retried"),
    [(LLMRetryableError("timeout"), True), (LLMOutputError("refused"), False), (None, False)],
)
def test_only_transient_failures_are_retried(failure, retried):
    ctx = ImageContext.for_image(_ref())
    ctx.failure = failure
    assert ctx.should_be_retried is retried


def test_a_failing_step_stops_only_its_image():
    class Boom(Step):
        name = "boom"

        def process(self, ctx):
            raise RuntimeError("bad photo")

    ctx = Boom().handle(ImageContext.for_image(_ref()))
    assert ctx.result.error == "boom: bad photo"
    assert str(ctx.failure) == "bad photo" and not ctx.should_be_retried


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


def test_from_config_adds_the_evaluation_only_with_a_golden_set(tmp_path):
    def post_processors(golden_dir):
        config = RunConfig(input_path="x.zip", artifacts_dir=tmp_path, golden_dir=golden_dir)
        return PipelineBuilder.from_config(config).build().post_processors

    assert [step.name for step in post_processors(tmp_path)] == ["golden_set_evaluation"]
    assert post_processors(None) == [] and post_processors(tmp_path / "missing") == []


class _Export(PostProcessor):
    name = "export"

    def __init__(self, fail=False):
        self.fail = fail

    def run(self, outcome):
        if self.fail:
            raise RuntimeError("disk full")
        path = outcome.run_dir / "export.txt"
        path.write_text(str(outcome.n_items))
        return [path]


def test_post_processors_run_after_the_run_and_never_lose_it(tmp_path):
    pieces = _pieces(tmp_path)
    builder = PipelineBuilder().add(pieces["ocr"]).add(pieces["rules"])
    pipeline = builder.add_post_processor(_Export(fail=True)).add_post_processor(_Export()).build()
    refs, _ = pipeline.validate_inputs(_image_dir(tmp_path))
    outcome = pipeline.run(refs, RunConfig(input_path="x.zip", runs_dir=tmp_path / "runs"))
    assert [report.name for report in outcome.reports] == ["export.txt"]
    assert "Post-processor export failed: disk full" in outcome.warnings
    assert outcome.manifest()["reports"] == ["export.txt"]


def test_pipeline_process_returns_a_result_with_total_time(tmp_path):
    pieces = _pieces(tmp_path)
    pipeline = PipelineBuilder().add(pieces["ocr"]).add(pieces["rules"]).build()
    result = pipeline.process(_ref())
    assert result.dishes[0].name == "Lamb curry" and "total" in result.timings_s


# --- run-level validation ---------------------------------------------------------------------


def _image_dir(tmp_path):
    folder = tmp_path / "images" / "r"
    folder.mkdir(parents=True)
    (folder / "a.png").write_bytes(_ref().data)
    return tmp_path / "images"


def test_validate_inputs_fails_when_ocr_cannot_load(tmp_path):
    class BrokenOcr(_Ocr):
        def ensure_ready(self):
            raise RuntimeError("model file missing")

    store = ArtifactStore(tmp_path)
    pipeline = PipelineBuilder().add(OcrStep(_Pre(), BrokenOcr(), store)).build()
    with pytest.raises(InputValidationError, match="model file missing"):
        pipeline.validate_inputs(_image_dir(tmp_path))


@pytest.mark.parametrize("make_input", [lambda tmp: tmp / "missing.zip", lambda tmp: tmp])
def test_validate_inputs_fails_without_images(tmp_path, make_input):
    pipeline = PipelineBuilder().add(OcrStep(_Pre(), _Ocr(), ArtifactStore(tmp_path))).build()
    with pytest.raises(InputValidationError):
        pipeline.validate_inputs(make_input(tmp_path))


def test_no_llm_configured_warns_how_to_enable_it(tmp_path):
    pieces = _pieces(tmp_path)
    pipeline = PipelineBuilder().add(pieces["ocr"]).add(pieces["rules"]).build()
    refs, warnings = pipeline.validate_inputs(_image_dir(tmp_path))
    assert [ref.image_id for ref in refs] == ["r/a"] and not pipeline.uses_llm
    assert "OCR + rules only" in warnings[0] and "--llm-model haiku" in warnings[0]


def test_unavailable_llm_turns_its_step_off(tmp_path):
    class NoKeyStructurer(_Structurer):
        def unavailable_reason(self):
            return "no key"

    store = ArtifactStore(tmp_path)
    llm = LlmStep(NoKeyStructurer(), store, policy="always")
    pipeline = PipelineBuilder().add(OcrStep(_Pre(), _Ocr(), store)).add(_rules()).add(llm).build()
    _, warnings = pipeline.validate_inputs(_image_dir(tmp_path))
    assert warnings == ["LLM step disabled: no key. Results come from OCR + rules only."]
    assert not pipeline.uses_llm and not pipeline.process(_ref()).llm_used
