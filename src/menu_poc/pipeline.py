"""Cascade orchestration: preprocess -> OCR -> rules -> (LLM if needed) -> grounding.

Stage outputs that are slow or cost money (OCR, LLM) go through the artifact store, keyed by
their input and the stage fingerprint. Re-running with another LLM model reuses the stored OCR.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from menu_poc.cache.store import ArtifactStore, make_key
from menu_poc.config import RunConfig
from menu_poc.dedup import find_duplicates
from menu_poc.interfaces import OcrEngine, Parser, Preprocessor, Structurer, Verifier
from menu_poc.io.dataset import ImageRef, iter_images
from menu_poc.io.writers import read_jsonl, write_jsonl, write_yaml
from menu_poc.models import ImageResult, OcrResult, StructureResult

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        config: RunConfig,
        store: ArtifactStore,
        preprocessor: Preprocessor,
        ocr: OcrEngine,
        parser: Parser,
        structurer: Structurer | None = None,
        verifier: Verifier | None = None,
    ) -> None:
        if config.uses_llm and (structurer is None or verifier is None):
            raise ValueError("mode 'ocr+llm' needs a structurer and a verifier")
        self.config = config
        self.store = store
        self.preprocessor = preprocessor
        self.ocr = ocr
        self.parser = parser
        self.structurer = structurer
        self.verifier = verifier

    def _run_ocr(self, ref: ImageRef) -> tuple[OcrResult, str, bool]:
        key = make_key("ocr", ref.sha256, self.preprocessor.fingerprint(), self.ocr.fingerprint())
        cached = self.store.get("ocr", key)
        if cached is not None:
            # Identical bytes under another id share the entry: re-label it for this image.
            ocr = OcrResult.model_validate(cached).model_copy(update={"image_id": ref.image_id})
            return ocr, key, True
        start = time.perf_counter()
        result = self.ocr.run(self.preprocessor(ref.open()), ref.image_id)
        if not result.elapsed_s:
            result.elapsed_s = time.perf_counter() - start
        self.store.put("ocr", key, result.model_dump(mode="json"))
        return result, key, False

    def _run_llm(self, ocr: OcrResult, ocr_key: str) -> StructureResult | None:
        """LLM dishes, from cache when possible. None on a cache miss in offline mode."""
        assert self.structurer is not None
        key = make_key("llm", ocr_key, self.structurer.fingerprint())
        cached = self.store.get("llm", key)
        if cached is not None:
            result = StructureResult.model_validate(cached)
            result.usage = result.usage.model_copy(update={"cached": True})
            return result
        if self.config.llm_offline:
            return None
        result = self.structurer.structure(ocr)
        self.store.put("llm", key, result.model_dump(mode="json"))
        return result

    def process(self, ref: ImageRef) -> ImageResult:
        """Process one image.

        Reported cost and timings are those of *producing* the result, also when it comes from
        the cache, so runs stay comparable; ``llm_usage.cached`` tells whether the API was hit.
        """
        result = ImageResult(image_id=ref.image_id, restaurant_id=ref.restaurant_id)
        try:
            result = self._process(ref, result)
        finally:
            result.timings_s["total"] = sum(result.timings_s.values())
        return result

    def _process(self, ref: ImageRef, result: ImageResult) -> ImageResult:
        try:
            ocr, ocr_key, _ = self._run_ocr(ref)
            result.timings_s["ocr"] = ocr.elapsed_s

            start = time.perf_counter()
            parsed = self.parser.parse(ocr)
            result.timings_s["rules"] = time.perf_counter() - start
            result.dishes = parsed.dishes
            result.rules_confidence = parsed.confidence
            result.rules_dish_count = len(parsed.dishes)
            result.rules_signals = parsed.signals
        except Exception as exc:  # one bad image must not stop the run
            log.exception("OCR/rules failed for %s", ref.image_id)
            result.error = f"ocr/rules: {exc}"
            return result

        needs_llm = self.config.uses_llm and (
            self.config.llm_policy == "always"
            or parsed.confidence < self.config.confidence_threshold
        )
        if not needs_llm:
            return result

        assert self.verifier is not None
        try:
            structured = self._run_llm(ocr, ocr_key)
            if structured is None:
                result.error = "llm: cache miss in offline mode, kept rules output"
                return result
            verified = self.verifier.verify(structured.dishes, ocr)
        except Exception as exc:  # keep the rules answer when the LLM stage fails
            log.exception("LLM failed for %s", ref.image_id)
            result.error = f"llm: {exc}"
            # A billed but unusable answer (refusal, truncation) still counts as cost.
            result.llm_usage = getattr(exc, "usage", None)
            return result

        result.llm_used = True
        result.llm_usage = structured.usage
        result.llm_prompt_version = structured.prompt_version
        result.timings_s["llm"] = structured.usage.latency_s
        result.ungrounded_dropped = [d.name for d in verified.dropped]
        if verified.kept or not parsed.dishes:
            result.dishes = verified.kept
        else:  # an empty LLM answer should not wipe out what the rules found
            result.error = "llm: no grounded dishes, kept rules output"
        return result

    def run(self, refs: Iterable[ImageRef], run_dir: Path) -> list[ImageResult]:
        refs = list(refs)
        duplicates = (
            find_duplicates(refs, self.config.dedup_max_distance) if self.config.dedup else {}
        )
        done: dict[str, ImageResult] = {}
        for ref in refs:
            if ref.image_id in duplicates:
                continue
            log.info("processing %s", ref.image_id)
            done[ref.image_id] = self.process(ref)
            # Rewrite after every image so an interrupted run keeps its progress.
            write_jsonl(run_dir / "results.jsonl", done.values())

        results = []
        for ref in refs:
            dup = duplicates.get(ref.image_id)
            if dup is None:
                results.append(done[ref.image_id])
                continue
            original = done[dup.original_id]
            results.append(
                ImageResult(
                    image_id=ref.image_id,
                    restaurant_id=ref.restaurant_id,
                    dishes=original.dishes,
                    rules_confidence=original.rules_confidence,
                    duplicate_of=dup.original_id,
                    duplicate_reason=dup.reason,
                )
            )
        write_jsonl(run_dir / "results.jsonl", results)
        return results


def merge_previous(config: RunConfig, results: list[ImageResult]) -> list[ImageResult]:
    """With ``reuse_from``: carry over the previous run's results for the images not re-run, so
    the new run covers the same image set and stays comparable with other runs."""
    if config.reuse_from is None:
        return results
    redone = {r.image_id for r in results}
    previous = read_jsonl(config.reuse_from / "results.jsonl", ImageResult)
    merged = [r for r in previous if r.image_id not in redone] + results
    return sorted(merged, key=lambda r: r.image_id)


def select_images(config: RunConfig) -> list[ImageRef]:
    """Images for this run: explicit ids, or the low-confidence ones of a previous run."""
    ids = config.image_ids
    if config.reuse_from is not None and config.only_low_confidence:
        previous = read_jsonl(config.reuse_from / "results.jsonl", ImageResult)
        low = [r.image_id for r in previous if r.rules_confidence < config.confidence_threshold]
        ids = [i for i in low if ids is None or i in ids]
        if not ids:
            return []
    return list(iter_images(config.input_path, ids))


def new_run_dir(config: RunConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    label = "ocr" if not config.uses_llm else f"llm-{config.llm_policy}-{config.llm_model}"
    run_dir = config.runs_dir / f"{stamp}_{label}"
    suffix = 1
    while run_dir.exists():  # two runs started in the same second
        suffix += 1
        run_dir = config.runs_dir / f"{stamp}_{label}_{suffix}"
    run_dir.mkdir(parents=True)
    write_yaml(run_dir / "config.yaml", config.model_dump(mode="json"))
    return run_dir
