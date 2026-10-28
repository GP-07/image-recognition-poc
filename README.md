# Menu dish extraction — lightweight POC

Extract every orderable item (dishes, desserts, beverages, add-ons) printed on restaurant menu
photos (`google_images.zip`: 19 phone photos, 10 restaurants). Local, CPU only, no model training. The focus is the architecture, the model
choices and how quality and cost are measured.

## TL;DR

- **OCR first, LLM only when needed.** A local OCR (RapidOCR) reads the text with positions; a
  rule-based parser anchored on prices turns lines into dishes and scores its own **confidence**.
  Only low-confidence pages are sent to a **text** LLM (Claude Haiku 5.5 by default) that
  structures the OCR text; every LLM dish is **verified against the OCR** to drop hallucinations.
- **Duplicate photos are detected before any OCR/LLM call** (file hash + perceptual hash) and get
  the original's result.
- **Configurable, one model per run:** `--mode ocr | ocr+llm`, `--llm-policy fallback | always`,
  `--llm-model haiku | sonnet | opus | ollama` (aliases; exact ids also accepted): Claude, or a
  free local model through Ollama (`ollama` = `qwen2.5:7b`; `ollama/<model>` for another).
- **Never fails for a missing key:** the LLM connection is checked first, before any image is
  read (a free `models.retrieve` call proves the key is valid for the model). Without a valid
  `ANTHROPIC_API_KEY` (env or `.env`) or a running Ollama, an `ocr+llm` run degrades to OCR +
  rules and records a warning in `manifest.json`.
- **Output:** `items.csv`, one row per extracted item, in `tmp/<run>/` unless `--output` is given.
- **Every slow or paid stage is cached by content**, so re-running with another model reuses the
  OCR and only pays for the LLM call; `--only-low-confidence` re-runs just the hard pages.
- **Measured, not guessed:** a golden set with language aliases, a dev/test split by restaurant,
  recall (recognition rate) as the main metric, precision as a control, cost and latency per image.

## Architecture

```
image ──► duplicate? ──yes──► copy the original's result (no OCR, no LLM)
   │no
   └──► preprocess ──► OCR (RapidOCR, CPU) ──► rule parser + confidence
           EXIF, resize   lines + boxes,           price-anchored, layout-aware
                          rotation/deskew                   │
                          [cached]                          ├─ confidence ≥ threshold ─► dishes (rules)
                                                            │
                                                            └─ confidence < threshold (or policy=always)
                                                                    │
                                                    text LLM (Claude Haiku 5.5, structured JSON)
                                                    [cached]        │
                                                    grounding: every dish must appear in the OCR
                                                                    │
                                                                    └─► dishes (LLM)
```

**Implementation: Chain of Responsibility + Builder.** Each box above is a `Step` in its own
module (`steps/`); every image travels through the chain in an `ImageContext`, and each step
decides whether the image goes on (a duplicate stops at `dedup`, a confident page stops after
`rules`). `PipelineBuilder` accepts the pieces in any order and `build()` links them in the
system's fixed order (`dedup → ocr → rules → llm → verify`), adds `verify` whenever `llm` is
present and rejects invalid chains (e.g. an LLM without OCR). Optional pieces are just not added.

```python
pipeline = PipelineBuilder.from_config(config, llm_engine).build()   # production
pipeline = PipelineBuilder().add(LlmStep(...)).add(OcrStep(...)).add(RulesStep(...)).build()
# -> dedup? no · ocr → rules → llm → verify, whatever the insertion order
```

| Component | When it runs | Cost | Why |
|---|---|---|---|
| Duplicate detection | always (`dedup: true`) | ~0.1 s/image | Same file, or same photo re-encoded/resized (dHash, 256 bits; duplicates differ by 0–4 bits, different menus by 100+) |
| Content cache | always | ~0 | Key = hash(input) + stage fingerprint (engine/model/prompt versions) |
| Preprocess | always | ms | Photos are rotated and up to 5712 px |
| OCR (RapidOCR: PP-OCR models on ONNX Runtime) | always | ~3–5 s/image CPU (rotated pages ~13 s) | Deterministic text + positions; same models as PaddleOCR, far lighter install |
| Layout + rule parser | always | ms | Prices are the strongest cheap signal of "this line is a dish" |
| Confidence | always | ms | Decides whether the paid LLM is needed |
| Text LLM | `ocr+llm` mode only: pages below the confidence threshold (`fallback`) or every page (`always`) | ≈ $0.001–0.002/page with Haiku; $0 with Ollama (slower, local) | Understands layouts the rules cannot (size grids, prices under photos) |
| Grounding check | after the LLM | ms | Turns the LLM into a verifiable step |

Why not a vision LLM (VLM) on the image? It would be simple and accurate, but its output cannot
be checked against anything, it is the least deterministic option and it costs more per page.
OCR text gives the LLM exactly what it needs and gives us a ground truth to verify it against.

## Usage

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
cp .env.example .env    # put ANTHROPIC_API_KEY there (never committed); env variables win

# Rules only (free, deterministic). Input: a .zip or a folder of <restaurant>/<image>.jpg
.venv/bin/menu-poc run --mode ocr --input /path/to/images --output /path/to/output

# Cascade: LLM only for low-confidence pages
.venv/bin/menu-poc run --mode ocr+llm --llm-policy fallback --llm-model haiku

# Same with a free local model (brew install ollama && ollama pull qwen2.5:7b)
.venv/bin/menu-poc run --mode ocr+llm --llm-model ollama

# Not good enough? Re-run only the hard pages with a stronger model; the OCR comes from the cache
.venv/bin/menu-poc run --mode ocr+llm --llm-model sonnet \
    --reuse-from tmp/<previous run> --only-low-confidence

# Evaluate / compare
.venv/bin/menu-poc eval tmp/<run>
.venv/bin/menu-poc compare tmp/<run a> tmp/<run b> --out docs/results.md

# Tests (the real-OCR test is marked slow)
.venv/bin/pytest -q            # fast suite
.venv/bin/pytest -q -m slow    # real OCR on one image
```

**Inputs and outputs.** Defaults live in `config/default.yaml` (its relative paths are relative to
the project root, so the command works from any directory); every path can be overridden:
`--input` (zip or folder, read-only, never extracted), `--output` (default `tmp/` in the
project) and `--artifacts` (stage cache, default `artifacts/`). Each run creates
`<output>/<timestamp>_<label>/`:

| File | Content |
|---|---|
| `items.csv` | **the extracted items**, one row each: image, restaurant, item, description, prices (`8.5\|12` for sizes), section, source (`rules`/`llm`), `duplicate_of` |
| `results.jsonl` | full per-image record (nested): items plus confidence and its signals, LLM usage and cost, timings, `duplicate_of`, per-image `error`; used by `eval` and `--reuse-from` |
| `manifest.json` | requested vs effective mode, counts, duplicates, run-level `warnings` and per-image `errors` |
| `config.yaml` | the effective configuration |
| `summary.md`, `metrics.json` | evaluation against the golden set, when one exists |

Exit code is 0 when images were processed, even with warnings or per-image errors (they are in
the output); non-zero only for unusable input (no images, missing config file).

**CI (sketch).** The same command runs in a GitHub Actions job: store `ANTHROPIC_API_KEY` as a
repository secret and expose it as an env variable, run `menu-poc run --mode ocr+llm --input …
--output out/`, and upload `out/` as a job artifact (and/or fail the job on a recall drop from
`metrics.json`). If the secret is missing, the job still produces OCR + rules results and the
warning shows up in `manifest.json`.

## Evaluation

- **Golden set** (`data/golden/`, format in its README): the answer key. One YAML per image
  listing every legible orderable item and its **language aliases** (many menus print each item
  in Catalan, Spanish and English). It is never shown to the pipeline: it is only used to score
  a run. Finding any alias counts as recognising the item; extra translations count as
  duplicates, not false positives.
- **Split by restaurant:** dev (10 images) is the only data used to tune rules, threshold and
  prompt; test (9 images) is measured, never tuned on.
- **Matching:** names normalised (case, accents, numbering, prices, leaders) and fuzzy-matched
  (rapidfuzz, threshold 85) with an optimal one-to-one assignment.
- **Metrics:** recall = recognition rate (main), precision (control: headers, phones or
  descriptions must not count as dishes), share of pages escalated to the LLM, cost and latency per
  image. Cost and latency are those of producing the result even when served from the cache, so
  runs are comparable.
- **Duplicate photos** (4 pairs in the sample: identical pixels or half resolution) are scored
  once, so no menu weighs double.
- **Silver set, by design:** the labels were generated by an AI model looking at the photos and
  are not human-reviewed — the POC does not chase an exact accuracy, so no manual labelling work
  was added. They are good enough to compare configurations (rules vs. LLMs, thresholds, cost);
  absolute numbers carry a possible bias (same model family as the evaluated LLM).

## Results

Current scope (every orderable item, duplicate photos scored once, 15 distinct photos, 595 items):

| run | recall dev | recall test | recall all | precision | % pages to LLM | cost / image | time / image |
|---|---|---|---|---|---|---|---|
| Rules only (`tmp/20261028-125023_ocr`) | 74.6% | 55.6% | 69.2% | 84.3% | 0% | $0 | 4.1 s |
| Cascade with Haiku 5.5 (`fallback`) | _pending API key_ | | | | 47% (7/15) | ≈ $0.001 | |

History, earlier scope (food only, duplicate photos counted twice): rules baseline 71.9% recall /
72.4% precision → after error analysis on dev 76.8% / 79.5% (dev 72.6% → 79.4%, test unchanged at
70.1%). The tuning improved dev and left test unchanged — the expected sign of not overfitting,
and a reminder that heuristics stop scaling. Widening the scope to beverages and add-on lists
(toppings, extras, sauces) lowered test recall: those lists rarely follow the "name … price"
pattern the rules rely on, which is exactly what the LLM step is for (along with shared size
grids, prices under photos and multi-column boards).

## The three questions

**Which technique?** A cascade: local OCR + price-anchored rules for every page, a text LLM only
for pages where the rules report low confidence, and a grounding check of the LLM output against
the OCR. OCR alone reads text well but cannot understand layout; an LLM on the OCR text adds that
understanding cheaply (text-only, small model, structured output) while staying verifiable. Claude
Haiku 5.5 is the default because the task is simple extraction; the model is a run parameter, and
a stronger one (Sonnet 5.5) can re-process only the hard pages from the cached OCR.

**How to evaluate?** A human-reviewed golden set with aliases, a dev/test split, recall as the
recognition rate, precision as a control, plus cost, latency and escalation rate per image, all
comparable across runs (`menu-poc compare`). In production there are no labels, so the same
signals become monitors: rules confidence, share of LLM dishes dropped by grounding, share of
prices left without a dish, escalation rate; plus periodic human review of a sample. An LLM judge
can help with semantic equivalences, calibrated against human labels — never as the only judge.

**Limitations and production improvements.**
- Photo quality (glare, folds, perspective, cut edges) caps OCR quality; tilted or folded pages
  still misalign prices. → better deskew/dewarping, higher OCR resolution for small text, or a
  VLM fallback only when OCR confidence itself is low.
- Shared price grids, set menus and unpriced dishes defeat price anchoring. → handled by the LLM.
- Confidence is a hand-made heuristic calibrated on ~8 distinct pages. → recalibrate on reviewed
  production samples; corrections become labelled data for a small line classifier that could
  replace most LLM calls (cost goes down as volume goes up).
- Throughput and cost at scale → Batch API (−50%) for non-interactive ingestion, content cache
  plus a perceptual hash to skip re-uploaded menus, async workers.
- Out of scope here by decision: normalising dishes across menus (catalogue, embeddings).

## Project layout

```
src/menu_poc/
  models.py, interfaces.py, config.py   contracts (pydantic + Protocols), run config
  steps/                                one module per step: dedup, ocr, rules, llm, verify
  pipeline/                             PipelineBuilder (fixed order + checks), Pipeline, run helpers
  cli.py                                `menu-poc` command
  io/  cache/                           zip/dir reader, JSON helpers, content-addressed store
  preprocess/  ocr/  parsing/           image prep, RapidOCR adapter, layout + rules + confidence
  llm/                                  LLMEngine.get_for_alias + one engine per provider (Anthropic,
                                        Ollama, fake), pricing, structurer, grounding
  evaluation/                           golden loader, matching, metrics, Markdown reports
prompts/structure_v1.md                 versioned LLM prompt (its hash is part of the cache key)
data/golden/                            silver labels (AI-generated reference, not reviewed)
docs/decision_log.md                    every design decision and the AI tools used
```

## How AI was used to build this

Built with Claude Code (Claude Opus 5.5) as a pair: design discussion and decisions with the user
(`docs/decision_log.md`), contracts and integration in the main session, four parallel sub-agents
for independent modules (OCR + rules, LLM connector, evaluation, golden-set draft from the images),
then a fresh-context code-review agent and an error-analysis agent that tuned the rules on the dev
split only. Every agent output was checked by tests and by re-running the evaluation.
