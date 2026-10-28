# Menu item extraction — proof of concept

## Goal

This proof of concept takes a set of menu photos and turns each one into a list of the things a customer
can order — dishes, desserts, drinks and extras — with their descriptions and prices. 

## How it works

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

Each photo goes through a short chain of steps. First, the system checks whether the photo is a
copy of one it has already seen (the same file, or the same picture resized); copies simply reuse
the earlier result. Then a text-recognition engine (OCR) reads every line of text in the photo
along with its position on the page, straightening pictures taken sideways or at an angle. A set
of hand-written rules turns those lines into menu items, using the prices as anchors: a line with
a price next to it is very likely an item. The rules also rate how well they understood the page,
as a confidence score between 0 and 1.

When the rules are confident, their answer is final and the photo costs nothing. When they are
not — a price grid shared by several pizzas, prices printed under pictures, a crowded board — the
text read by the OCR is sent to a language model (LLM), which is good at understanding layouts.
The language model only sees the text, never the photo, and every item it returns is checked
against that text: anything that was not actually printed on the menu is discarded. This keeps
the model from inventing dishes. Every slow or paid step is saved to a cache, so running the same
photos again, even with a different model, does not repeat work that is already done.

The language model is optional, and you choose which one when you run the command:

- **No model** (default): only OCR and rules. Free, fast and always gives the same result.
- **`haiku`**, **`sonnet`** or **`opus`**: Anthropic's Claude models, from cheapest to most
  capable. Haiku costs about a tenth of a cent per page and is the recommended choice; they need
  an Anthropic API key.
- **`ollama`**: a free model that runs on your own computer through Ollama (Qwen 2.5, 7B by
  default; `ollama/<name>` picks another one). Nothing leaves the machine, but it is slower.

If the chosen model cannot be used (no key, invalid key, Ollama not running), the run does not
fail: it carries on with OCR and rules and says why in its output.

## How to run it

Install it once (Python 3.10 or newer):

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Every run takes `--input`, a `.zip` file or a folder that holds one sub-folder per restaurant,
and `--output`, the folder where the results go. The simplest run uses only OCR and rules:

```bash
.venv/bin/menu-poc run --input path/to/menus.zip --output path/to/results
```

With a free local model through Ollama, install Ollama, download the model and pass
`--llm-model ollama`:

```bash
brew install ollama && ollama serve &
ollama pull qwen2.5:7b
.venv/bin/menu-poc run --input path/to/menus.zip --output path/to/results --llm-model ollama
```

With a Claude model, copy `.env.example` to `.env`, write your key in it
(`ANTHROPIC_API_KEY=...`) and pick the model:

```bash
cp .env.example .env
.venv/bin/menu-poc run --input path/to/menus.zip --output path/to/results --llm-model haiku
```

Without `--input` and `--output` the command uses the sample photos and the project's `tmp/`
folder. Other settings (which pages go to the language model, the confidence threshold) live in
`config/default.yaml`.

## Output files

Each run creates its own folder, named after the date, time and model used, inside the
`--output` folder. It holds:

| File | What it shows |
|---|---|
| `items.csv` | The result: one row per item found, with the photo and restaurant it comes from, its name, description, prices, menu section, whether it came from the rules or the language model, and whether the photo was a copy of another one. Opens in any spreadsheet. |
| `summary.md` | How good the run was, measured against reference answers (`data/golden/`): how many of the real items were found, how many found items were wrong, cost and time, photo by photo, with the items that were missed. |
| `manifest.json` | A short record of the run: when it ran, which model was asked for and which one was actually used, how many photos and items, and any warnings or errors. |
| `results.jsonl` | The full detail per photo (confidence, costs, timings), for anyone who wants to dig deeper. |
| `metrics.json`, `config.yaml` | The evaluation figures and the exact settings used, in machine-readable form. |

To compare several runs side by side (for example, rules only against Haiku against Sonnet), use
`.venv/bin/menu-poc compare path/to/results/<run 1> path/to/results/<run 2>`.

The design decisions, the evaluation method, the current results and the answers to the test
questions are in [docs/technical_notes.md](docs/technical_notes.md); the full history of
decisions and of the AI tools used is in [docs/decision_log.md](docs/decision_log.md).
