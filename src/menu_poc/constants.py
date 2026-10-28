"""Every constant of the project, grouped by the component that uses it.

A leaf module: it imports nothing from ``menu_poc``, so any module can import it. Versions are
part of the cache keys: bump one when the output of its component changes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

# --- paths and defaults -------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_GOLDEN_DIR = PROJECT_ROOT / "data" / "golden"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "tmp"  # one sub-directory per run
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"  # stage cache, reused across runs
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DEFAULT_PROMPT_PATH = PROMPTS_DIR / "structure_v2.md"
CONFIG_PATH_FIELDS = ("input_path", "runs_dir", "artifacts_dir")

DEFAULT_CONFIDENCE_THRESHOLD = 0.6  # rules confidence below this sends the page to the LLM
DEFAULT_DEDUP_MAX_DISTANCE = 10  # perceptual hash bits out of 256
DEFAULT_MAX_IMAGE_SIDE = 2000

LLM_HELP = (
    "Add --llm-model haiku|sonnet|opus (needs ANTHROPIC_API_KEY in .env) or --llm-model ollama "
    "(free, local) to let an LLM handle the pages the rules are not confident about."
)

# --- input ----------------------------------------------------------------------------------------

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# --- duplicate detection --------------------------------------------------------------------------

DHASH_SIZE = 16  # 16x16 = 256-bit hash

# --- preprocessing and OCR ------------------------------------------------------------------------

PREPROCESS_VERSION = "1"
ORIENTATION_VERSION = "3"
TALL_RATIO = 1.5  # a box is "tall" when height > TALL_RATIO * width
TALL_SHARE = 0.5  # the page looks sideways when more than this share of text boxes is tall
MIN_CHARS = 3  # ignore very short boxes when judging orientation (digits, icons)
KEEP_UPRIGHT = 0.8  # keep rotation 0 if the best sideways pass scores below this share of it
DESKEW_MIN_DEG = 1.5  # re-read tilted pages from this median line angle
DESKEW_MAX_DEG = 15.0  # larger angles are not a tilt but mixed orientations: leave them
LONG_LINE = 4.0  # lines at least this many times wider than tall vote for the tilt angle
MIN_TILT_VOTES = 5  # fewer long lines than this are too few to judge the tilt
# PIL transposes rotate counter-clockwise; the key is the rotation applied to the image.
ROTATION_TRANSPOSES = {90: Image.Transpose.ROTATE_90, 270: Image.Transpose.ROTATE_270}

# --- page layout ----------------------------------------------------------------------------------

ROW_OVERLAP = 0.5  # min vertical overlap, as a share of the shorter box, to share a row
GUTTER_MIN_WIDTH = 1.0  # min gutter width between columns, in median line heights
BLOCK_OVERLAP = 0.2  # row overlap from which a line between two others separates them
GUTTER_MAX_COVER = 0.02  # an x position is a gutter when <= this share of lines cover it

# --- prices ---------------------------------------------------------------------------------------

MIN_PRICE = 0.5
MAX_PRICE = 500.0

_CURRENCY = r"(?:€|eur(?:os?)?\b)"
_NUMBER_TWO_DECIMALS = r"\d{1,3}[.,]\d{2}"
_NUMBER_ANY = r"\d{1,3}(?:[.,]\d{1,2})?"
# Not glued to other digits, separators or times. Letters are allowed: OCR often glues the
# price to the name ("Vindaloo12,95").
_BEFORE = r"(?<![\d,:/])(?<!\d\.)"
_AFTER = r"(?![\d%:/]|[.,]\d)"
_NO_UNIT = r"(?!\s?(?:cl|ml|l|kg|gr?|cm|uds?|unid\w*|pzs?|%)\b)"
# Per-portion suffix that belongs to the price: "11,95 €/8uds".
_PORTION = r"(?:\s?/\s?\d*\s?(?:uds?|unid\w*|pzs?|pcs?|piezas?|peces|unitats?)\b)?"

PRICE_RE = re.compile(
    rf"{_CURRENCY}\s?(?P<currency_first>{_NUMBER_ANY}){_AFTER}"  # € 8.50
    rf"|{_BEFORE}(?P<currency_last>{_NUMBER_ANY})\s?{_CURRENCY}{_PORTION}"  # 3€, 15,90 €
    rf"|{_BEFORE}(?P<no_currency>{_NUMBER_TWO_DECIMALS}){_AFTER}{_NO_UNIT}",  # 8,95
    re.IGNORECASE,
)
PRICE_GROUPS = ("currency_first", "currency_last", "no_currency")
# Spanish phone numbers: 9 digits split by spaces/dashes (93 444 47 37) or only by dots
# (934.44.47.37); mixing dots and spaces would also match a row of prices like "9.50 12.45".
PHONE_RE = re.compile(r"(?<!\d)(?:\+34\s?)?[6-9](?:(?:[ -]?\d){8}|(?:\.?\d){8})(?!\d)")
# A decimal number alone on its OCR line, at most with dotted leaders around it.
BARE_PRICE_RE = re.compile(r"^[.…_\-·•\s]*(\d{1,3}[.,]\d{1,2})[.…_\-·•\s]*$")
PORTION_NOTE_RE = re.compile(r"\([^()]{0,8}\)")  # "(8u)", "(2p)", "(4 pzs)"
PRICE_LEADERS_RE = re.compile(r"[.…_\-·•\s]{3,}")

# --- rule-based parser ----------------------------------------------------------------------------

RULES_VERSION = "3"
MIN_OCR_CONF = 0.6  # lines read with less confidence are treated as noise
MIN_LETTERS = 3  # a line with fewer letters is no name, header or description
HEADER_MAX_WORDS = 5
HEADER_MAX_CHARS = 40
CAPS_SHARE = 0.8  # share of uppercase letters for a line to read as ALL CAPS
HEADER_HEIGHT = 1.3  # a line this many median heights tall reads as a header
TALL_HEADER_MAX_WORDS = 3
DESC_MAX_GAP = 1.2  # max vertical gap (median heights) between a dish line and its description
DESC_MAX_LINES = 6
MAX_NAME_CHARS = 70
FRAGMENT_LETTERS = 4  # a name with this many letters or fewer is likely cut or misread
NUMBERED_OVERLAP = 0.3  # row overlap enough to give a price to a numbered dish name
DESCRIPTIVE_WORDS = 6  # a priced line this long reads as a description, not a name
DECORATION_REPEATS = 3  # a header text repeated this often is a logo or slogan, not a section
PRICE_COLUMN_SHARE = 0.7  # a column with this share of price-only lines is a price column
ORPHAN_LOOKBACK = 3  # recent dishes an orphan price may belong to
CONTINUATION_MIN_WORDS = 4  # a lowercase line this long continues the sentence above

DISH_NUMBERING_RE = re.compile(r"^\s*[.·•]?\s*(\d{1,3})\s*(?:[.)\-:]\s*|\s+(?=[A-ZÀ-Ý]))")
LEADING_JUNK_RE = re.compile(r"^[\s.,:;·•\-_*|/]+")
TRAILING_JUNK_RE = re.compile(r"[\s.,:;·•\-_*|/(]+$")
NOISE_RE = re.compile(
    r"www\.|https?://|\S+@\S+|\.(?:com|es|cat|net)\b"  # web, email
    r"|\biva\b|i\.v\.a|impuestos|\btax(es)?\b"  # VAT notes
    r"|al[·.]?l?[eè]rg[eè]n|allergen|alérgen"  # allergen legends
    r"|whats?app|instagram|facebook|wi-?fi|domicilio|delivery|pedidos?\b"
    r"|horario|abierto|obert|opening|\bhours\b|tarjeta|efectivo|cheques?",
    re.IGNORECASE,
)
# A "name" that is only a size label belongs to a size table, not to a dish.
SIZE_RE = re.compile(
    r"^(?:peque[ñn]as?|medianas?|grandes?|familiar|small|medium|large|petita|mitjana|gran"
    r"|individual|media|entera|1/2|tapa|raci[oó]n|ració)$",
    re.IGNORECASE,
)
HOURS_RE = re.compile(r"\d{1,2}\s*[:h]\s*\d{2}\s*(?:-|–|a|to|h)\s*\d{1,2}\s*[:h]?\s*\d{0,2}", re.I)
# Diet / allergen codes read as a box of their own next to the name: "O.V.F.S.", "s.a.", "KSG".
DIET_CODE_RE = re.compile(r"^(?:[A-Za-z]\s?\.\s?){2,}[A-Za-z]?\.?$|^[B-DF-HJ-NP-TV-XZ]{2,5}$")
# A bulleted list item ("· MARGARITA") is a dish name like a numbered one; "·PLATOS SECOS·" is not.
BULLETS = "·•▪●◦"
BULLET_RE = re.compile(rf"^\s*[{BULLETS}]\s*[A-ZÀ-Ý]")
NON_ALNUM_ASCII_RE = re.compile(r"[^a-z0-9]")

# --- rules confidence score -----------------------------------------------------------------------

CONFIDENCE_WEIGHTS = {
    "price_attach": 0.40,
    "priced_dishes": 0.20,
    "coverage": 0.20,
    "ocr_conf": 0.20,
}
CONFIDENCE_GATES = ("volume", "density", "recall_proxy", "name_gate", "header_gate")
MIN_DISHES = 4  # fewer dishes than this scales the score down linearly
MIN_DENSITY = 0.12  # expected dishes per content line (a dish spans ~1-6 lines with translations)
MIN_PRICE_TOKENS = 3  # with fewer prices on the page the recall proxy does not apply
MAX_SUSPICIOUS_NAMES = 0.2  # share of description-like / fragment names tolerated
MAX_HEADERS_PER_DISH = 0.8  # section titles per dish tolerated

# --- LLM ------------------------------------------------------------------------------------------

# USD per million tokens, (input, output): standard rates for prompts up to 100K input tokens
# (a menu page is ~3K). Batch API (-50%) and prompt-caching discounts are not modelled.
MODEL_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
ANTHROPIC_RETRYABLE_STATUS = (408, 409, 429)  # plus every 5xx
OLLAMA_PREFIX = "ollama/"
OLLAMA_DEFAULT_HOST = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "qwen2.5:7b"
CHARS_PER_TOKEN = 4  # rough estimate, good enough for the fake engine's usage figures

RENDER_VERSION = "lines_v1"  # bump when render_lines / the user message format changes
_NULLABLE_STRING = {"anyOf": [{"type": "string"}, {"type": "null"}]}
DISHES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "dishes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": _NULLABLE_STRING,
                    "prices": {"type": "array", "items": {"type": "number"}},
                    "section": _NULLABLE_STRING,
                    "source_line_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["name", "description", "prices", "section", "source_line_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["dishes"],
    "additionalProperties": False,
}

# Lowercase text: leading numbering such as "12.", "a)", "46 ".
GROUNDING_NUMBERING_RE = re.compile(r"^\s*(?:\d{1,3}|[a-z])\s*[.)\-:]\s*|^\s*\d{1,3}\s+(?=[^\W\d])")
NON_ALNUM_RE = re.compile(r"[\W_]+")  # Unicode-aware: keeps Greek, Cyrillic, CJK, Devanagari...

# --- grounding ------------------------------------------------------------------------------------

GROUNDING_VERSION = "v1"
GROUNDING_THRESHOLD = 85.0  # min rapidfuzz score for an LLM dish name to count as read
SHORT_NAME_CHARS = 3  # names this short must equal a whole word of the OCR text
MIN_WORD_CHARS = 3  # name words this long must each appear in the OCR text
BELOW_MAX_GAP = 1.5  # max gap, in line heights, to the line right below (names on two lines)
BELOW_MIN_DROP = 0.25  # the line below starts at least this share of a line height lower

# --- evaluation -----------------------------------------------------------------------------------

MATCH_THRESHOLD = 85.0  # min similarity for a predicted name to match a golden dish
TIE_BREAK_SCALE = 1000  # plain edit similarity / this breaks ties between equal scores
SPLITS: tuple[str, ...] = ("all", "dev", "test")
TOP_ERRORS = 5  # unmatched gold dishes / false positives listed per image

MATCH_NUMBERING_RE = re.compile(r"^\s*(?:n[ºo°]\s*)?\d{1,3}\s*[.)\-:]\s*|^\s*\d{1,3}\s+(?=\D)")
MATCH_PRICE_RE = re.compile(
    r"\d+(?:[.,]\d{1,2})?\s*(?:€|eur\b|euros?\b)|(?:€|\$)\s*\d+(?:[.,]\d{1,2})?"
)
# Only decimal numbers: "lamb curry 12,50" is a price, but "combo 2" is part of the name.
TRAILING_NUMBER_RE = re.compile(r"\s\d+[.,]\d{1,2}\s*$")
MATCH_LEADERS_RE = re.compile(r"[.·…_\-]{2,}")
NON_WORD_RE = re.compile(r"[^\w\s]")
SPACES_RE = re.compile(r"\s+")
