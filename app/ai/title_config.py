"""Title Generator configuration (what the user set in the Title Generator panel or a preset).

``apply_command_overrides`` lets the natural-language AI command override these
settings, e.g. *"Do not use the original filenames. Make them cinematic and
suspenseful, 10-16 words each. Start every title with Blue Bloods."*
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields, replace
from typing import Any

TITLE_MODES = ("independent", "partial", "connected")
SOURCES = ("ai_only", "search_ai", "search_only", "custom_prompt")
CATEGORIES = ("crime", "investigation", "mystery", "family", "drama", "action", "suspense", "justice", "police", "emotional", "custom")
STYLES = ("cinematic", "story", "dramatic", "suspenseful", "professional", "simple", "emotional", "youtube", "facebook", "custom")
TONES = ("neutral", "dramatic", "suspenseful", "emotional", "serious", "exciting", "professional", "custom")
CAPACITIES = {"short": (5, 8), "medium": (8, 12), "long": (12, 18), "custom": None}
VARIATIONS = ("low", "medium", "high")
QUERY_MODES = ("automatic", "filename", "filename_series", "custom")
MAX_RESULTS = (3, 5, 10)
MIN_WORDS, MAX_WORDS = 1, 40


@dataclass(slots=True)
class TitleConfig:
    """Every Title Generator option (also the content of a preset)."""

    enabled: bool = False
    mode: str = "independent"  # filename is only a file identifier
    source: str = "ai_only"
    category: str = "crime"
    custom_category: str = ""
    topic: str = ""
    keywords: str = ""
    style: str = "cinematic"
    custom_style: str = ""
    tone: str = "dramatic"
    custom_tone: str = ""
    capacity: str = "medium"
    min_words: int = 8
    max_words: int = 16
    prefix: str = ""
    use_prefix: bool = True
    prefix_counts: bool = False  # the series prefix does not count toward the word limit by default
    variation: str = "high"
    unique: bool = True
    avoid_filename: bool = True
    prevent_duplicates: bool = True
    custom_instructions: str = ""
    # --- Google Search options
    use_filename_lookup: bool = True  # filename may be used to *look up* facts - never as the title idea
    query_mode: str = "automatic"
    max_results: int = 5
    custom_query: str = ""
    compare_results: bool = True
    context_only: bool = True
    completely_new: bool = True
    search_fallback: bool = True  # search failed -> continue with AI only

    # ------------------------------------------------------------ derived
    @property
    def word_range(self) -> tuple[int, int]:
        fixed = CAPACITIES.get(self.capacity)
        if fixed:
            return fixed
        low, high = sorted((max(MIN_WORDS, self.min_words), min(MAX_WORDS, self.max_words)))
        return low, high

    @property
    def category_text(self) -> str:
        return self.custom_category.strip() if self.category == "custom" else self.category

    @property
    def style_text(self) -> str:
        return self.custom_style.strip() if self.style == "custom" else self.style

    @property
    def tone_text(self) -> str:
        return self.custom_tone.strip() if self.tone == "custom" else self.tone

    @property
    def keyword_list(self) -> list[str]:
        return [k.strip() for k in re.split(r"[,;\n،]", self.keywords) if k.strip()]

    @property
    def series_prefix(self) -> str:
        return self.prefix.strip() if self.use_prefix else ""

    @property
    def sends_filename(self) -> bool:
        """Whether the AI may see the filename as creative input (never in Independent Creative mode)."""
        return self.mode != "independent" and not self.avoid_filename

    @property
    def uses_search(self) -> bool:
        if self.source in ("search_ai", "search_only"):
            return True
        return self.source == "custom_prompt" and bool(_SEARCH_WORDS.search(self.custom_instructions))

    # --------------------------------------------------------- persistence
    def normalised(self) -> TitleConfig:
        out = replace(self)
        out.mode = _choice(out.mode, TITLE_MODES, "independent")
        out.source = _choice(out.source, SOURCES, "ai_only")
        out.category = _choice(out.category, CATEGORIES, "crime")
        out.style = _choice(out.style, STYLES, "cinematic")
        out.tone = _choice(out.tone, TONES, "dramatic")
        out.capacity = _choice(out.capacity, tuple(CAPACITIES), "medium")
        out.variation = _choice(out.variation, VARIATIONS, "high")
        out.query_mode = _choice(out.query_mode, QUERY_MODES, "automatic")
        out.max_results = min(MAX_RESULTS, key=lambda n: abs(n - int(out.max_results)))
        out.min_words = max(MIN_WORDS, min(int(out.min_words), MAX_WORDS))
        out.max_words = max(out.min_words, min(int(out.max_words), MAX_WORDS))
        if out.mode == "independent":
            out.avoid_filename = True
        for name in ("custom_category", "topic", "keywords", "custom_style", "custom_tone", "prefix", "custom_query"):
            setattr(out, name, str(getattr(out, name))[:300])
        out.custom_instructions = str(out.custom_instructions)[:2000]
        return out

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TitleConfig:
        """Tolerant loader: unknown keys are ignored, wrong types fall back to defaults."""
        base = cls()
        values: dict[str, Any] = {}
        for f in fields(cls):
            if data and f.name in data:
                default = getattr(base, f.name)
                value = data[f.name]
                if isinstance(default, bool):
                    values[f.name] = bool(value)
                elif isinstance(default, int):
                    try:
                        values[f.name] = int(value)
                    except (TypeError, ValueError):
                        continue
                else:
                    values[f.name] = str(value)
        return replace(base, **values).normalised()


def _choice(value: str, allowed: tuple[str, ...], default: str) -> str:
    return value if value in allowed else default


# ===================================================================== overrides
_SEARCH_WORDS = re.compile(r"\b(?:search|google|look\s*up|public information|online)\b|সার্চ|খুঁজ|গুগল", re.IGNORECASE)
_NO_FILENAME = re.compile(
    r"(?:do\s*n[o']t|never|without|don't)\s+(?:use|using|copy|copying|keep|keeping|rewrite|look at)\s+(?:the\s+)?(?:original\s+)?"
    r"(?:file\s*names?|names?)|ignore\s+(?:the\s+)?(?:original\s+)?file\s*names?|"
    r"(?:not|unrelated)\s+(?:related|connected|based)\s+(?:to|on)\s+(?:the\s+)?file\s*names?|"
    r"ফাইলের\s*নাম\s*(?:ব্যবহার|কপি)\s*(?:করবে|করো|করবেন)\s*না",
    re.IGNORECASE,
)
_FROM_FILENAME = re.compile(r"\b(?:based on|from|using|keep(?:ing)?)\s+(?:the\s+)?(?:original\s+)?file\s*names?\b", re.IGNORECASE)
_LOOSELY = re.compile(r"\b(?:loosely|partially|inspired by)\b", re.IGNORECASE)
_RANGE = re.compile(r"(\d{1,2})\s*(?:-|–|—|to|থেকে)\s*(\d{1,2})\s*(?:words?|শব্দ)", re.IGNORECASE)
_EXACT = re.compile(r"(\d{1,2})[\s-]*(?:words?|শব্দের?)\b", re.IGNORECASE)
_CAPACITY = re.compile(r"\b(short|medium|long)(?:er)?\s+titles?\b", re.IGNORECASE)
_PREFIX = re.compile(
    r"(?:start|begin|prefix)\s+(?:every|each|all|the)?\s*(?:titles?|names?|videos?)?\s*(?:with|by)\s+[\"“'‘]?(?P<p>[^\"”’'.,!\n]{2,60})",
    re.IGNORECASE,
)
_UNIQUE = re.compile(r"\b(?:completely\s+different|unique|different\s+titles?|no\s+duplicates?)\b|আলাদা|ভিন্ন", re.IGNORECASE)
_STYLE_WORDS = {s: re.compile(rf"\b{s}\b", re.IGNORECASE) for s in STYLES if s != "custom"}
_STYLE_WORDS["story"] = re.compile(r"\bstory(?:-like|telling)?\b", re.IGNORECASE)
_STYLE_WORDS["youtube"] = re.compile(r"\byou\s*tube\b", re.IGNORECASE)


def apply_command_overrides(config: TitleConfig, command: str) -> tuple[TitleConfig, list[str]]:
    """Return ``config`` changed by what the natural-language ``command`` asks for, plus a description of each change."""
    out = replace(config)
    changes: list[str] = []
    text = command or ""
    if _NO_FILENAME.search(text):
        out.mode, out.avoid_filename = "independent", True
        changes.append("mode=independent")
    elif _FROM_FILENAME.search(text):
        out.mode = "partial" if _LOOSELY.search(text) else "connected"
        out.avoid_filename = False
        changes.append(f"mode={out.mode}")
    matched = [name for name, rx in _STYLE_WORDS.items() if (m := rx.search(text))]
    matched.sort(key=lambda name: _STYLE_WORDS[name].search(text).start())  # type: ignore[union-attr]
    if matched:
        out.style = matched[0]
        changes.append(f"style={out.style}")
        tone = next((name for name in matched[1:] if name in TONES), None)
        if tone is None and matched[0] in TONES and len(matched) == 1:
            tone = matched[0]
        if tone:
            out.tone = tone
            changes.append(f"tone={tone}")
    if m := _RANGE.search(text):
        low, high = sorted((int(m.group(1)), int(m.group(2))))
        out.capacity, out.min_words, out.max_words = "custom", low, high
        changes.append(f"words={low}-{high}")
    elif m := _EXACT.search(text):
        n = int(m.group(1))
        out.capacity, out.min_words, out.max_words = "custom", max(1, n - 1), n + 1
        changes.append(f"words≈{n}")
    elif m := _CAPACITY.search(text):
        out.capacity = m.group(1).lower()
        changes.append(f"capacity={out.capacity}")
    if m := _PREFIX.search(text):
        prefix = m.group("p").strip().strip("\"“”'‘’").strip()
        if prefix:
            out.prefix, out.use_prefix = prefix, True
            changes.append(f"prefix={prefix}")
    if _UNIQUE.search(text):
        out.unique = out.prevent_duplicates = True
        if out.variation != "high":
            out.variation = "high"
        changes.append("unique")
    return out.normalised(), changes


# ================================================================== persistence
CURRENT_KEY = "title.current"


def load_current(db: Any) -> TitleConfig:
    """The Title Generator settings last used (auto-saved, non-sensitive)."""
    raw = db.get_raw(CURRENT_KEY)
    try:
        return TitleConfig.from_dict(json.loads(raw)) if raw else TitleConfig()
    except ValueError:
        return TitleConfig()


def save_current(db: Any, config: TitleConfig) -> None:
    db.set_raw(CURRENT_KEY, json.dumps(config.normalised().to_dict(), ensure_ascii=False))


def save_preset(db: Any, name: str, config: TitleConfig) -> None:
    data = config.normalised().to_dict()
    data.pop("enabled", None)  # a preset describes *how* to title, not whether the panel is on
    db.save_title_preset(name, data)


def load_preset(db: Any, name: str) -> TitleConfig | None:
    data = db.get_title_preset(name)
    return TitleConfig.from_dict(data) if data is not None else None

