"""Local, deterministic checks for a batch of generated titles (no AI request needed).

Detects exact duplicates, near-duplicates (shared vocabulary / almost the same word
sequence), repeated openings, repeated sentence skeletons, over-used words, titles
outside the word range, and titles that copy the original filename or a search-result
title. Only the *later* title of a problematic pair is flagged, so regeneration can be
limited to the titles that actually need it.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
STOPWORDS = frozenset(
    "a an the and or but of to in on at for with by from as into onto over under after before about against between "
    "through during without within is are was were be been being it its this that these those his her their our your "
    "my he she they we you i one all every each no not up out off down than then so too very just can will".split()
)


@dataclass(frozen=True, slots=True)
class Strictness:
    jaccard: float  # shared content words / all content words
    sequence: float  # difflib ratio of the word sequences
    opening_share: float  # share of the batch that may start with the same two words
    skeleton_share: float  # share that may use the same sentence skeleton
    word_share: float  # share of titles one content word may appear in


LEVELS = {
    "high": Strictness(jaccard=0.5, sequence=0.75, opening_share=0.04, skeleton_share=0.04, word_share=0.15),
    "medium": Strictness(jaccard=0.6, sequence=0.82, opening_share=0.10, skeleton_share=0.10, word_share=0.25),
    "low": Strictness(jaccard=0.75, sequence=0.9, opening_share=0.25, skeleton_share=0.25, word_share=0.45),
}


def words(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return _WORD.findall(folded)


def count_words(text: str) -> int:
    return len(words(text))


def content_words(text: str) -> set[str]:
    return {w for w in words(text) if w not in STOPWORDS and not w.isdigit()}


def skeleton(text: str) -> tuple[str, ...]:
    """Sentence shape: stop-words kept, every other word replaced by ``X`` (``a X X is X``)."""
    return tuple(w if w in STOPWORDS else "X" for w in words(text))


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _shares_run(a: list[str], b: list[str], length: int) -> bool:
    """True if ``a`` and ``b`` share ``length`` consecutive words."""
    if len(a) < length or len(b) < length:
        return False
    grams = {tuple(a[i : i + length]) for i in range(len(a) - length + 1)}
    return any(tuple(b[i : i + length]) in grams for i in range(len(b) - length + 1))


def resembles(title: str, source: str, *, min_run: int = 3, jaccard: float = 0.5) -> bool:
    """Does ``title`` copy or lightly paraphrase ``source`` (a filename or a search-result title)?"""
    t, s = words(title), words(source)
    if not t or not s:
        return False
    tc, sc = content_words(title), content_words(source)
    if len(sc) >= 2 and _jaccard(tc, sc) >= jaccard:
        return True
    if len(sc) >= 2 and len(tc & sc) >= max(2, math.ceil(len(sc) * 0.8)):
        return True
    return _shares_run(t, s, min_run) and len(s) >= min_run


@dataclass(slots=True)
class QualityReport:
    """Problems per title index (only titles that should be regenerated)."""

    problems: dict[int, list[str]] = field(default_factory=dict)

    def add(self, index: int, reason: str) -> None:
        reasons = self.problems.setdefault(index, [])
        if reason not in reasons:
            reasons.append(reason)

    @property
    def ok(self) -> bool:
        return not self.problems

    def exact_duplicates(self) -> list[int]:
        return [i for i, reasons in self.problems.items() if any(r.startswith("exact duplicate") for r in reasons)]


def check_titles(
    titles: Sequence[str],
    *,
    variation: str = "high",
    word_range: tuple[int, int] | None = None,
    check_similarity: bool = True,
    check_duplicates: bool = True,
    keywords: Sequence[str] = (),
    ignore_words: Sequence[str] = (),
    filenames: Sequence[str] | None = None,
    sources: Sequence[Sequence[str]] | None = None,
) -> QualityReport:
    """Validate a whole batch. ``titles`` are the generated titles *without* the series prefix.

    ``filenames[i]`` (when given) must not be copied into ``titles[i]``; ``sources[i]`` are
    search-result titles that must not be reproduced.
    """
    level = LEVELS.get(variation, LEVELS["high"])
    report = QualityReport()
    n = len(titles)
    ignore = {w for text in ignore_words for w in words(text)}
    toks = [words(t) for t in titles]
    contents = [content_words(t) - ignore for t in titles]
    normalised = [" ".join(t) for t in toks]

    for i, title in enumerate(titles):
        if not title.strip() or not toks[i]:
            report.add(i, "empty title")
            continue
        if word_range is not None:
            low, high = word_range
            count = len(toks[i])
            if count < low or count > high:
                report.add(i, f"has {count} words (needs {low}-{high})")
        if filenames is not None and i < len(filenames) and filenames[i] and resembles(title, filenames[i]):
            report.add(i, "copies the original filename")
        if sources is not None and i < len(sources):
            for src in sources[i]:
                if src and resembles(title, src, min_run=4, jaccard=0.6):
                    report.add(i, "copies a search-result title")
                    break

    if check_duplicates:
        seen: dict[str, int] = {}
        for i, key in enumerate(normalised):
            if key and key in seen:
                report.add(i, f"exact duplicate of title {seen[key] + 1}")
            elif key:
                seen[key] = i

    if check_similarity and n > 1:
        for i in range(n):
            for j in range(i):
                if normalised[i] == normalised[j] or not toks[i] or not toks[j]:
                    continue
                if _jaccard(contents[i], contents[j]) >= level.jaccard:
                    report.add(i, f"too similar to title {j + 1}")
                    break
                if SequenceMatcher(None, toks[i], toks[j], autojunk=False).ratio() >= level.sequence:
                    report.add(i, f"nearly the same wording as title {j + 1}")
                    break
        allowed_openings = max(1, math.floor(n * level.opening_share)) if n > 1 else n
        openings: Counter[tuple[str, ...]] = Counter()
        for i, tokens in enumerate(toks):
            opening = tuple(tokens[:2])
            if len(opening) < 2:
                continue
            openings[opening] += 1
            if openings[opening] > allowed_openings:
                report.add(i, f"repeats the opening “{' '.join(opening)}”")
        allowed_shapes = max(1, math.floor(n * level.skeleton_share))
        shapes: Counter[tuple[str, ...]] = Counter()
        for i, title in enumerate(titles):
            shape = skeleton(title)
            if len(shape) < 4 or all(part == "X" for part in shape):
                continue  # too short / no structure words to compare
            shapes[shape] += 1
            if shapes[shape] > allowed_shapes:
                report.add(i, "repeats the same sentence structure")
        keyword_words = {w for k in keywords for w in words(k)}
        base_allowed = max(3, math.ceil(n * level.word_share))
        usage: Counter[str] = Counter()
        for i, content in enumerate(contents):
            for word in sorted(content):
                usage[word] += 1
                allowed = base_allowed * 2 if word in keyword_words else base_allowed
                if usage[word] > allowed:
                    report.add(i, f"over-uses the word “{word}”")
    return report
