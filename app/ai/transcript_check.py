"""Checks that a suggested title is supported by the video's transcript (no invented claims).

The AI returns every suggestion with ``evidence`` (words copied from the transcript) and ``names``
(people, places, products the title mentions). A suggestion is accepted only when

* the evidence (when given) really occurs in the transcript (allowing for small recognition/inflection differences),
* the title's main subject words appear in the transcript,
* every number and every declared name in the title appears in the transcript,
* it is not a generic title ("Amazing Video", "New Video" ...), and
* it is not a near-duplicate of a suggestion already accepted for the same video.

Works for English, Bangla and mixed text (Unicode words, Bangla digits normalised).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from app.ai.title_quality import STOPWORDS, _jaccard, content_words, words

_BANGLA_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
BANGLA_STOPWORDS = frozenset(
    "এবং ও আর কি কী কে না এই সেই ওই একটি একটা এক থেকে জন্য হয় হবে হলো হল করে করা করেন করছে আছে ছিল তার তাঁর তিনি "
    "আমি আমরা তুমি আপনি সে যে এর যা কিন্তু সব কোন কোনো কেন কীভাবে কেমন নিয়ে দিয়ে মধ্যে পরে আগে খুব শুধু তো ভাবে".split()
)
GENERIC_TITLES = (
    "amazing video", "interesting moments", "new video", "must watch", "you won't believe", "watch till the end",
    "full video", "viral video", "best video", "funny video",
)  # fmt: skip
MIN_EVIDENCE_OVERLAP = 0.8


@dataclass(frozen=True, slots=True)
class Suggestion:
    title: str
    evidence: str = ""
    names: tuple[str, ...] = ()


def normalise(text: str) -> str:
    return " ".join(words(unicodedata.normalize("NFKC", text).translate(_BANGLA_DIGITS)))


def _stem(word: str) -> str:
    return word[:5] if len(word) > 5 else word  # tolerant of inflection: "investigation" ~ "investigate"


class TranscriptIndex:
    """Normalised transcript text, its words and word stems."""

    def __init__(self, transcript: str) -> None:
        self.text = normalise(transcript)
        self.words = set(self.text.split())
        self.stems = {_stem(w) for w in self.words}
        self.numbers = set(_NUMBER.findall(unicodedata.normalize("NFKC", transcript).translate(_BANGLA_DIGITS)))

    def has_word(self, word: str) -> bool:
        return word in self.words or (len(word) >= 4 and _stem(word) in self.stems)

    def has_phrase(self, phrase: str) -> bool:
        norm = normalise(phrase)
        if not norm:
            return False
        if f" {norm} " in f" {self.text} ":
            return True
        tokens = norm.split()
        return len(tokens) >= 2 and sum(self.has_word(t) for t in tokens) / len(tokens) >= MIN_EVIDENCE_OVERLAP


def _subject_words(title: str, ignore: Sequence[str]) -> list[str]:
    ignored = {w for text in ignore for w in words(text)}
    return [w for w in content_words(title) if w not in BANGLA_STOPWORDS and w not in ignored and len(w) >= 2]


def problem(suggestion: Suggestion, index: TranscriptIndex, *, ignore: Sequence[str] = ()) -> str | None:
    """Why ``suggestion`` is not supported by the transcript (``None`` = accepted).

    ``ignore``: words the user supplied (series prefix, keywords, topic) - not required in the transcript.
    """
    title = suggestion.title.strip()
    lowered = title.casefold()
    if not title:
        return "empty title"
    if any(generic in lowered for generic in GENERIC_TITLES) or len(_subject_words(title, ignore)) == 0:
        return "too generic"
    if suggestion.evidence.strip() and not index.has_phrase(suggestion.evidence):
        return "the quoted words are not in the transcript"  # a made-up quote means a made-up title
    subject = _subject_words(title, ignore)
    if not any(index.has_word(w) for w in subject):
        return "its subject does not appear in the transcript"
    for number in _NUMBER.findall(unicodedata.normalize("NFKC", title).translate(_BANGLA_DIGITS)):
        if number not in index.numbers:
            return f"the number {number} is not in the transcript"
    for name in suggestion.names:
        tokens = [w for w in normalise(name).split() if w not in STOPWORDS]
        if tokens and not all(index.has_word(t) for t in tokens):
            return f"'{name}' is not mentioned in the transcript"
    return None


def near_duplicate(title: str, others: Sequence[str]) -> bool:
    mine = content_words(title)
    return any(_jaccard(mine, content_words(other)) >= 0.6 or normalise(title) == normalise(other) for other in others)


def validate_suggestions(suggestions: Sequence[Suggestion], transcript: str, *, ignore: Sequence[str] = ()
                         ) -> tuple[list[Suggestion], list[tuple[str, str]]]:  # fmt: skip
    """Split into (accepted, [(title, reason)] rejected); accepted keeps the AI's order."""
    index = TranscriptIndex(transcript)
    accepted: list[Suggestion] = []
    rejected: list[tuple[str, str]] = []
    for suggestion in suggestions:
        reason = problem(suggestion, index, ignore=ignore)
        if reason is None and near_duplicate(suggestion.title, [s.title for s in accepted]):
            reason = "nearly the same as another suggestion"
        if reason is None:
            accepted.append(suggestion)
        else:
            rejected.append((suggestion.title, reason))
    return accepted, rejected
