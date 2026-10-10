"""Offline recognition of simple, deterministic commands (English and Bangla).

Anything that can be expressed as one plain rule - "remove 1080p from all
filenames", "add prefix X", "number files from 01" - is turned into actions here,
so *no AI request is made*. The parser is deliberately conservative: it only
fires when the **whole** sentence fits a known pattern. Anything ambiguous
returns ``None`` and is handed to the AI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.ai.schemas import (
    Action,
    AddPrefixAction,
    AddSuffixAction,
    ChangeCaseAction,
    NumberingAction,
    RemoveTextAction,
    ReplaceTextAction,
    SortAction,
)
from app.i18n import tr

_FLAGS = re.IGNORECASE | re.UNICODE | re.DOTALL
_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’", "`": "`", "«": "»"}
_QUOTED = re.compile(r"""(?:"([^"]+)"|“([^”]+)”|'([^']+)'|‘([^’]+)’|`([^`]+)`)""")
_TRAILING_PUNCT = re.compile(r"[\s.।!?,;:]+$")
_GENERIC_WORDS = frozenset(
    {"all", "every", "each", "file", "files", "video", "videos", "everything", "name", "names", "filename",
     "filenames", "these", "them", "it", "duplicates", "duplicate", "unused", "extra", "empty", "folder",
     "folders", "সব", "ফাইল", "ভিডিও", "নাম"}
)  # fmt: skip
_SYMBOL_WORDS = {
    "underscore": "_", "underscores": "_", "dot": ".", "dots": ".", "period": ".", "periods": ".",
    "dash": "-", "dashes": "-", "hyphen": "-", "hyphens": "-", "space": " ", "spaces": " ",
    "comma": ",", "commas": ",", "আন্ডারস্কোর": "_", "ডট": ".", "ড্যাশ": "-", "হাইফেন": "-", "স্পেস": " ",
}  # fmt: skip
_EMPTY_WORDS = frozenset({"nothing", "empty", "blank", "none", "no text", "কিছু না", "খালি"})
_SCOPE_WORDS = re.compile(r"(?:file\s*names?|names?|titles?|videos?|files?|নাম|ভিডিও|ফাইল)", _FLAGS)
_ALL = r"(?:all|every|each|the|these|my|selected|those|everything)"
_VERB_REMOVE = r"(?:remove|delete|strip|erase|drop|cut(?:\s+out)?|take\s+out|get\s+rid\s+of|clean\s+out)"
_SORT_KEYS = {
    "name": "name", "title": "name", "alphabet": "name", "alphabetically": "name", "size": "size", "date": "date",
    "time": "date", "modified": "date", "type": "type", "extension": "type", "duration": "duration",
    "length": "duration", "নাম": "name", "আকার": "size", "সাইজ": "size", "তারিখ": "date", "সময়": "date",
    "ধরন": "type", "সময়কাল": "duration",
}  # fmt: skip


@dataclass(frozen=True, slots=True)
class LocalPlan:
    """Actions derived without AI plus a short human description for the status line."""

    actions: list[Action]
    description: str


# ----------------------------------------------------------------- helpers
def _clean(command: str) -> str:
    return _TRAILING_PUNCT.sub("", re.sub(r"\s+", " ", command.strip()))


def _unquote(text: str) -> tuple[str, bool]:
    """Return ``(text, was_quoted)`` - quoted text keeps inner spaces, unquoted is stripped."""
    text = text.strip()
    if len(text) >= 2 and text[0] in _QUOTE_PAIRS and text[-1] == _QUOTE_PAIRS[text[0]]:
        return text[1:-1], True
    return text, False


def _split_items(text: str) -> list[str]:
    """Split ``"1080p, WEB-DL and x264"`` into items (quoted segments win when present)."""
    quoted = [next(g for g in m.groups() if g is not None) for m in _QUOTED.finditer(text)]
    if quoted:
        return [q for q in quoted if q]
    # NOTE: no \b here - Python treats Bengali vowel signs as non-word characters, so \b breaks on them.
    parts = re.split(r"\s*[,;&]\s*|\s+(?:and|এবং|আর|ও)\s+", text, flags=re.IGNORECASE)
    return [_unquote(p)[0].strip() for p in parts if p and p.strip()]


def _is_generic(items: list[str]) -> bool:
    """True if any item is a vague word/phrase ("all files", "duplicates") rather than literal text."""
    if not items:
        return True
    for item in items:
        tokens = {t for t in re.split(r"[\s,;:/\\\-_.()\[\]{}]+", item.lower()) if t}
        if len(item) > 60 or len(item.split()) > 4 or not item.strip() or tokens & _GENERIC_WORDS:
            return True
    return False


def _symbols(text: str) -> list[str] | None:
    """Map words such as ``underscores and dots`` to ``["_", "."]``; ``None`` if any word is not a symbol."""
    words = [w for w in re.split(r"\s*[,&]\s*|\s+(?:and|এবং)\s+", text.strip().lower()) if w]
    mapped = [_SYMBOL_WORDS.get(w) for w in words]
    return [m for m in mapped if m is not None] if words and all(m is not None for m in mapped) else None


def _text_arg(raw: str) -> str:
    """Prefix/suffix payload: quoted text is kept verbatim, otherwise trimmed."""
    text, _quoted = _unquote(raw)
    return text


# ------------------------------------------------------------------ English
_EN_REMOVE = re.compile(
    rf"^(?:please\s+)?(?P<verb>{_VERB_REMOVE})\s+(?P<items>.+?)"
    rf"(?:\s+(?:from|in|out\s+of)\s+{_ALL}\b(?P<scope>.*))?$",
    _FLAGS,
)
_EN_REPLACE = re.compile(
    rf"^(?:please\s+)?(?:replace|substitute|swap)\s+(?P<a>.+?)\s+(?:with|by|to|->|→)\s+(?P<b>.+?)"
    rf"(?:\s+(?:in|from|on|for)\s+{_ALL}\b.*)?$",
    _FLAGS,
)
_EN_CHANGE_QUOTED = re.compile(
    r"""^(?:please\s+)?change\s+(?P<a>["“'‘`][^"”'’`]+["”'’`])\s+to\s+(?P<b>["“'‘`][^"”'’`]*["”'’`])(?:\s+.*)?$""", _FLAGS
)
_VERB_ADD = r"(?:add|put|insert|prepend|append|attach|place)"
_EN_PREFIX = (
    re.compile(
        rf"^(?:please\s+)?{_VERB_ADD}\s+(?:a\s+|the\s+)?prefix\s*(?::|=|of)?\s*(?P<t>.+?)"
        rf"(?:\s+(?:to|on|in|for)\s+{_ALL}\b.*)?$",
        _FLAGS,
    ),
    re.compile(
        rf"^(?:please\s+)?{_VERB_ADD}\s+(?P<t>.+?)\s+(?:at|to|in|on)\s+(?:the\s+)?(?:start|beginning|front)\s+of\s+.*$", _FLAGS
    ),
    re.compile(rf"^(?:please\s+)?prepend\s+(?P<t>.+?)(?:\s+to\s+{_ALL}\b.*)?$", _FLAGS),
)
_EN_SUFFIX = (
    re.compile(
        rf"^(?:please\s+)?{_VERB_ADD}\s+(?:a\s+|the\s+)?suffix\s*(?::|=|of)?\s*(?P<t>.+?)"
        rf"(?:\s+(?:to|on|in|for)\s+{_ALL}\b.*)?$",
        _FLAGS,
    ),
    re.compile(
        rf"^(?:please\s+)?{_VERB_ADD}\s+(?P<t>.+?)\s+(?:at|to|in|on)\s+(?:the\s+)?(?:end|back)\s+of\s+.*$", _FLAGS
    ),
    re.compile(rf"^(?:please\s+)?append\s+(?P<t>.+?)(?:\s+to\s+{_ALL}\b.*)?$", _FLAGS),
)
_EN_NUMBER = (
    re.compile(
        rf"^(?:please\s+)?(?:{_VERB_ADD}\s+)?(?:a\s+)?(?:sequential\s+|serial\s+|incremental\s+)?"
        rf"(?:numbering|numbers?)\b(?P<rest>.*)$",
        _FLAGS,
    ),
    re.compile(
        rf"^(?:please\s+)?number\s+{_ALL}?\s*(?:selected\s+)?(?:video\s*)?(?:files?|videos?|names?)\b(?P<rest>.*)$", _FLAGS
    ),
)
_NUMBER_START = re.compile(r"(?:from|starting(?:\s+(?:from|at|with))?|start(?:ing)?\s+(?:from|at|with)|beginning\s+(?:from|at|with))\s*#?(\d+)", _FLAGS)
_CASE_MODES = (
    ("title", re.compile(r"title\s*-?\s*case|capitali[sz]e(?:d)?|each\s+word", _FLAGS)),
    ("sentence", re.compile(r"sentence\s*-?\s*case", _FLAGS)),
    ("lower", re.compile(r"lower\s*-?\s*case|all\s+lower|small\s+letters?|ছোট\s*হাতের", _FLAGS)),
    ("upper", re.compile(r"upper\s*-?\s*case|all\s+caps|capital\s+letters?|all\s+upper|বড়\s*হাতের", _FLAGS)),
)
_EN_CASE_START = re.compile(
    r"^(?:please\s+)?(?:convert|change|make|set|turn|transform|rename|capitali[sz]e|lowercase|uppercase)\b", _FLAGS
)
_EN_SORT = re.compile(
    r"^(?:please\s+)?sort\s+(?:all\s+|the\s+)?(?:files|videos|names|them)?\s*(?:by\s+)?(?P<by>[a-z]+)\b(?P<rest>.*)$", _FLAGS
)
_DESC_WORDS = re.compile(r"desc|reverse|largest\s+first|biggest\s+first|newest\s+first|latest\s+first|z\s*-?\s*a|longest\s+first", _FLAGS)


def _parse_remove(text: str) -> LocalPlan | None:
    match = _EN_REMOVE.match(text)
    if not match:
        return None
    verb = match["verb"].lower()
    scope = match["scope"]  # None when there is no "from all ..." clause
    names_mentioned = scope is not None and bool(_SCOPE_WORDS.search(scope))
    if scope is not None and scope.strip() and not names_mentioned:
        return None
    if verb in {"delete", "cut"} and not names_mentioned:
        return None  # "delete X" may mean deleting *files* - let the AI decide
    items = _split_items(match["items"])
    if _is_generic(items):
        return None
    return _remove_plan(items)


def _remove_plan(items: list[str]) -> LocalPlan:
    return LocalPlan([RemoveTextAction(texts=tuple(items))], tr("offline.remove", items=", ".join(f"“{i}”" for i in items)))


def _parse_replace(text: str) -> LocalPlan | None:
    match = _EN_CHANGE_QUOTED.match(text) or _EN_REPLACE.match(text)
    if not match:
        return None
    a_raw, b_raw = match["a"].strip(), match["b"].strip()
    b_text, b_quoted = _unquote(b_raw)
    if not b_quoted and b_text.lower() in _EMPTY_WORDS:
        b_text = ""
    a_text, a_quoted = _unquote(a_raw)
    symbols = None if a_quoted else _symbols(a_text)
    b_symbols = None if b_quoted else _symbols(b_text)
    replacement = b_symbols[0] if b_symbols and len(b_symbols) == 1 else b_text
    if symbols:
        actions = [ReplaceTextAction(find=s, replace=replacement) for s in symbols if s != replacement]
        if not actions:
            return None
        return LocalPlan(actions, tr("offline.replace", find=a_text, replace=b_text or "∅"))
    if _is_generic([a_text]) or (b_text and _is_generic([b_text])) or a_text == b_text:
        return None
    return LocalPlan([ReplaceTextAction(find=a_text, replace=replacement)], tr("offline.replace", find=a_text, replace=b_text or "∅"))


def _affix_payload(match: re.Match[str]) -> str | None:
    payload = _text_arg(match["t"])
    if not payload or payload.lower() in _GENERIC_WORDS or len(payload) > 120:
        return None
    return payload


def _parse_prefix_suffix(text: str) -> LocalPlan | None:
    for pattern in _EN_PREFIX:
        match = pattern.match(text)
        if match and not re.match(r"(?i)^append\b", text):
            payload = _affix_payload(match)
            if payload:
                return LocalPlan([AddPrefixAction(text=payload)], tr("offline.prefix", text=payload))
    for pattern in _EN_SUFFIX:
        match = pattern.match(text)
        if match:
            payload = _affix_payload(match)
            if payload:
                return LocalPlan([AddSuffixAction(text=payload)], tr("offline.suffix", text=payload))
    return None


def _numbering_plan(rest: str) -> LocalPlan:
    start_match = _NUMBER_START.search(rest)
    digits = start_match.group(1) if start_match else "1"
    remainder = _NUMBER_START.sub("", rest).lower()
    position = "prefix" if re.search(r"\b(?:start|beginning|front|prefix)\b", remainder) else "suffix"
    width = max(2, len(digits))
    action = NumberingAction(start=int(digits), width=width, position=position, separator=" ")
    return LocalPlan([action], tr("offline.numbering", start=str(int(digits)).zfill(width), position=tr(f"offline.pos.{position}")))


def _parse_numbering(text: str) -> LocalPlan | None:
    for pattern in _EN_NUMBER:
        match = pattern.match(text)
        if match:
            rest = match["rest"]
            if re.search(r"\b(?:remove|delete|replace|rename|prefix|episode|season|title)\b", rest, _FLAGS):
                return None  # needs interpretation (templates etc.)
            return _numbering_plan(rest)
    return None


def _parse_case(text: str) -> LocalPlan | None:
    start = _EN_CASE_START.match(text)
    if not start or len(text.split()) > 12:
        return None
    if re.search(r"\b(?:prefix|suffix|remove|replace|number|numbering|folder|move|copy|delete)\b", text, _FLAGS):
        return None
    direct = start.group(0).lower() in {"lowercase", "uppercase", "capitalize", "capitalise"}
    if not (direct or _SCOPE_WORDS.search(text) or re.search(r"\b(?:them|everything)\b", text, _FLAGS)):
        return None
    for mode, pattern in _CASE_MODES:
        if pattern.search(text):
            return LocalPlan([ChangeCaseAction(mode=mode)], tr("offline.case", mode=tr(f"offline.case.{mode}")))
    return None


def _parse_sort(text: str) -> LocalPlan | None:
    match = _EN_SORT.match(text)
    if not match:
        return None
    key = _SORT_KEYS.get(match["by"].lower())
    if key is None:
        return None
    descending = bool(_DESC_WORDS.search(match["rest"]))
    return LocalPlan([SortAction(by=key, descending=descending)], tr("offline.sort", by=tr(f"sortkey.{key}"), order=tr("sort.desc" if descending else "sort.asc")))


# ------------------------------------------------------------------- Bangla
_BN_REMOVE_WORDS = (
    r"(?:বাদ\s*(?:দাও|দিন|দেবে|দেওয়া\s*হোক)|মুছে\s*(?:ফেল(?:ো|ুন)?|দাও|দিন)|সরিয়ে\s*(?:ফেল(?:ো|ুন)?|দাও|দিন)|"
    r"রিমুভ\s*(?:কর(?:ো|ুন)?)?|remove\s*(?:কর(?:ো|ুন)?)?|delete\s*(?:কর(?:ো|ুন)?)?)"
)
_BN_SCOPE = r"(?:সব\s*(?:ভিডিও(?:র)?|ফাইল(?:ের)?)?\s*(?:নাম(?:ের)?)?\s*(?:থেকে|হতে)?\s*|(?:সব\s*)?(?:ভিডিও(?:র)?|ফাইল(?:ের)?)\s*নাম(?:ের)?\s*(?:থেকে|হতে)\s*)?"
_BN_REMOVE = re.compile(rf"^{_BN_SCOPE}(?P<items>.+?)\s*{_BN_REMOVE_WORDS}$", _FLAGS)
_BN_REPLACE = re.compile(
    rf"^{_BN_SCOPE}(?P<a>.+?)\s*(?:-?কে|টাকে|টিকে)\s+(?P<b>.+?)\s*(?:দিয়ে|দ্বারা)\s*(?:replace|রিপ্লেস|পরিবর্তন|বদল(?:ে)?)\s*"
    rf"(?:দাও|দিন|কর(?:ো|ুন)?|করে\s*দাও)?$",
    _FLAGS,
)
_BN_WHERE_START = r"(?:শুরুতে|শুরুর\s*দিকে|সামনে|আগে)"
_BN_WHERE_END = r"(?:শেষে|পিছনে|পেছনে|শেষের\s*দিকে)"
_BN_ADD_WORDS = r"(?:যোগ\s*(?:কর(?:ো|ুন)?|দাও|দিন)|add\s*(?:কর(?:ো|ুন)?)?|বসাও|বসান|লেখো|রাখো|দাও|দিন)"
_BN_AFFIX_TEXT_FIRST = re.compile(
    rf"^(?:সব\s*(?:ভিডিও(?:র)?|ফাইল(?:ের)?)\s*)?(?:নাম(?:ের)?\s*)?(?P<t>.+?)\s+(?:(?:নামটা|নামটি|লেখাটা|লেখাটি|শব্দটা|শব্দটি|টেক্সটটা)\s*)?"
    rf"(?P<where>{_BN_WHERE_START}|{_BN_WHERE_END})\s+{_BN_ADD_WORDS}$",
    _FLAGS,
)
_BN_AFFIX_WHERE_FIRST = re.compile(
    rf"^(?:সব\s*(?:ভিডিও(?:র)?|ফাইল(?:ের)?)\s*)?(?:নাম(?:ের)?\s*)?(?P<where>{_BN_WHERE_START}|{_BN_WHERE_END})\s+(?P<t>.+?)\s+{_BN_ADD_WORDS}$",
    _FLAGS,
)
_BN_NUMBER = re.compile(
    r"^(?:.*?(?P<where>শুরুতে|শুরুর\s*দিকে|সামনে|শেষে|পিছনে|পেছনে|শেষের\s*দিকে))?.*?(?:(?P<n>\d+)\s*(?:থেকে|হতে)\s*)?"
    r"(?:numbering|নাম্বারিং|নম্বরিং|নম্বর(?:িং)?|সিরিয়াল|ক্রমিক\s*নম্বর)\s*(?:দাও|দিন|কর(?:ো|ুন)?|যোগ\s*কর(?:ো|ুন)?|করে\s*দাও)$",
    _FLAGS,
)
_BN_SORT = re.compile(
    r"^(?:সব\s*)?(?:ফাইল(?:গুলো)?|ভিডিও(?:গুলো)?)?\s*(?P<by>নাম|আকার|সাইজ|তারিখ|সময়|ধরন|সময়কাল)\s*(?:অনুযায়ী|অনুসারে|দিয়ে)\s*"
    r"(?:সাজাও|সাজান|সাজিয়ে\s*দাও|সর্ট\s*কর(?:ো|ুন)?|sort\s*কর(?:ো|ুন)?)(?P<rest>.*)$",
    _FLAGS,
)
_BN_CASE = re.compile(
    r"^(?:সব\s*)?(?:ভিডিও(?:র)?|ফাইল(?:ের)?)?\s*(?:নাম(?:গুলো)?)?\s*(?P<mode>ছোট\s*হাতের|বড়\s*হাতের)\s*(?:অক্ষরে)?\s*(?:কর(?:ো|ুন)?|করে\s*দাও|দাও)$",
    _FLAGS,
)


def _parse_bangla(text: str) -> LocalPlan | None:
    if not re.search(r"[ঀ-৿]", text):
        return None
    if match := _BN_NUMBER.match(text):
        where = match["where"]
        position = "prefix" if where and re.match(r"শুরু|সামনে", where) else "suffix"
        digits = match["n"] or "1"
        width = max(2, len(digits))
        action = NumberingAction(start=int(digits), width=width, position=position, separator=" ")
        return LocalPlan([action], tr("offline.numbering", start=str(int(digits)).zfill(width), position=tr(f"offline.pos.{position}")))
    if match := _BN_CASE.match(text):
        mode = "lower" if match["mode"].startswith("ছোট") else "upper"
        return LocalPlan([ChangeCaseAction(mode=mode)], tr("offline.case", mode=tr(f"offline.case.{mode}")))
    if match := _BN_SORT.match(text):
        key = _SORT_KEYS[match["by"]]
        descending = bool(re.search(r"উল্টো|বিপরীত|বড়\s*থেকে|desc", match["rest"], _FLAGS))
        return LocalPlan([SortAction(by=key, descending=descending)], tr("offline.sort", by=tr(f"sortkey.{key}"), order=tr("sort.desc" if descending else "sort.asc")))
    if match := _BN_REPLACE.match(text):
        a, _ = _unquote(match["a"])
        b, _ = _unquote(match["b"])
        if not _is_generic([a]) and a != b:
            return LocalPlan([ReplaceTextAction(find=a, replace=b)], tr("offline.replace", find=a, replace=b or "∅"))
    for pattern in (_BN_AFFIX_TEXT_FIRST, _BN_AFFIX_WHERE_FIRST):
        if match := pattern.match(text):
            payload = _text_arg(match["t"])
            if not payload or payload.lower() in _GENERIC_WORDS:
                continue
            if re.match(r"শুরু|সামনে|আগে", match["where"]):
                return LocalPlan([AddPrefixAction(text=payload)], tr("offline.prefix", text=payload))
            return LocalPlan([AddSuffixAction(text=payload)], tr("offline.suffix", text=payload))
    if match := _BN_REMOVE.match(text):
        items = _split_items(match["items"])
        if not _is_generic(items):
            return _remove_plan(items)
    return None


# -------------------------------------------------------------------- entry
def parse_local_command(command: str) -> LocalPlan | None:
    """Return a :class:`LocalPlan` if ``command`` is a simple deterministic request, else ``None``."""
    text = _clean(command)
    if not text or len(text) > 240 or "\n" in command.strip():
        return None
    if re.search(r"[।!?]|\.\s", text):
        return None  # several sentences: too complex to interpret without the AI
    bangla = _parse_bangla(text)
    if bangla is not None:
        return bangla
    for parser in (_parse_sort, _parse_case, _parse_numbering, _parse_replace, _parse_prefix_suffix, _parse_remove):
        plan = parser(text)
        if plan is not None:
            return plan
    return None
