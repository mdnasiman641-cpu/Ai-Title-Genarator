"""Google Search as a *context* source for the Title Generator.

Google Programmable Search (Custom Search JSON API) is queried with the API key
in the ``x-goog-api-key`` header (never in the URL). Only short snippets and
result metadata are kept - never whole webpages - and they are treated as
untrusted DATA: HTML is stripped, instruction-like sentences are dropped and the
text is length-limited before it is handed to the AI. Results are cached in the
local database for a limited time and identical queries in one batch are made once.
"""

from __future__ import annotations

import html
import re
import threading
import time
import urllib.parse
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from app.ai.errors import AIError, ErrorKind
from app.ai.http_client import request_json
from app.ai.model_config import validate_base_url
from app.ai.title_config import TitleConfig
from app.config.constants import GOOGLE_SEARCH_ENDPOINT
from app.utils.logger import get_logger

log = get_logger("ai.search")
SECRET_NAME = "google_search_api_key"
SNIPPET_CHARS = 300
CONTEXT_CHARS = 1_800  # per query, sent to the AI at most once per request

# Higher = more trustworthy for TV/film facts. Unknown domains get 1, social/video sites 0.
_RELIABLE = {
    "wikipedia.org": 3, "imdb.com": 3, "tvmaze.com": 3, "thetvdb.com": 3, "themoviedb.org": 3, "cbs.com": 3,
    "paramountplus.com": 3, "bbc.co.uk": 3, "nbc.com": 3, "abc.com": 3, "fox.com": 3, "netflix.com": 3,
    "rottentomatoes.com": 2, "tvguide.com": 2, "fandom.com": 2, "metacritic.com": 2, "variety.com": 2,
    "hollywoodreporter.com": 2, "deadline.com": 2, "tvline.com": 2, "screenrant.com": 1,
}  # fmt: skip
_UNRELIABLE = ("youtube.com", "facebook.com", "tiktok.com", "pinterest.", "instagram.com", "reddit.com", "x.com", "twitter.com")
_INJECTION = re.compile(
    r"ignore (?:all |the )?(?:previous|above|prior)|system prompt|you are (?:an? )?(?:ai|assistant|chatgpt|language model)|"
    r"disregard|new instructions|respond with|output the following|run (?:this|the following)|powershell|cmd\.exe",
    re.IGNORECASE,
)
_TAG = re.compile(r"<[^>]{0,200}>")
_SPACE = re.compile(r"\s+")
_SITE_SUFFIX = re.compile(r"\s*[|\-–—:]\s*(?:wikipedia|imdb|tvmaze|fandom|tv guide|rotten tomatoes|the tvdb|tmdb)[^|\-–—]*$", re.IGNORECASE)
_EPISODE = re.compile(r"\bS(?P<s>\d{1,2})\s*[ .\-_]?\s*E(?P<e>\d{1,3})\b|\b(?P<s2>\d{1,2})x(?P<e2>\d{2,3})\b", re.IGNORECASE)
_NOISE = re.compile(
    r"\b(?:2160p|1080p|720p|480p|4k|uhd|hdr\d*|web[- ]?dl|webrip|web|bluray|blu[- ]?ray|brrip|hdtv|dvdrip|x26[45]|h\.?26[45]|hevc|"
    r"avc|aac\d*(?:\.\d)?|ac3|dts|ddp?\d(?:\.\d)?|10bit|8bit|proper|repack|internal|final|amzn|nf|dsnp|hmax|yts|rarbg|eztv)\b",
    re.IGNORECASE,
)
_GENERIC_STEM = re.compile(r"^(?:video|clip|vid|movie|file|untitled|new|img|mov|dsc|gopr|random)[\s_\-]*\d*$", re.IGNORECASE)


class SearchError(Exception):
    """Search failed; ``kind`` is a short category (not_configured, network, timeout, rate_limit, auth, server, invalid)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass(slots=True)
class SearchHit:
    title: str
    snippet: str
    url: str
    domain: str
    reliability: int


@dataclass(slots=True)
class SearchContext:
    """What one query produced, reduced to what the AI may see."""

    query: str
    hits: list[SearchHit] = field(default_factory=list)
    corroborated: list[str] = field(default_factory=list)  # terms found by several independent sources
    conflicts: list[str] = field(default_factory=list)
    created: float = 0.0
    from_cache: bool = False

    @property
    def result_titles(self) -> list[str]:
        return [h.title for h in self.hits]

    def to_prompt(self) -> dict[str, Any]:
        """Compact, size-limited context for the AI prompt."""
        sources: list[dict[str, Any]] = []
        used = 0
        for hit in self.hits:
            entry = {"source": hit.domain, "reliability": {3: "high", 2: "medium"}.get(hit.reliability, "low"),
                     "result_title": hit.title, "snippet": hit.snippet}  # fmt: skip
            size = len(hit.title) + len(hit.snippet) + 40
            if used + size > CONTEXT_CHARS and sources:
                break
            sources.append(entry)
            used += size
        data: dict[str, Any] = {"sources": sources}
        if self.corroborated:
            data["confirmed_by_several_sources"] = self.corroborated
        if self.conflicts:
            data["sources_disagree_about"] = self.conflicts
        return data

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "hits": [asdict(h) for h in self.hits], "corroborated": self.corroborated,
                "conflicts": self.conflicts, "created": self.created}  # fmt: skip

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchContext:
        hits = [SearchHit(**{k: h.get(k, "" if k != "reliability" else 1) for k in SearchHit.__dataclass_fields__}) for h in data.get("hits") or []]
        return cls(str(data.get("query", "")), hits, list(data.get("corroborated") or []), list(data.get("conflicts") or []),
                   float(data.get("created") or 0.0), True)  # fmt: skip


# =================================================================== cleaning
def clean_text(text: str, limit: int = SNIPPET_CHARS) -> str:
    """Untrusted web text -> plain, short, instruction-free text."""
    text = html.unescape(_TAG.sub(" ", str(text or "")))
    text = "".join(ch for ch in text if ch.isprintable() or ch == " ")
    sentences = re.split(r"(?<=[.!?])\s+", _SPACE.sub(" ", text).strip())
    kept = [s for s in sentences if s and not _INJECTION.search(s)]
    out = " ".join(kept).strip()
    return out[: limit - 1].rstrip() + "…" if len(out) > limit else out


def domain_of(url: str) -> str:
    host = urllib.parse.urlparse(url).hostname or ""
    return host.removeprefix("www.").lower()


def reliability(domain: str) -> int:
    if any(bad in domain for bad in _UNRELIABLE):
        return 0
    for known, score in _RELIABLE.items():
        if domain == known or domain.endswith("." + known):
            return score
    return 1


def strip_site_name(title: str) -> str:
    previous = None
    while previous != title:
        previous, title = title, _SITE_SUFFIX.sub("", title).strip()
    return title


_COMMON_CAPS = frozenset(
    "the this that these those when after before while with from into about their there where what which season episode "
    "series watch online full free official review recap plot summary cast trailer".split()
)
_CODE_PART = re.compile(r"(?i)\bS\d{1,2}\s*E\d{1,3}\b|\b\d{1,2}x\d{2,3}\b|\bseason\s+\d+|\bepisode\s+\d+")


def _proper_terms(text: str) -> set[str]:
    """Capitalised words (names, places) - candidates for facts several sources agree on."""
    return {w for w in re.findall(r"\b[A-Z][a-z]{3,}\b", text) if w.lower() not in _COMMON_CAPS}


def episode_name(result_title: str, query: str = "") -> str:
    """``Partners - Blue Bloods S05E03 - Wikipedia`` -> ``Partners`` ("" if none can be found)."""
    known = query.casefold()
    parts = [p.strip().strip('"“”') for p in re.split(r"\s+[\-–—|]\s+|:\s+", strip_site_name(result_title)) if p.strip()]
    names = [p for p in parts if not _CODE_PART.search(p) and p.casefold() not in known]
    return names[-1] if names else ""


def build_context(query: str, hits: list[SearchHit], *, compare: bool) -> SearchContext:
    """Rank by reliability; with ``compare`` also find agreeing terms and disagreements."""
    ranked = sorted(hits, key=lambda h: -h.reliability)
    context = SearchContext(query, ranked, created=time.time())
    if compare and len(ranked) > 1:
        seen_by: dict[str, set[str]] = {}
        for hit in ranked:
            for term in _proper_terms(f"{hit.title}. {hit.snippet}"):
                seen_by.setdefault(term, set()).add(hit.domain)
        context.corroborated = sorted(t for t, domains in seen_by.items() if len(domains) >= 2)[:12]
        names: Counter[str] = Counter()
        for hit in ranked:
            if hit.reliability >= 2 and (name := episode_name(hit.title, query)):
                names[name.casefold()] += 1
        if len(names) > 1:
            context.conflicts = [f"episode name ({' / '.join(sorted(names))})"]
    return context


# =================================================================== queries
@dataclass(frozen=True, slots=True)
class FilenameInfo:
    cleaned: str  # stem without quality/release tags
    series: str
    season: int | None
    episode: int | None

    @property
    def meaningful(self) -> bool:
        """``video001`` or ``DSC_0042`` say nothing; ``Blue Bloods S05E03`` does."""
        return bool(self.cleaned) and not _GENERIC_STEM.match(self.cleaned) and len(re.findall(r"[^\W\d_]{2,}", self.cleaned)) >= 1


def parse_filename(stem: str) -> FilenameInfo:
    text = re.sub(r"[\[\(][^\]\)]*[\]\)]", " ", stem)
    text = _NOISE.sub(" ", re.sub(r"[._]+", " ", text))
    text = _SPACE.sub(" ", text).strip(" -")
    season = episode = None
    series = ""
    if m := _EPISODE.search(text):
        season = int(m.group("s") or m.group("s2"))
        episode = int(m.group("e") or m.group("e2"))
        series = text[: m.start()].strip(" -")
    return FilenameInfo(text, series, season, episode)


def build_query(config: TitleConfig, stem: str) -> str:
    """The search query for one file ("" = nothing useful to search for)."""
    info = parse_filename(stem) if config.use_filename_lookup else FilenameInfo("", "", None, None)
    series = config.prefix.strip() or info.series
    episode_code = f"S{info.season:02d}E{info.episode:02d}" if info.season is not None and info.episode is not None else ""
    general = " ".join(filter(None, [series, config.topic.strip()[:60] or config.category_text, "episode plot" if series else ""]))
    mode = config.query_mode
    if mode == "custom" and config.custom_query.strip():
        values = {"series": series, "season": f"{info.season:02d}" if info.season is not None else "",
                  "episode": f"{info.episode:02d}" if info.episode is not None else "", "filename": info.cleaned,
                  "topic": config.topic.strip(), "category": config.category_text, "keywords": ", ".join(config.keyword_list)}  # fmt: skip
        query = re.sub(r"\{(\w+)\}", lambda m: values.get(m.group(1), ""), config.custom_query)
    elif mode == "filename" and info.meaningful:
        query = info.cleaned
    elif mode == "filename_series" and info.meaningful:
        query = info.cleaned if series and series.lower() in info.cleaned.lower() else f"{series} {info.cleaned}"
    elif info.meaningful and episode_code:
        query = f"{series} {episode_code} episode"
    elif info.meaningful:
        query = f"{info.cleaned} episode" if not series or series.lower() in info.cleaned.lower() else f"{series} {info.cleaned}"
    else:
        query = general
    return _SPACE.sub(" ", query).strip()[:128]


# ==================================================================== client
class GoogleSearchClient:
    """Google Programmable Search (Custom Search JSON API)."""

    def __init__(self, api_key: str, engine_id: str, endpoint: str = GOOGLE_SEARCH_ENDPOINT, timeout: float = 15) -> None:
        self.api_key = api_key
        self.engine_id = engine_id
        self.endpoint = endpoint.strip().rstrip("/") or GOOGLE_SEARCH_ENDPOINT
        self.timeout = timeout

    def search(self, query: str, num: int) -> list[SearchHit]:
        problem = validate_base_url(self.endpoint)
        if problem:
            raise SearchError("not_configured", problem)
        params = urllib.parse.urlencode({"cx": self.engine_id, "q": query, "num": max(1, min(10, num)), "safe": "active"})
        try:
            data = request_json("Google Search", "GET", f"{self.endpoint}?{params}", headers={"x-goog-api-key": self.api_key},
                                timeout=self.timeout)  # fmt: skip
        except AIError as exc:
            kinds = {ErrorKind.AUTH: "auth", ErrorKind.RATE_LIMIT: "rate_limit", ErrorKind.TIMEOUT: "timeout",
                     ErrorKind.NETWORK: "network", ErrorKind.SERVER: "server", ErrorKind.INVALID_RESPONSE: "invalid"}  # fmt: skip
            raise SearchError(kinds.get(exc.kind, "error"), exc.category) from None
        items = data.get("items", [])
        if not isinstance(items, list):
            raise SearchError("invalid", "unexpected response format")
        hits: list[SearchHit] = []
        for item in items[:num]:
            if not isinstance(item, dict):
                continue
            url = str(item.get("link") or "")
            domain = domain_of(url) or str(item.get("displayLink") or "").lower()
            title = clean_text(item.get("title") or "", 160)
            snippet = clean_text(item.get("snippet") or "")
            if title or snippet:
                hits.append(SearchHit(title, snippet, url[:300], domain, reliability(domain)))
        return hits


class SearchService:
    """Client + cache + per-batch de-duplication. Thread-safe enough for one worker at a time."""

    def __init__(self, client: GoogleSearchClient | None, db: Any | None = None, *, cache_hours: float = 24.0) -> None:
        self.client = client
        self.db = db
        self.cache_hours = cache_hours
        self._lock = threading.Lock()
        self.requests = 0  # real HTTP searches made (for tests / status)

    @property
    def configured(self) -> bool:
        return self.client is not None

    def _key(self, query: str, num: int, compare: bool) -> str:
        return f"{query.casefold()}|{num}|{int(compare)}"

    def search(self, query: str, num: int, *, compare: bool = True) -> SearchContext:
        if self.client is None:
            raise SearchError("not_configured", "Google Search is not set up (Settings → AI → Google Search).")
        key = self._key(query, num, compare)
        if self.db is not None and self.cache_hours > 0:
            cached = self.db.get_search_cache(key, time.time() - self.cache_hours * 3600)
            if cached:
                return SearchContext.from_dict(cached)
        with self._lock:
            self.requests += 1
        hits = self.client.search(query, num)
        context = build_context(query, hits, compare=compare)
        if self.db is not None and self.cache_hours > 0:
            try:
                self.db.put_search_cache(key, context.created, context.to_dict())
                self.db.purge_search_cache(time.time() - self.cache_hours * 3600)
            except Exception:  # noqa: BLE001 - a cache problem must never break generation
                log.debug("search cache write failed", exc_info=True)
        log.info("Google Search: %d result(s) for a %d-char query", len(hits), len(query))
        return context

    def clear_cache(self) -> int:
        return self.db.purge_search_cache(None) if self.db is not None else 0


def make_search_service(ctx: Any) -> SearchService:
    """Build the service from saved settings + the encrypted key (client is None when not configured)."""
    settings = ctx.settings.load()
    key = ctx.settings.secrets.get(SECRET_NAME)
    client = None
    if key and settings.search_engine_id.strip():
        client = GoogleSearchClient(key, settings.search_engine_id.strip(), settings.search_endpoint or GOOGLE_SEARCH_ENDPOINT)
    return SearchService(client, ctx.db, cache_hours=settings.search_cache_hours)
