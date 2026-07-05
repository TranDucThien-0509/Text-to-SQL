"""
CellValueRetriever – matches question tokens to actual database cell values.

Uses SQLite to query distinct column values, with three match strategies:
  exact   – token is a verbatim cell value
  partial – token is a substring of a cell value
  fuzzy   – edit-distance approximation (SequenceMatcher)

For cross-lingual matching (VI question ↔ EN cell values), candidate value
spans are extracted from the raw question and translated via an injected
translator — NOT the whole question, to avoid garbling.

Candidate extraction strategy:
  1. Quoted substrings 'x' / "x"       – highest precision, rare in practice
  2. Content-token n-grams (1..N)      – main path. Built from ViTokenizer
     output with Vietnamese stopwords removed, NOT from a capitalization
     heuristic. Capitalization-based extraction (e.g. `[A-Z][a-z]*` style
     regexes) does not work for Vietnamese: machine-translated questions
     rarely preserve reliable capitalization on proper nouns, and Latin-1
     "À-Ỹ" style Unicode ranges do not cleanly separate upper/lowercase
     Vietnamese diacritic letters, so such regexes match mid-word garbage
     rather than real spans (verified: they shatter ordinary Vietnamese
     sentences into fragments like 'ững', 'áo', 'ên').

Question-level extraction + translation is cached (keyed by NFC-normalised
question text) so repeated calls to match() across many columns for the
same question do not re-translate the same candidates over and over.

Results are cached per (db_id, table, column) tuple.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import string
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Protocol, Tuple

try:
    from pyvi import ViTokenizer
except ImportError:  # pragma: no cover
    ViTokenizer = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_PUNCT = str.maketrans("", "", string.punctuation)

# Quoted substrings: 'x' or "x" — kept as a high-precision fast path
_QUOTED_RE = re.compile(r"['\"]([^'\"]+)['\"]")

# Same stopword set used by schema_linker.py's question tokenizer, so
# n-gram candidates line up with how the question is segmented elsewhere.
_VI_STOPWORDS = {
    "là", "của", "các", "những", "một", "có", "cho", "và", "hoặc", "nhưng",
    "không", "nào", "gì", "ai", "hãy", "liệt_kê", "hiển_thị", "tìm", "đưa",
    "trả_về", "lấy", "cho_biết", "tôi", "này", "kia", "rằng", "thì", "mà",
    "để", "với", "tại", "từ", "đến", "trong", "ngoài", "dưới", "trên",
    "khi", "lúc", "đang", "đã", "sẽ", "nhé", "nha", "ạ", "liệt", "kê",
}


def _normalise(text: str) -> str:
    return text.lower().translate(_PUNCT).strip()


def _normalise_vi(text: str) -> str:
    """NFC-normalise Vietnamese text before any regex/tokenizer processing.
    Text arriving in NFD (decomposed diacritics) breaks both regex character
    classes and ViTokenizer's compound-word matching silently."""
    return unicodedata.normalize("NFC", text)


class Translator(Protocol):
    """Minimal interface CellValueRetriever needs — implement with mt5 or anything else."""
    def translate(self, text: str) -> str: ...


class CellValueRetriever:
    """
    Connects to SQLite databases and retrieves distinct cell values
    for schema-linking purposes.

    Args:
        db_dir:          Directory containing one sub-folder per db_id,
                         each with a ``<db_id>.sqlite`` file.
        max_cell_values:  Max distinct values to fetch per column.
        fuzzy_threshold:  Min SequenceMatcher ratio to accept a fuzzy hit.
        timeout_seconds:  SQLite query timeout.
        translator:       Optional translator used ONLY on extracted value
                          candidates (not the full question).
    """

    def __init__(
        self,
        db_dir: Path,
        max_cell_values: int = 50,
        fuzzy_threshold: float = 0.8,
        timeout_seconds: float = 3.0,
        translator: Optional[Translator] = None,
        max_value_ngram: int = 4,
    ) -> None:
        self._db_dir = db_dir
        self._max_cell_values = max_cell_values
        self._fuzzy_threshold = fuzzy_threshold
        self._timeout = timeout_seconds
        self._translator = translator
        self._max_value_ngram = max_value_ngram

        # Cache: (db_id, table, column) → List[str]
        self._cache: Dict[Tuple[str, str, str], List[str]] = {}
        # Cache: normalised question → List[(candidate_vi, candidate_en)]
        # Avoids re-extracting/re-translating candidates once per column.
        self._question_cache: Dict[str, List[Tuple[str, str]]] = {}
        self._lock = Lock()

    # ── Public API ────────────────────────────────────────────

    def match(
        self,
        db_id: str,
        table: str,
        column: str,
        question: str,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Return up to *top_k* (cell_value, score) pairs where score > 0.
        Score = 1.0 for exact, 0.8 for partial, fuzzy score otherwise.

        Matching runs twice:
          1. against the raw (Vietnamese) question tokens — in case cell
             values happen to already match (numbers, codes, EN loanwords)
          2. against translated *value candidates* extracted from the
             question (capitalized phrases / quoted substrings) — for
             cases like "Toán" → "Math"
        Results from both passes are merged, deduped by value (keep max score).
        """
        cell_values = self._get_values(db_id, table, column)
        if not cell_values:
            return []

        merged: Dict[str, float] = {}

        # Pass 1: raw question tokens
        for val, score in self._score_against_tokens(self._tokenize(question), question, cell_values):
            merged[val] = max(merged.get(val, 0.0), score)

        # Pass 2: translated value candidates (extracted + translated once
        # per question, cached, and reused across every column of this db).
        if self._translator is not None:
            for _candidate_vi, translated in self._get_translated_candidates(question):
                if not translated:
                    continue
                for val, score in self._score_against_tokens(
                    self._tokenize(translated), translated, cell_values
                ):
                    merged[val] = max(merged.get(val, 0.0), score)

        hits = sorted(merged.items(), key=lambda x: -x[1])
        return hits[:top_k]

    # ── Internals: per-question candidate cache ───────────────

    def _get_translated_candidates(self, question: str) -> List[Tuple[str, str]]:
        """
        Returns [(candidate_vi, candidate_en), ...] for *question*, computing
        and translating candidates only on first call for a given question.
        """
        key = _normalise_vi(question)
        with self._lock:
            cached = self._question_cache.get(key)
        if cached is not None:
            return cached

        pairs: List[Tuple[str, str]] = []
        for candidate in self._extract_value_candidates(question):
            try:
                translated = self._translator.translate(candidate)  # type: ignore[union-attr]
            except Exception as exc:
                logger.debug("[CellValueRetriever] translate failed for '%s': %s", candidate, exc)
                continue
            if translated:
                pairs.append((candidate, translated))

        with self._lock:
            self._question_cache[key] = pairs
        return pairs

    def prefetch_db(self, db_id: str, table_names: List[str], col_infos: List) -> None:
        """
        Pre-warm the cache for all columns in a database.
        Call once before schema linking if performance matters.
        """
        for col in col_infos:
            tbl = table_names[col.table_idx]
            self._get_values(db_id, tbl, col.name)

    # ── Internals: value-candidate extraction ────────────────

    def _extract_value_candidates(self, question: str) -> List[str]:
        """
        Pull out substrings from the question that are likely to be literal
        values, e.g. "Toán" in "những giáo viên nào dạy môn Toán".

        Strategy:
          1. Quoted substrings (highest confidence, rare in this dataset)
          2. Content-token n-grams, longest first (main path)

        NOTE: capitalization is NOT used as a signal. Vietnamese questions
        (especially machine-translated ones) do not reliably capitalize
        proper nouns/values, and naive `[A-Z][a-z]*`-style Unicode ranges
        do not cleanly separate Vietnamese upper/lowercase diacritic
        letters — they match mid-word garbage instead of real spans.
        """
        question = _normalise_vi(question)
        candidates: List[str] = []

        candidates.extend(_QUOTED_RE.findall(question))

        tokens = self._content_tokens(question)
        max_n = min(self._max_value_ngram, len(tokens))
        for n in range(max_n, 0, -1):
            for i in range(len(tokens) - n + 1):
                span = " ".join(t.replace("_", " ") for t in tokens[i:i + n])
                candidates.append(span)

        # Dedup, preserve order (longest spans first), drop very short junk
        seen = set()
        out = []
        for c in candidates:
            c = c.strip()
            key = c.lower()
            if len(c) < 2 or key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def _content_tokens(self, question: str) -> List[str]:
        """
        Segment *question* with ViTokenizer (compound words joined by '_'),
        lowercased for stopword lookup but returned with original casing
        stripped of punctuation — and drop Vietnamese stopwords so n-grams
        built on top are content-bearing spans rather than function words.
        """
        punct_no_underscore = str.maketrans("", "", string.punctuation.replace("_", ""))
        cleaned = question.translate(punct_no_underscore)

        if ViTokenizer is not None:
            segmented = ViTokenizer.tokenize(cleaned)
            raw_tokens = segmented.split()
        else:  # pragma: no cover - fallback if pyvi unavailable at runtime
            raw_tokens = cleaned.split()

        return [t for t in raw_tokens if t.lower() not in _VI_STOPWORDS and len(t) > 1]

    # ── Internals: matching ───────────────────────────────────

    def _score_against_tokens(
        self, tokens: List[str], full_text: str, cell_values: List[str]
    ) -> List[Tuple[str, float]]:
        hits: List[Tuple[str, float]] = []
        full_norm = _normalise(full_text)

        for val in cell_values:
            val_norm = _normalise(val)
            if not val_norm:
                continue

            # Exact match
            if val_norm in tokens or val_norm in full_norm:
                hits.append((val, 1.0))
                continue

            # Partial (substring)
            if any(val_norm in t or t in val_norm for t in tokens if len(t) > 2):
                hits.append((val, 0.8))
                continue

            # Fuzzy
            for token in tokens:
                ratio = SequenceMatcher(None, token, val_norm).ratio()
                if ratio >= self._fuzzy_threshold:
                    hits.append((val, round(ratio, 3)))
                    break

        return hits

    def _get_values(self, db_id: str, table: str, column: str) -> List[str]:
        key = (db_id, table, column)
        with self._lock:
            if key in self._cache:
                return self._cache[key]

        values = self._query_sqlite(db_id, table, column)
        with self._lock:
            self._cache[key] = values
        return values

    def _query_sqlite(self, db_id: str, table: str, column: str) -> List[str]:
        db_path = self._resolve_db_path(db_id)
        if db_path is None:
            return []

        try:
            con = sqlite3.connect(str(db_path), timeout=self._timeout)
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute(
                f'SELECT DISTINCT "{column}" FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL LIMIT ?',
                (self._max_cell_values,),
            )
            rows = cur.fetchall()
            con.close()
            return [str(r[0]) for r in rows if r[0] is not None]
        except Exception as exc:
            logger.debug(
                "[CellValueRetriever] Failed %s.%s.%s: %s", db_id, table, column, exc
            )
            return []

    def _resolve_db_path(self, db_id: str) -> Optional[Path]:
        # Convention: db_dir/<db_id>/<db_id>.sqlite
        candidate = self._db_dir / db_id / f"{db_id}.sqlite"
        if candidate.exists():
            return candidate
        # Flat layout: db_dir/<db_id>.sqlite
        flat = self._db_dir / f"{db_id}.sqlite"
        if flat.exists():
            return flat
        logger.debug("[CellValueRetriever] DB file not found for db_id='%s'", db_id)
        return None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in _normalise(text).split() if len(t) > 1]