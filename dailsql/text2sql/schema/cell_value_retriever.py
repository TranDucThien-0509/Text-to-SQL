"""
CellValueRetriever – matches question tokens to actual database cell values.

Uses SQLite to query distinct column values, with three match strategies:
  exact   – token is a verbatim cell value
  partial – token is a substring of a cell value
  fuzzy   – edit-distance approximation (SequenceMatcher)

Results are cached per (db_id, table, column) tuple.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import string
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_PUNCT = str.maketrans("", "", string.punctuation)


def _normalise(text: str) -> str:
    return text.lower().translate(_PUNCT).strip()


class CellValueRetriever:
    """
    Connects to SQLite databases and retrieves distinct cell values
    for schema-linking purposes.

    Args:
        db_dir:         Directory containing one sub-folder per db_id,
                        each with a ``<db_id>.sqlite`` file.
        max_cell_values: Max distinct values to fetch per column.
        fuzzy_threshold: Min SequenceMatcher ratio to accept a fuzzy hit.
        timeout_seconds: SQLite query timeout.
    """

    def __init__(
        self,
        db_dir: Path,
        max_cell_values: int = 50,
        fuzzy_threshold: float = 0.8,
        timeout_seconds: float = 3.0,
    ) -> None:
        self._db_dir = db_dir
        self._max_cell_values = max_cell_values
        self._fuzzy_threshold = fuzzy_threshold
        self._timeout = timeout_seconds

        # Cache: (db_id, table, column) → List[str]
        self._cache: Dict[Tuple[str, str, str], List[str]] = {}
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
        """
        cell_values = self._get_values(db_id, table, column)
        if not cell_values:
            return []

        q_tokens = self._tokenize(question)
        hits: List[Tuple[str, float]] = []

        for val in cell_values:
            val_norm = _normalise(val)
            if not val_norm:
                continue

            # Exact match
            if val_norm in q_tokens or val_norm in _normalise(question):
                hits.append((val, 1.0))
                continue

            # Partial (substring)
            if any(val_norm in t or t in val_norm for t in q_tokens if len(t) > 2):
                hits.append((val, 0.8))
                continue

            # Fuzzy
            for token in q_tokens:
                ratio = SequenceMatcher(None, token, val_norm).ratio()
                if ratio >= self._fuzzy_threshold:
                    hits.append((val, round(ratio, 3)))
                    break

        hits.sort(key=lambda x: -x[1])
        return hits[:top_k]

    def prefetch_db(self, db_id: str, table_names: List[str], col_infos: List) -> None:
        """
        Pre-warm the cache for all columns in a database.
        Call once before schema linking if performance matters.
        """
        for col in col_infos:
            tbl = table_names[col.table_idx]
            self._get_values(db_id, tbl, col.name)

    # ── Internals ─────────────────────────────────────────────

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