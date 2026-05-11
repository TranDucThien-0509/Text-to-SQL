"""
SchemaLinker – identifies which tables and columns are relevant to a question.

Implements four match types from the DAIL-SQL / IRNet lineage:
  q_col_match    – question tokens ↔ column names
  q_tab_match    – question tokens ↔ table names
  cell_match     – question tokens ↔ actual DB cell values  (via CellValueRetriever)
  num_date_match – numeric / date literals in the question
"""
from __future__ import annotations

import logging
import re
import string
from pyvi import ViTokenizer
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from text2sql.schema.schema_processor import DatabaseSchema

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Stop words
# ─────────────────────────────────────────────────────────────
_STOP_WORDS: Set[str] = {
    # Tiếng Anh
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "and", "or", "but", "not", "no", "what", "which", "who", "how",
    "many", "much", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "list", "show", "find", "give",
    "return", "get", "tell", "me", "their", "its", "this", "that",
    
    # Tiếng Việt
    "là", "của", "các", "những", "một", "có", "cho", "và", "hoặc", "nhưng",
    "không", "nào", "gì", "ai", "bao_nhiêu", "thế_nào", "hãy", "liệt_kê",
    "hiển_thị", "tìm", "đưa", "trả_về", "lấy", "cho_biết", "tôi", "này", "kia",
    "rằng", "thì", "mà", "để", "với", "tại", "từ", "đến", "trong", "ngoài",
    "dưới", "trên", "khi", "lúc", "đang", "đã", "sẽ", "nhé", "nha", "ạ"
}

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Number / date patterns
_NUM_PAT = re.compile(r"\b\d{1,4}([.,]\d+)?\b")
_DATE_PAT = re.compile(
    r"\b\d{4}[-/]\d{2}[-/]\d{2}\b"          # 2023-01-15
    r"|\b\d{2}/\d{2}/\d{4}\b"               # 01/15/2023
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────

@dataclass
class ColumnMatch:
    col_global_idx: int
    col_name: str
    table_name: str
    match_type: str     # "exact" | "partial" | "ngram" | "cell" | "num_date"
    score: float = 1.0
    matched_span: Optional[str] = None


@dataclass
class TableMatch:
    table_idx: int
    table_name: str
    match_type: str
    score: float = 1.0
    matched_span: Optional[str] = None


@dataclass
class SchemaLinkingResult:
    db_id: str
    question: str
    q_col_match: List[ColumnMatch] = field(default_factory=list)
    q_tab_match: List[TableMatch] = field(default_factory=list)
    cell_match: List[ColumnMatch] = field(default_factory=list)
    num_date_match: List[ColumnMatch] = field(default_factory=list)

    @property
    def relevant_table_indices(self) -> Set[int]:
        indices: Set[int] = set()
        for m in self.q_tab_match:
            indices.add(m.table_idx)
        for m in (*self.q_col_match, *self.cell_match, *self.num_date_match):
            # retrieve table via schema col info – stored on ColumnMatch
            pass  # filled by linker
        return indices

    @property
    def relevant_col_indices(self) -> Set[int]:
        return {
            m.col_global_idx
            for m in (*self.q_col_match, *self.cell_match, *self.num_date_match)
        }

    def to_dict(self) -> dict:
        def _col(m: ColumnMatch) -> dict:
            return {
                "col_idx": m.col_global_idx,
                "col": m.col_name,
                "table": m.table_name,
                "type": m.match_type,
                "score": round(m.score, 4),
                "span": m.matched_span,
            }

        def _tab(m: TableMatch) -> dict:
            return {
                "table_idx": m.table_idx,
                "table": m.table_name,
                "type": m.match_type,
                "score": round(m.score, 4),
                "span": m.matched_span,
            }

        return {
            "db_id": self.db_id,
            "question": self.question,
            "q_col_match": [_col(m) for m in self.q_col_match],
            "q_tab_match": [_tab(m) for m in self.q_tab_match],
            "cell_match": [_col(m) for m in self.cell_match],
            "num_date_match": [_col(m) for m in self.num_date_match],
        }


# ─────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────

class SchemaLinker:
    """
    Runs all four linking steps and returns a SchemaLinkingResult.

    Usage::

        linker = SchemaLinker()
        result = linker.link(question, schema, cell_retriever)
    """

    def __init__(self, ngram_sizes: Tuple[int, ...] = (1, 2, 3)) -> None:
        self._ngram_sizes = ngram_sizes

    # ── Public entry point ────────────────────────────────────

    def link(
        self,
        question: str,
        schema: DatabaseSchema,
        cell_retriever: Optional["CellValueRetriever"] = None,   # type: ignore[name-defined]
    ) -> SchemaLinkingResult:
        """
        Run all linking steps and return a consolidated SchemaLinkingResult.
        Priority: exact > partial > ngram (deduplication by col/table idx).
        """
        result = SchemaLinkingResult(db_id=schema.db_id, question=question)

        tokens = self._tokenize(question)
        ngrams = self._build_ngrams(tokens)

        # Step 1 – column matching
        result.q_col_match = self._match_columns(schema, tokens, ngrams)

        # Step 2 – table matching
        result.q_tab_match = self._match_tables(schema, tokens, ngrams)

        # Step 3 – cell value matching (optional)
        if cell_retriever is not None:
            result.cell_match = self._match_cells(schema, question, cell_retriever)

        # Step 4 – num / date matching
        result.num_date_match = self._match_num_date(schema, question)

        logger.debug(
            "[SchemaLinker] %s: %d col | %d tab | %d cell | %d num/date matches",
            schema.db_id,
            len(result.q_col_match),
            len(result.q_tab_match),
            len(result.cell_match),
            len(result.num_date_match),
        )
        return result

    # ── Step helpers ──────────────────────────────────────────

    def _match_columns(
        self,
        schema: DatabaseSchema,
        tokens: List[str],
        ngrams: List[str],
    ) -> List[ColumnMatch]:
        seen: Set[int] = set()
        matches: List[ColumnMatch] = []

        for col in schema.columns:
            if col.global_idx in seen:
                continue
            col_tokens = self._tokenize(col.name)
            tbl_name = schema.table_names[col.table_idx]

            match = self._best_token_match(tokens, ngrams, col_tokens)
            if match:
                seen.add(col.global_idx)
                matches.append(
                    ColumnMatch(
                        col_global_idx=col.global_idx,
                        col_name=col.name,
                        table_name=tbl_name,
                        match_type=match[0],
                        score=match[1],
                        matched_span=match[2],
                    )
                )

        return sorted(matches, key=lambda m: -m.score)

    def _match_tables(
        self,
        schema: DatabaseSchema,
        tokens: List[str],
        ngrams: List[str],
    ) -> List[TableMatch]:
        seen: Set[int] = set()
        matches: List[TableMatch] = []

        for t_idx, t_name in enumerate(schema.table_names):
            if t_idx in seen:
                continue
            tbl_tokens = self._tokenize(t_name)
            match = self._best_token_match(tokens, ngrams, tbl_tokens)
            if match:
                seen.add(t_idx)
                matches.append(
                    TableMatch(
                        table_idx=t_idx,
                        table_name=t_name,
                        match_type=match[0],
                        score=match[1],
                        matched_span=match[2],
                    )
                )

        return sorted(matches, key=lambda m: -m.score)

    def _match_cells(
        self,
        schema: DatabaseSchema,
        question: str,
        cell_retriever: "CellValueRetriever",   # type: ignore[name-defined]
    ) -> List[ColumnMatch]:
        matches: List[ColumnMatch] = []
        seen_cols: Set[int] = set()

        for col in schema.columns:
            if col.global_idx in seen_cols:
                continue
            tbl_name = schema.table_names[col.table_idx]
            hits = cell_retriever.match(
                db_id=schema.db_id,
                table=tbl_name,
                column=col.name,
                question=question,
            )
            for cell_val, score in hits:
                if score > 0:
                    seen_cols.add(col.global_idx)
                    matches.append(
                        ColumnMatch(
                            col_global_idx=col.global_idx,
                            col_name=col.name,
                            table_name=tbl_name,
                            match_type="cell",
                            score=score,
                            matched_span=cell_val,
                        )
                    )
                    break   # one hit per column is enough

        return matches

    def _match_num_date(
        self, schema: DatabaseSchema, question: str
    ) -> List[ColumnMatch]:
        """
        Tag columns whose type suggests numeric / date semantics when
        the question contains a matching literal.
        """
        num_hits = _NUM_PAT.findall(question)
        date_hits = _DATE_PAT.findall(question)
        if not num_hits and not date_hits:
            return []

        matches: List[ColumnMatch] = []
        seen: Set[int] = set()

        numeric_types = {"NUMBER", "INT", "INTEGER", "FLOAT", "REAL", "DOUBLE", "DECIMAL"}
        date_types = {"DATE", "DATETIME", "TIME", "TIMESTAMP", "YEAR"}

        for col in schema.columns:
            if col.global_idx in seen:
                continue
            ctype = col.col_type.upper()
            relevant = (num_hits and ctype in numeric_types) or (
                date_hits and any(dt in ctype for dt in date_types)
            )
            if relevant:
                seen.add(col.global_idx)
                matches.append(
                    ColumnMatch(
                        col_global_idx=col.global_idx,
                        col_name=col.name,
                        table_name=schema.table_names[col.table_idx],
                        match_type="num_date",
                        score=0.8,
                    )
                )
        return matches

    # ── Token utilities ───────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
            text = text.translate(_PUNCT_TABLE).lower()
            tokenized_text = ViTokenizer.tokenize(text)
            tokens = tokenized_text.split()
        
            return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]

    def _build_ngrams(self, tokens: List[str]) -> List[str]:
        result = []
        for n in self._ngram_sizes:
            for i in range(len(tokens) - n + 1):
                result.append(" ".join(tokens[i : i + n]))
        return result

    @staticmethod
    def _best_token_match(
        q_tokens: List[str],
        q_ngrams: List[str],
        target_tokens: List[str],
    ) -> Optional[Tuple[str, float, str]]:
        """
        Returns (match_type, score, matched_span) or None.
        Priority: exact > partial > ngram.
        """
        if not target_tokens:
            return None

        target_joined = " ".join(target_tokens)

        # Exact: all target tokens appear somewhere in question tokens
        if all(t in q_tokens for t in target_tokens):
            return ("exact", 1.0, target_joined)

        # Partial: any single target token appears in question
        for t in target_tokens:
            if t in q_tokens:
                score = len(t) / max(len(tt) for tt in target_tokens)
                return ("partial", min(0.9, 0.5 + score * 0.4), t)

        # N-gram overlap
        for ng in q_ngrams:
            ng_tokens = ng.split()
            overlap = len(set(ng_tokens) & set(target_tokens))
            if overlap > 0:
                score = overlap / max(len(ng_tokens), len(target_tokens))
                if score >= 0.5:
                    return ("ngram", score * 0.7, ng)

        return None