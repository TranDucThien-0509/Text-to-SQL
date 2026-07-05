"""
SQLNormalizer – normalizes and skeletonizes SQL queries.

Normalization:
  - Uppercase keywords
  - Collapse whitespace
  - Strip/standardize aliases
  - Normalize literals

Skeleton extraction:
  SELECT name FROM student WHERE age > 20
  → SELECT _ FROM _ WHERE _ > _
"""
from __future__ import annotations

import re
from typing import List

# SQL keywords for casing
_KEYWORDS = {
    "SELECT", "FROM", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER",
    "FULL", "CROSS", "ON", "AND", "OR", "NOT", "IN", "EXISTS", "BETWEEN",
    "LIKE", "IS", "NULL", "AS", "DISTINCT", "GROUP", "BY", "HAVING",
    "ORDER", "ASC", "DESC", "LIMIT", "OFFSET", "UNION", "INTERSECT",
    "EXCEPT", "ALL", "CASE", "WHEN", "THEN", "ELSE", "END", "WITH",
    "INSERT", "UPDATE", "DELETE", "SET", "INTO", "VALUES", "CREATE",
    "TABLE", "DROP", "ALTER", "INDEX", "PRIMARY", "KEY", "FOREIGN",
    "REFERENCES", "COUNT", "SUM", "AVG", "MIN", "MAX", "COALESCE",
    "IFNULL", "CAST", "SUBSTR", "LENGTH", "TRIM", "UPPER", "LOWER",
}

_NUM_PAT = re.compile(r"\b\d+(\.\d+)?\b")
_STR_PAT = re.compile(r"'[^']*'|\"[^\"]*\"")
_ALIAS_PAT = re.compile(r"\bAS\s+\w+", re.IGNORECASE)
_WS_PAT = re.compile(r"\s+")


class SQLNormalizer:
    """
    Provides SQL normalization and skeleton extraction utilities.
    All methods are stateless class methods – no instantiation needed.
    """

    @classmethod
    def normalize(cls, sql: str) -> str:
        """
        Canonical normalization:
          1. Uppercase SQL keywords
          2. Collapse whitespace
          3. Strip trailing semicolons
        """
        sql = sql.strip().rstrip(";")
        sql = cls._uppercase_keywords(sql)
        sql = _WS_PAT.sub(" ", sql)
        return sql.strip()

    @classmethod
    def skeleton(cls, sql: str) -> str:
        """
        Replace all literal values and identifiers with '_'.
        Used for SQL skeleton similarity retrieval.

        SELECT name FROM student WHERE age > 20
        → SELECT _ FROM _ WHERE _ > _
        """
        sql = cls.normalize(sql)
        sql = _STR_PAT.sub("_", sql)
        sql = _NUM_PAT.sub("_", sql)
        # Replace column/table identifiers (words not in keyword set)
        tokens = sql.split()
        result: List[str] = []
        for tok in tokens:
            clean = tok.strip("(),")
            if clean.upper() in _KEYWORDS or clean in ("_", ">", "<", "=", ">=", "<=", "!=", "*"):
                result.append(tok)
            else:
                result.append(tok.replace(clean, "_"))
        return " ".join(result)

    @classmethod
    def mask_values(cls, sql: str) -> str:
        """Replace only literal values (numbers/strings) with placeholders."""
        sql = _NUM_PAT.sub("NUM", sql)
        sql = _STR_PAT.sub("STR", sql)
        return " ".join(sql.split())

    @classmethod
    def extract_keywords(cls, sql: str) -> List[str]:
        """Return SQL clause keywords present in the query (for component matching)."""
        sql_upper = sql.upper()
        return [kw for kw in _KEYWORDS if re.search(rf"\b{kw}\b", sql_upper)]

    @classmethod
    def has_join(cls, sql: str) -> bool:
        return bool(re.search(r"\bJOIN\b", sql, re.IGNORECASE))

    @classmethod
    def has_subquery(cls, sql: str) -> bool:
        return bool(re.search(r"\bSELECT\b.*\bSELECT\b", sql, re.IGNORECASE | re.DOTALL))

    @classmethod
    def has_aggregation(cls, sql: str) -> bool:
        return bool(re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", sql, re.IGNORECASE))

    @classmethod
    def has_group_by(cls, sql: str) -> bool:
        return bool(re.search(r"\bGROUP\s+BY\b", sql, re.IGNORECASE))

    @classmethod
    def has_order_by(cls, sql: str) -> bool:
        return bool(re.search(r"\bORDER\s+BY\b", sql, re.IGNORECASE))

    @classmethod
    def complexity_label(cls, sql: str) -> str:
        """
        Rough Spider complexity classification.
        Returns one of: 'easy' | 'medium' | 'hard' | 'extra'
        """
        sql_u = sql.upper()
        has_nested = cls.has_subquery(sql)
        has_agg = cls.has_aggregation(sql)
        has_grp = cls.has_group_by(sql)
        has_join = cls.has_join(sql)
        join_count = len(re.findall(r"\bJOIN\b", sql_u))

        if has_nested or join_count >= 3:
            return "extra"
        if join_count >= 2 or (has_join and has_grp):
            return "hard"
        if has_join or has_agg or has_grp:
            return "medium"
        return "easy"

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _uppercase_keywords(sql: str) -> str:
        def _replace(m: re.Match) -> str:
            word = m.group(0)
            return word.upper() if word.upper() in _KEYWORDS else word

        return re.sub(r"\b[A-Za-z_]\w*\b", _replace, sql)