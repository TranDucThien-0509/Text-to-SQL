"""
SchemaProcessor – CR_P (Code Representation with Primary/Foreign Keys).

Converts Spider/BIRD table metadata into CREATE TABLE DDL strings that
give the LLM full structural awareness of the database.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────

class ColumnInfo:
    """Lightweight wrapper for a single column's metadata."""

    __slots__ = ("global_idx", "table_idx", "name", "col_type", "is_pk")

    def __init__(
        self,
        global_idx: int,
        table_idx: int,
        name: str,
        col_type: str,
        is_pk: bool = False,
    ) -> None:
        self.global_idx = global_idx
        self.table_idx = table_idx
        self.name = name
        self.col_type = col_type
        self.is_pk = is_pk


class DatabaseSchema:
    """Holds fully parsed metadata for one database."""

    def __init__(
        self,
        db_id: str,
        table_names: List[str],
        columns: List[ColumnInfo],
        foreign_keys: List[Tuple[int, int]],   # (src_col_idx, dst_col_idx)
        ddl: str,
    ) -> None:
        self.db_id = db_id
        self.table_names = table_names
        self.columns = columns
        self.foreign_keys = foreign_keys
        self.ddl = ddl

        # Build quick-access indexes
        self._table_columns: Dict[int, List[ColumnInfo]] = {}
        for col in columns:
            self._table_columns.setdefault(col.table_idx, []).append(col)

    def get_table_columns(self, table_idx: int) -> List[ColumnInfo]:
        return self._table_columns.get(table_idx, [])

    def get_fk_neighbors(self, table_idx: int) -> Set[int]:
        """Return table indices reachable from *table_idx* via any FK edge."""
        col_to_table = {c.global_idx: c.table_idx for c in self.columns}
        neighbors: Set[int] = set()
        for src, dst in self.foreign_keys:
            src_t = col_to_table.get(src)
            dst_t = col_to_table.get(dst)
            if src_t == table_idx and dst_t is not None:
                neighbors.add(dst_t)
            elif dst_t == table_idx and src_t is not None:
                neighbors.add(src_t)
        return neighbors


# ─────────────────────────────────────────────────────────────
# Processor
# ─────────────────────────────────────────────────────────────

class SchemaProcessor:
    """
    Loads Spider/BIRD tables.json and exposes:
      - get_ddl(db_id)           → full CREATE TABLE DDL string
      - get_schema(db_id)        → DatabaseSchema object
      - get_pruned_ddl(db_id, relevant_tables, relevant_columns)
    """

    def __init__(self, tables_path: Path) -> None:
        self._tables_path = tables_path
        self._schemas: Dict[str, DatabaseSchema] = {}

    # ── Public API ────────────────────────────────────────────

    def load(self) -> "SchemaProcessor":
        logger.info("Loading schemas (CR_P format) from %s", self._tables_path)
        with open(self._tables_path, "r", encoding="utf-8") as fh:
            raw: List[dict] = json.load(fh)

        for db in raw:
            schema = self._parse_db(db)
            self._schemas[schema.db_id] = schema

        logger.info("Loaded %d database schemas.", len(self._schemas))
        return self

    def get_ddl(self, db_id: str) -> str:
        """Return full CREATE TABLE DDL for a database."""
        schema = self._get_schema_or_raise(db_id)
        return schema.ddl

    def get_schema(self, db_id: str) -> DatabaseSchema:
        return self._get_schema_or_raise(db_id)

    def get_pruned_ddl(
        self,
        db_id: str,
        relevant_table_indices: Optional[Set[int]] = None,
        relevant_col_indices: Optional[Set[int]] = None,
    ) -> str:
        """
        Return DDL restricted to relevant tables and/or columns.
        If no filter sets are provided, returns the full DDL.
        """
        schema = self._get_schema_or_raise(db_id)
        if relevant_table_indices is None and relevant_col_indices is None:
            return schema.ddl

        return self._build_ddl(
            schema,
            allowed_tables=relevant_table_indices,
            allowed_cols=relevant_col_indices,
        )

    def list_databases(self) -> List[str]:
        return list(self._schemas.keys())

    # ── Internal helpers ──────────────────────────────────────

    def _get_schema_or_raise(self, db_id: str) -> DatabaseSchema:
        schema = self._schemas.get(db_id)
        if schema is None:
            raise KeyError(f"Schema not found for db_id='{db_id}'")
        return schema

    @staticmethod
    def _parse_db(db: dict) -> DatabaseSchema:
        db_id: str = db["db_id"]
        table_names: List[str] = db["table_names_original"]
        raw_columns: List[Tuple[int, str]] = db["column_names_original"]
        col_types: List[str] = db.get("column_types", [])
        primary_keys: Set[int] = set(db.get("primary_keys", []))
        foreign_keys: List[List[int]] = db.get("foreign_keys", [])

        columns: List[ColumnInfo] = []
        for g_idx, (t_idx, col_name) in enumerate(raw_columns):
            if t_idx < 0:
                continue   # skip virtual '*' column
            c_type = col_types[g_idx].upper() if g_idx < len(col_types) else "TEXT"
            columns.append(
                ColumnInfo(
                    global_idx=g_idx,
                    table_idx=t_idx,
                    name=col_name,
                    col_type=c_type,
                    is_pk=(g_idx in primary_keys),
                )
            )

        fk_pairs: List[Tuple[int, int]] = [(s, d) for s, d in foreign_keys]
        schema = DatabaseSchema(
            db_id=db_id,
            table_names=table_names,
            columns=columns,
            foreign_keys=fk_pairs,
            ddl="",   # placeholder; filled below
        )
        schema.ddl = SchemaProcessor._build_ddl(schema)
        return schema

    @staticmethod
    def _build_ddl(
        schema: DatabaseSchema,
        allowed_tables: Optional[Set[int]] = None,
        allowed_cols: Optional[Set[int]] = None,
    ) -> str:
        col_to_table = {c.global_idx: c.table_idx for c in schema.columns}

        # Pre-build FK clauses per table
        fk_clauses: Dict[int, List[str]] = {}
        for src_idx, dst_idx in schema.foreign_keys:
            src_table = col_to_table.get(src_idx)
            dst_table = col_to_table.get(dst_idx)
            if src_table is None or dst_table is None:
                continue
            if allowed_tables and src_table not in allowed_tables:
                continue
            src_col_obj = next((c for c in schema.columns if c.global_idx == src_idx), None)
            dst_col_obj = next((c for c in schema.columns if c.global_idx == dst_idx), None)
            if src_col_obj and dst_col_obj:
                dst_tbl_name = schema.table_names[dst_table]
                fk_clauses.setdefault(src_table, []).append(
                    f"  FOREIGN KEY ({src_col_obj.name})"
                    f" REFERENCES {dst_tbl_name}({dst_col_obj.name})"
                )

        blocks: List[str] = []
        for t_idx, t_name in enumerate(schema.table_names):
            if allowed_tables and t_idx not in allowed_tables:
                continue

            table_cols = schema.get_table_columns(t_idx)
            col_defs: List[str] = []
            for col in table_cols:
                if allowed_cols and col.global_idx not in allowed_cols:
                    continue
                pk = " PRIMARY KEY" if col.is_pk else ""
                col_defs.append(f"  {col.name} {col.col_type}{pk}")

            col_defs.extend(fk_clauses.get(t_idx, []))
            if not col_defs:
                continue

            body = ",\n".join(col_defs)
            blocks.append(f"CREATE TABLE {t_name} (\n{body}\n);")

        return "\n\n".join(blocks)