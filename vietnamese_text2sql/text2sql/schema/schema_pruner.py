"""
SchemaPruner – reduces the schema to only relevant tables/columns.

Strategy:
  1. Include all tables/columns directly matched by SchemaLinker.
  2. Expand via FK graph traversal (configurable depth).
  3. Always include PK columns of selected tables.
  4. Re-render DDL with the pruned subset.
"""
from __future__ import annotations

import logging
from typing import Optional, Set

from text2sql.schema.schema_processor import DatabaseSchema, SchemaProcessor
from text2sql.schema.schema_linker import SchemaLinkingResult

logger = logging.getLogger(__name__)


class SchemaPruner:
    """
    Prunes a full DatabaseSchema to relevant subset.

    Args:
        fk_expand_depth: How many FK hops to follow from a matched table.
                         0 = no expansion, 1 = immediate neighbours, etc.
        min_tables: Keep at least this many tables even if none matched.
    """

    def __init__(self, fk_expand_depth: int = 1, min_tables: int = 1) -> None:
        self._fk_expand_depth = fk_expand_depth
        self._min_tables = min_tables

    def prune(
        self,
        schema: DatabaseSchema,
        linking_result: SchemaLinkingResult,
        schema_processor: SchemaProcessor,
    ) -> str:
        """
        Return a DDL string containing only the relevant subset.

        Falls back to the full schema if pruning would leave fewer
        than `min_tables` tables.
        """
        relevant_tables = self._collect_tables(schema, linking_result)

        if len(relevant_tables) < self._min_tables:
            logger.debug(
                "[SchemaPruner] %s: pruning kept %d tables (< min %d); using full schema.",
                schema.db_id,
                len(relevant_tables),
                self._min_tables,
            )
            return schema_processor.get_ddl(schema.db_id)

        relevant_cols = self._collect_columns(schema, linking_result, relevant_tables)

        logger.debug(
            "[SchemaPruner] %s: %d/%d tables, %d cols retained.",
            schema.db_id,
            len(relevant_tables),
            len(schema.table_names),
            len(relevant_cols),
        )

        return schema_processor.get_pruned_ddl(
            schema.db_id,
            relevant_table_indices=relevant_tables,
            relevant_col_indices=relevant_cols,
        )

    # ── Internals ──────────────────────────────────────────────────────────

    def _collect_tables(
        self, schema: DatabaseSchema, linking_result: SchemaLinkingResult
    ) -> Set[int]:
        """Gather directly-matched tables then expand via FK graph."""
        seed: Set[int] = set()

        # From table matches
        for m in linking_result.q_tab_match:
            seed.add(m.table_idx)

        # From column matches (all types)
        col_idx_to_table = {c.global_idx: c.table_idx for c in schema.columns}
        for col_m in (
            *linking_result.q_col_match,
            *linking_result.cell_match,
            *linking_result.num_date_match,
        ):
            t = col_idx_to_table.get(col_m.col_global_idx)
            if t is not None:
                seed.add(t)

        # FK expansion
        expanded = set(seed)
        frontier = set(seed)
        for _ in range(self._fk_expand_depth):
            next_frontier: Set[int] = set()
            for t_idx in frontier:
                neighbors = schema.get_fk_neighbors(t_idx)
                new = neighbors - expanded
                expanded.update(new)
                next_frontier.update(new)
            frontier = next_frontier
            if not frontier:
                break

        return expanded

    @staticmethod
    def _collect_columns(
        schema: DatabaseSchema,
        linking_result: SchemaLinkingResult,
        relevant_tables: Set[int],
    ) -> Set[int]:
        """Columns = matched cols + all PKs of retained tables."""
        col_indices: Set[int] = set()

        # Matched columns (only those in relevant tables)
        col_idx_to_table = {c.global_idx: c.table_idx for c in schema.columns}
        for col_m in (
            *linking_result.q_col_match,
            *linking_result.cell_match,
            *linking_result.num_date_match,
        ):
            if col_idx_to_table.get(col_m.col_global_idx) in relevant_tables:
                col_indices.add(col_m.col_global_idx)

        # Always include PK columns (essential for JOIN clarity)
        for col in schema.columns:
            if col.is_pk and col.table_idx in relevant_tables:
                col_indices.add(col.global_idx)

        # FK columns required to represent FK constraints
        for src_idx, dst_idx in schema.foreign_keys:
            src_t = col_idx_to_table.get(src_idx)
            dst_t = col_idx_to_table.get(dst_idx)
            if src_t in relevant_tables and dst_t in relevant_tables:
                col_indices.add(src_idx)
                col_indices.add(dst_idx)

        return col_indices