"""
SchemaGraphLinker – graph-based schema linking theo SchemaGraphSQL (Safdarian et al., 2025).

Paper: "SchemaGraphSQL: Efficient Schema Linking with Pathfinding Graph Algorithms
        for Text-to-SQL on Large-Scale Databases", arXiv:2505.18363

Idea cốt lõi:
  1. Xây schema graph G = (Tables, ForeignKeys) từ DatabaseSchema
  2. Dùng SchemaLinker hiện tại để xác định "seed tables" (bảng được mention trong câu hỏi)
  3. Enumerate tất cả shortest paths giữa các seed tables trên FK graph
  4. Union các paths → connected sub-schema đảm bảo không thiếu bảng JOIN trung gian
  5. Trả về SchemaLinkingResult mở rộng với graph_tables (thêm vào q_tab_match)

Cách dùng (drop-in thay SchemaPruner trong pipeline):
    linker   = SchemaLinker()
    gl       = SchemaGraphLinker()
    base     = linker.link(question, schema, cell_retriever)
    enhanced = gl.enhance(base, schema)
    pruned   = schema_processor.get_pruned_ddl(
                   schema.db_id,
                   relevant_table_indices=enhanced.graph_table_indices,
               )

Hoặc dùng standalone (không cần SchemaLinker trước):
    result = SchemaGraphLinker().link(question, schema, cell_retriever)
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from text2sql.schema.schema_processor import DatabaseSchema
from text2sql.schema.schema_linker import (
    ColumnMatch,
    SchemaLinker,
    SchemaLinkingResult,
    TableMatch,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# FK Graph
# ─────────────────────────────────────────────────────────────

class FKGraph:
    """
    Undirected adjacency graph: node = table_idx, edge = FK relation.

    Augmentation rule (từ paper):
    Nếu schema quá sparse (< 2 edges), thêm edges giữa các bảng
    có cột tên chứa "id" — đảm bảo graph đủ connected cho path enumeration.
    """

    def __init__(self, schema: DatabaseSchema) -> None:
        self._n = len(schema.table_names)
        self._adj: Dict[int, Set[int]] = {i: set() for i in range(self._n)}
        self._edge_count = 0
        self._build(schema)

    def _build(self, schema: DatabaseSchema) -> None:
        col_to_table = {c.global_idx: c.table_idx for c in schema.columns}

        for src_col, dst_col in schema.foreign_keys:
            src_t = col_to_table.get(src_col)
            dst_t = col_to_table.get(dst_col)
            if src_t is None or dst_t is None or src_t == dst_t:
                continue
            self._adj[src_t].add(dst_t)
            self._adj[dst_t].add(src_t)
            self._edge_count += 1

        # Augmentation: nếu quá sparse → thêm edges dựa trên tên cột chứa "id"
        if self._edge_count < 2:
            id_tables: List[int] = []
            for col in schema.columns:
                if "id" in col.name.lower():
                    id_tables.append(col.table_idx)
            for i in range(len(id_tables)):
                for j in range(i + 1, len(id_tables)):
                    a, b = id_tables[i], id_tables[j]
                    if a != b and b not in self._adj[a]:
                        self._adj[a].add(b)
                        self._adj[b].add(a)

        logger.debug(
            "[FKGraph] Built: %d tables, %d FK edges (after augment: %d adj edges total)",
            self._n,
            self._edge_count,
            sum(len(v) for v in self._adj.values()) // 2,
        )

    def neighbors(self, node: int) -> Set[int]:
        return self._adj.get(node, set())

    def all_shortest_paths(self, src: int, dst: int) -> List[List[int]]:
        """
        BFS để tìm TẤT CẢ shortest simple paths từ src → dst.
        Trả về list of paths (mỗi path là list of table_idx).

        Dùng BFS layer-by-layer: mỗi bước expand tất cả paths có độ dài hiện tại.
        Dừng lại ngay khi tìm thấy dst ở một layer (không đi sâu hơn).
        """
        if src == dst:
            return [[src]]

        # BFS: lưu (current_node, path_so_far)
        # Dùng visited theo layer để tránh loop nhưng vẫn tìm được tất cả paths cùng độ dài
        found_depth: Optional[int] = None
        paths: List[List[int]] = []

        queue: deque = deque()
        queue.append([src])
        visited_at_depth: Dict[int, int] = {src: 0}  # node -> depth khi lần đầu gặp

        while queue:
            path = queue.popleft()
            current = path[-1]
            depth = len(path) - 1

            # Nếu đã vượt quá depth tìm thấy → dừng
            if found_depth is not None and depth >= found_depth:
                continue

            for neighbor in self.neighbors(current):
                if neighbor in path:
                    continue  # tránh cycle

                new_path = path + [neighbor]
                new_depth = depth + 1

                if neighbor == dst:
                    found_depth = new_depth
                    paths.append(new_path)
                else:
                    # Chỉ thêm vào queue nếu neighbor chưa bị thăm ở depth nhỏ hơn
                    prev_depth = visited_at_depth.get(neighbor, float("inf"))
                    if new_depth <= prev_depth:
                        visited_at_depth[neighbor] = new_depth
                        queue.append(new_path)

        return paths if paths else []

    def reachable_within(self, sources: Set[int], max_hops: int) -> Set[int]:
        """BFS từ nhiều sources, tối đa max_hops bước."""
        visited = set(sources)
        frontier = set(sources)
        for _ in range(max_hops):
            next_f: Set[int] = set()
            for node in frontier:
                for nb in self.neighbors(node):
                    if nb not in visited:
                        visited.add(nb)
                        next_f.add(nb)
            frontier = next_f
            if not frontier:
                break
        return visited


# ─────────────────────────────────────────────────────────────
# Enhanced result container
# ─────────────────────────────────────────────────────────────

@dataclass
class GraphEnhancedLinkingResult(SchemaLinkingResult):
    """
    Extends SchemaLinkingResult với thông tin graph path.
    Tương thích hoàn toàn với SchemaPruner vì kế thừa SchemaLinkingResult.
    """
    # Table indices được chọn qua graph path enumeration
    graph_table_indices: Set[int] = field(default_factory=set)

    # Các path được tìm ra (để debug / logging)
    join_paths: List[List[str]] = field(default_factory=list)  # list[list[table_name]]

    @property
    def relevant_table_indices(self) -> Set[int]:
        """Override: dùng graph tables thay vì chỉ text-matched tables."""
        return self.graph_table_indices if self.graph_table_indices else super().relevant_table_indices


# ─────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────

class SchemaGraphLinker:
    """
    Graph-based schema linker.

    Mode mặc định: force_union=True (paper cho thấy đây là config tốt nhất,
    Recall=95.71%, F6=95.43% trên BIRD benchmark).

    Args:
        force_union:   True = lấy union TẤT CẢ shortest paths (max recall).
                       False = chỉ lấy 1 path ngắn nhất (max precision).
        fallback_hops: Nếu không tìm được path (graph disconnected),
                       fallback về BFS expansion max N hops từ seed tables.
        base_linker:   SchemaLinker instance dùng để lấy seed tables.
                       Nếu None, tự khởi tạo mặc định.
    """

    def __init__(
        self,
        force_union: bool = True,
        fallback_hops: int = 1,
        base_linker: Optional[SchemaLinker] = None,
    ) -> None:
        self._force_union = force_union
        self._fallback_hops = fallback_hops
        self._base_linker = base_linker or SchemaLinker()

    # ── Public API ────────────────────────────────────────────

    def link(
        self,
        question: str,
        schema: DatabaseSchema,
        cell_retriever=None,
    ) -> GraphEnhancedLinkingResult:
        """
        Full pipeline: text linking → graph enhancement → enhanced result.
        """
        base = self._base_linker.link(question, schema, cell_retriever)
        return self.enhance(base, schema)

    def enhance(
        self,
        base: SchemaLinkingResult,
        schema: DatabaseSchema,
    ) -> GraphEnhancedLinkingResult:
        """
        Nhận kết quả từ SchemaLinker hiện tại và bổ sung graph path tables.

        Có thể dùng standalone nếu đã có SchemaLinkingResult:
            base = linker.link(q, schema, cell_retriever)
            enhanced = graph_linker.enhance(base, schema)
        """
        graph = FKGraph(schema)
        seed_tables = self._collect_seed_tables(base, schema)

        if not seed_tables:
            # Không có seed nào → trả về toàn bộ schema (safe fallback)
            logger.debug("[SchemaGraphLinker] %s: no seed tables, using full schema.", schema.db_id)
            result = self._to_enhanced(base)
            result.graph_table_indices = set(range(len(schema.table_names)))
            return result

        if len(seed_tables) == 1:
            # Chỉ 1 bảng → không cần path, nhưng vẫn expand FK neighbors 1 hop
            # để bắt các bảng thường JOIN kèm
            expanded = graph.reachable_within(seed_tables, max_hops=self._fallback_hops)
            result = self._to_enhanced(base)
            result.graph_table_indices = expanded
            result.join_paths = [[schema.table_names[t] for t in sorted(expanded)]]
            logger.debug(
                "[SchemaGraphLinker] %s: 1 seed → FK expand %d tables.",
                schema.db_id, len(expanded),
            )
            return result

        # Enumerate all shortest paths giữa các pairs seed tables
        path_table_sets, join_paths = self._enumerate_paths(
            seed_tables, graph, schema
        )

        if not path_table_sets:
            # Graph disconnected → fallback BFS expansion
            logger.debug(
                "[SchemaGraphLinker] %s: no paths found (disconnected graph), fallback BFS.",
                schema.db_id,
            )
            expanded = graph.reachable_within(seed_tables, max_hops=self._fallback_hops)
            result = self._to_enhanced(base)
            result.graph_table_indices = expanded | seed_tables
            return result

        # Union tất cả paths (force_union mode) hoặc chỉ lấy 1 path
        if self._force_union:
            union_tables: Set[int] = set()
            for ts in path_table_sets:
                union_tables.update(ts)
        else:
            # Lấy path nhỏ nhất (precision-focused)
            union_tables = min(path_table_sets, key=len)

        logger.debug(
            "[SchemaGraphLinker] %s: seed=%s → graph=%s (%d paths, %d tables)",
            schema.db_id,
            {schema.table_names[t] for t in seed_tables},
            {schema.table_names[t] for t in union_tables},
            len(path_table_sets),
            len(union_tables),
        )

        result = self._to_enhanced(base)
        result.graph_table_indices = union_tables
        result.join_paths = join_paths

        # Bổ sung TableMatch mới cho các bảng được thêm qua graph
        # (để SchemaPruner và PromptBuilder nhận ra)
        existing_tab_indices = {m.table_idx for m in result.q_tab_match}
        for t_idx in union_tables:
            if t_idx not in existing_tab_indices:
                result.q_tab_match.append(TableMatch(
                    table_idx=t_idx,
                    table_name=schema.table_names[t_idx],
                    match_type="graph_path",
                    score=0.9,
                    matched_span=None,
                ))
                existing_tab_indices.add(t_idx)

        return result

    # ── Internals ─────────────────────────────────────────────

    def _collect_seed_tables(
        self, base: SchemaLinkingResult, schema: DatabaseSchema
    ) -> Set[int]:
        """
        Lấy seed tables từ kết quả SchemaLinker.
        Seed = bảng được mention trực tiếp (q_tab_match exact/partial)
             + bảng chứa cột được mention (q_col_match exact)
             + bảng chứa cell match
        """
        seeds: Set[int] = set()
        col_to_table = {c.global_idx: c.table_idx for c in schema.columns}

        # Table matches (tất cả types)
        for m in base.q_tab_match:
            seeds.add(m.table_idx)

        # Column exact matches → lấy bảng chứa cột đó
        for m in base.q_col_match:
            if m.match_type == "exact":
                t = col_to_table.get(m.col_global_idx)
                if t is not None:
                    seeds.add(t)

        # Cell matches → bảng chứa cell value
        for m in base.cell_match:
            t = col_to_table.get(m.col_global_idx)
            if t is not None:
                seeds.add(t)

        return seeds

    def _enumerate_paths(
        self,
        seeds: Set[int],
        graph: FKGraph,
        schema: DatabaseSchema,
    ) -> Tuple[List[Set[int]], List[List[str]]]:
        """
        Enumerate shortest paths giữa tất cả pairs (src, dst) trong seed set.
        Trả về (list_of_table_sets, list_of_table_name_paths) để debug.
        """
        seed_list = sorted(seeds)
        path_sets: List[Set[int]] = []
        join_paths: List[List[str]] = []

        for i, src in enumerate(seed_list):
            for dst in seed_list[i + 1:]:
                paths = graph.all_shortest_paths(src, dst)
                for p in paths:
                    path_sets.append(set(p))
                    join_paths.append([schema.table_names[t] for t in p])

        # Luôn include seed tables trong result
        if path_sets:
            for ts in path_sets:
                ts.update(seeds)
        else:
            # Không có path nào
            pass

        return path_sets, join_paths

    @staticmethod
    def _to_enhanced(base: SchemaLinkingResult) -> GraphEnhancedLinkingResult:
        """Convert SchemaLinkingResult → GraphEnhancedLinkingResult."""
        return GraphEnhancedLinkingResult(
            db_id=base.db_id,
            question=base.question,
            q_col_match=list(base.q_col_match),
            q_tab_match=list(base.q_tab_match),
            cell_match=list(base.cell_match),
            num_date_match=list(base.num_date_match),
        )
