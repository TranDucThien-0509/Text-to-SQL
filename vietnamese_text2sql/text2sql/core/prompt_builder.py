"""
PromptBuilder – assembles the final LLM prompt.

Prompt structure (DAIL-SQL style):
  1. System instruction   (Rule Implication)
  2. Schema (pruned DDL)  (CR_P)
  3. FK hint block        (optional)
  4. Schema linking hints (optional)
     4a. Relevant tables / columns (struct_hint)
     4b. Example column values     (value_hint) — NEW
  5. Few-shot examples    (DAIL_O)
  6. Target question      (with "SELECT" prefix)
     - Cell value hints (exact WHERE candidates) placed right before it

Supports token-budget truncation: if combined prompt exceeds budget,
few-shot examples are dropped one-by-one (least similar first).
"""
from __future__ import annotations

import logging
from typing import List, Optional, TYPE_CHECKING

from text2sql.schema.schema_linker import SchemaLinkingResult

if TYPE_CHECKING:
    from text2sql.schema.cell_value_retriever import CellValueRetriever

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 4  # rough approximation


class PromptBuilder:
    """
    Builds a complete Text-to-SQL prompt.

    Args:
        token_budget: Approximate max prompt length in tokens.
                      0 or None = no limit.
    """

    _SYSTEM_INSTRUCTION = (
        "Bạn là chuyên gia Text-to-SQL:\n"
        "1. Chỉ sử dụng tên bảng và tên cột CHÍNH XÁC như trong DDL (đúng từng dấu câu, dấu tiếng Việt).\n"
        "2. Ưu tiên sử dụng giá trị trong 'Cell value hints' cho mệnh đề WHERE — giữ nguyên case "
        "(viết hoa/thường) như trong hint. "
        "3. Thứ tự các cột trong SELECT phải đúng 100% như thứ tự được nhắc đến trong câu hỏi.\n"
        "4. KHÔNG tự ý bỏ dấu tiếng Việt của tên cột. Giữ nguyên định dạng gốc từ DDL.\n"
        "5. CHỈ xuất duy nhất câu lệnh SQL. Không giải thích, không dùng Markdown, không thêm văn bản thừa.\n"
        "6. Từ khóa SQL viết thường (select, from, where, join, group by...). "
        "7. Luôn sử dụng alias bảng t1, t2, t3... khi JOIN. "
        "Thứ tự trong ON clause: t1.col = t2.col (bảng xuất hiện trước trong FROM/JOIN đứng trước).\n"
        "8. Luôn thêm khoảng trắng giữa các toán tử và dấu ngoặc: count ( * ), avg ( col ).\n"
        "9. Giá trị chuỗi trong WHERE BẮT BUỘC đặt trong DẤU NHÁY KÉP. KHÔNG dùng nháy đơn.\n"
        "10. Khi WHERE có nhiều giá trị, dùng OR trực tiếp. KHÔNG dùng subquery lồng nhau:\n"
        "11. GROUP BY: chỉ group by cột thực sự cần thiết (thường là khóa chính). KHÔNG thêm cột thừa.\n"
        "12. Khi câu hỏi có từ 'khác nhau', 'phân biệt', 'duy nhất' → BẮT BUỘC dùng SELECT DISTINCT.\n"
        "13. Khi câu hỏi hỏi về 'lớn nhất', 'nhỏ nhất', 'cao nhất', 'thấp nhất' của một thuộc tính → "
        "dùng MAX() hoặc MIN() trong SELECT. KHÔNG thay thế bằng ORDER BY ... LIMIT 1 trừ khi "
        "cần lấy toàn bộ hàng đó.\n"
        "14. KHÔNG tự thêm JOIN nếu câu hỏi chỉ liên quan đến một bảng. "
        "Trước khi JOIN, kiểm tra: cột cần lấy và điều kiện WHERE có đủ trong một bảng không?\n"
        "15. Giá trị chuỗi trong WHERE LUÔN đặt trong dấu nháy kép, kể cả khi không có hint. "
        "Nếu giá trị chuỗi trong WHERE là tiếng Việt, dịch sang tiếng Anh tương đương trước khi đưa vào SQL:\n"
    )

    # OpenAI Demonstration prompt prefix (ODp) — Almohaimeed et al. IJCNN 2025,
    # Table II: ODp scores highest in zero-shot (58.6% EX) vs BSp (56.6%), CRp (57.9%).
    # Reason: forces LLM to output SQL only, no explanation or extra text.
    _ODp_PREFIX = "###Complete sqlite SQL query only and with no explanation\n"

    # Max distinct example values shown per relevant column.
    _MAX_VALUE_SAMPLES = 5

    def __init__(self, token_budget: int = 10000) -> None:
        self._budget = token_budget

    # ── Public API ────────────────────────────────────────────────────────

    def build(
        self,
        schema_text: str,
        few_shot_block: str,
        question: str,
        linking_result: Optional[SchemaLinkingResult] = None,
        cell_retriever: Optional["CellValueRetriever"] = None,
        db_id: Optional[str] = None,
    ) -> str:
        """
        Assemble the full prompt.

        Prompt order (updated per Almohaimeed et al. IJCNN 2025):
          1. ODp prefix       — forces SQL-only output
          2. System rules     — 15 Vietnamese SQL rules
          3. Schema (CR_P)    — pruned DDL
          4. FK/column hints  — table + column linking + example column values
                                (NO WHERE-specific cell values here)
          5. Few-shot         — DAIL_O examples
          6. Question block   — cell value hints + question + "select"
             (cell values placed RIGHT BEFORE the question so LLM sees
              them as context for the specific WHERE it needs to write)

        Args:
            cell_retriever: optional, used ONLY to fetch example distinct
                             values for relevant columns (via _get_values).
                             Pass None to skip this feature entirely.
            db_id:           required alongside cell_retriever to know which
                             SQLite database to query.

        Returns:
            Prompt string ending with "select " so LLM continues directly.
        """
        schema_section = f"/* Given the following database schema: */\n{schema_text}\n"

        # Split linking hints: structural hints go above few-shot,
        # cell values go directly before the question (higher locality = better recall)
        struct_hint, cell_hint = self._build_linking_hints_split(linking_result)

        # Example values per relevant column (distinct values, not question-scored) —
        # placed alongside struct_hint, above few-shot.
        value_hint = self._build_column_value_hints(linking_result, cell_retriever, db_id)
        schema_hint_block = "\n".join(filter(None, [struct_hint, value_hint]))

        # Build the question block: cell hints + question + SQL prefix
        question_block_parts = []
        if cell_hint:
            question_block_parts.append(cell_hint)
        question_block_parts.append(f"/* Answer the following: {question} */\nselect ")
        question_block = "\n".join(question_block_parts)

        few_shot = self._fit_few_shot(
            few_shot_block,
            fixed_parts=[
                self._ODp_PREFIX,
                self._SYSTEM_INSTRUCTION,
                schema_section,
                schema_hint_block,
                question_block,
            ],
        )

        parts = [
            self._ODp_PREFIX,
            self._SYSTEM_INSTRUCTION,
            "",
            schema_section,
        ]
        if schema_hint_block:
            parts += [schema_hint_block, ""]
        if few_shot:
            parts += [few_shot]
        parts.append(question_block)

        return "\n".join(parts)

    # ── Internals ─────────────────────────────────────────────────────────

    def _build_linking_hints_split(
        self, linking_result: Optional[SchemaLinkingResult]
    ) -> tuple:
        """
        Returns (struct_hint, cell_hint) as two separate strings.

        struct_hint: table + column matches → placed above few-shot examples
        cell_hint:   cell value matches     → placed directly before the question
                     (locality: LLM writes WHERE immediately after seeing the values)
        """
        if linking_result is None:
            return "", ""

        struct_parts: List[str] = []
        if linking_result.q_tab_match:
            tables = ", ".join(m.table_name for m in linking_result.q_tab_match)
            struct_parts.append(f"/* Relevant tables: {tables} */")
        if linking_result.q_col_match:
            cols = ", ".join(
                f"{m.table_name}.{m.col_name}" for m in linking_result.q_col_match
            )
            struct_parts.append(f"/* Relevant columns: {cols} */")

        cell_parts: List[str] = []
        if linking_result.cell_match:
            cells = ", ".join(
                f'{m.table_name}.{m.col_name} = "{m.matched_span}"'
                for m in linking_result.cell_match
                if m.matched_span
            )
            if cells:
                cell_parts.append(f"/* Cell value hints: {cells} */")

        return "\n".join(struct_parts), "\n".join(cell_parts)

    def _build_column_value_hints(
        self,
        linking_result: Optional[SchemaLinkingResult],
        cell_retriever: Optional["CellValueRetriever"],
        db_id: Optional[str],
    ) -> str:
        """
        Fetch example distinct values for columns already flagged as
        relevant by schema linking (q_col_match) or cell matching
        (cell_match), via CellValueRetriever._get_values() — raw distinct
        values, NOT scored against the question. This lets the LLM see
        the real data format of a column (e.g. giới_tính → "Nam"/"Nữ"
        vs "M"/"F") even when no specific WHERE value was matched.

        Returns "" if linking_result, cell_retriever, or db_id is missing,
        or if no relevant columns have any values.
        """
        if linking_result is None or cell_retriever is None or not db_id:
            return ""

        # Dedup relevant columns, preserving first-seen (table_name, col_name).
        relevant: dict = {}
        for m in linking_result.q_col_match:
            relevant.setdefault(m.col_global_idx, (m.table_name, m.col_name))
        for m in linking_result.cell_match:
            relevant.setdefault(m.col_global_idx, (m.table_name, m.col_name))

        if not relevant:
            return ""

        lines: List[str] = []
        for table_name, col_name in relevant.values():
            try:
                values = cell_retriever._get_values(db_id, table_name, col_name)
            except Exception:
                logger.debug(
                    "[PromptBuilder] Failed to fetch sample values for %s.%s",
                    table_name, col_name,
                    exc_info=True,
                )
                continue
            if not values:
                continue
            sample = ", ".join(f'"{v}"' for v in values[: self._MAX_VALUE_SAMPLES])
            lines.append(f"{table_name}.{col_name}: {sample}")

        if not lines:
            return ""

        return "/* Example column values:\n" + "\n".join(lines) + "\n*/"

    # Keep old method for backward compatibility (used by external callers)
    def _build_linking_hint(
        self, linking_result: Optional[SchemaLinkingResult]
    ) -> str:
        struct, cell = self._build_linking_hints_split(linking_result)
        return "\n".join(filter(None, [struct, cell]))

    def _fit_few_shot(self, few_shot_block: str, fixed_parts: List[str]) -> str:
        """
        If prompt exceeds budget, drop few-shot examples one-by-one
        (from the end = least-similar) until it fits.
        """
        if not self._budget:
            return few_shot_block

        fixed_chars = sum(len(p) for p in fixed_parts)
        available_chars = self._budget * _CHARS_PER_TOKEN - fixed_chars

        if available_chars <= 0:
            return ""

        if len(few_shot_block) <= available_chars:
            return few_shot_block

        # Split by double-newline (each example ends with \n\n)
        examples = [e + "\n\n" for e in few_shot_block.split("\n\n") if e.strip()]
        retained: List[str] = []
        used = 0
        for ex in examples:
            if used + len(ex) <= available_chars:
                retained.append(ex)
                used += len(ex)
            else:
                break

        if not retained:
            logger.debug("[PromptBuilder] Token budget too tight for any few-shot examples.")
            return ""

        logger.debug(
            "[PromptBuilder] Token budget: retained %d/%d few-shot examples.",
            len(retained), len(examples)
        )
        header_end = few_shot_block.find("\n") + 1
        header = few_shot_block[:header_end]
        return header + "".join(retained)