"""
PromptBuilder – assembles the final LLM prompt.

Prompt structure (DAIL-SQL style):
  1. System instruction   (Rule Implication)
  2. Schema (pruned DDL)  (CR_P)
  3. FK hint block        (optional)
  4. Schema linking hints (optional)
  5. Few-shot examples    (DAIL_O)
  6. Target question      (with "SELECT" prefix)

Supports token-budget truncation: if combined prompt exceeds budget,
few-shot examples are dropped one-by-one (least similar first).
"""
from __future__ import annotations

import logging
from typing import List, Optional

from text2sql.schema.schema_linker import SchemaLinkingResult

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
            "Bạn là chuyên gia Text-to-SQL. Quy tắc tối thượng:\n"
            "1. Chỉ sử dụng tên bảng và tên cột CHÍNH XÁC như trong DDL (đúng từng dấu câu).\n"
            "2. Ưu tiên sử dụng giá trị trong 'Cell value hints' cho mệnh đề WHERE. "
            "Ví dụ: nếu câu hỏi là 'nữ' và hint có 'female', phải dùng 'female'.\n"
            "3. Thứ tự các cột trong SELECT phải đúng 100% như thứ tự được nhắc đến trong câu hỏi.\n"
            "4. KHÔNG tự ý bỏ dấu tiếng Việt của tên cột. Giữ nguyên định dạng gốc.\n"
            "5. CHỈ xuất duy nhất câu lệnh SQL. Không giải thích, không dùng Markdown (không nháy ```), không thêm văn bản thừa.\n"
            "6. Dữ liệu đầu ra viết thường hoàn toàn (lowercase), bao gồm cả các từ khóa SQL (select, from, where, join...).\n"
            "7. Luôn sử dụng Alias bảng theo định dạng t1, t2, t3... khi thực hiện JOIN.\n"
            "8. Luôn thêm khoảng trắng giữa các toán tử và dấu ngoặc (ví dụ: count ( * ) thay vì count(*)).\n"
            "9. Khi điều kiện WHERE là một chuỗi văn bản (ví dụ: tên riêng, địa điểm), BẮT BUỘC phải đặt trong dấu nháy kép hoặc nháy đơn"
            "(ví dụ: where loại = 'Grondzeiler')."
        )

    def __init__(self, token_budget: int = 3000) -> None:
        self._budget = token_budget

    # ── Public API ────────────────────────────────────────────────────────

    def build(
        self,
        schema_text: str,
        few_shot_block: str,
        question: str,
        linking_result: Optional[SchemaLinkingResult] = None,
    ) -> str:
        """
        Assemble the full prompt.

        Returns:
            Prompt string ending with "SELECT " so the LLM continues directly
            from the first keyword of the answer.
        """
        schema_section = f"/* Given the following database schema: */\n{schema_text}\n"
        linking_hint = self._build_linking_hint(linking_result)
        question_line = f"/* Answer the following: {question} */\nSELECT "

        # Try to fit within token budget
        few_shot = self._fit_few_shot(
            few_shot_block,
            fixed_parts=[
                self._SYSTEM_INSTRUCTION,
                schema_section,
                linking_hint,
                question_line,
            ],
        )

        parts = [
            self._SYSTEM_INSTRUCTION,
            "",
            schema_section,
        ]
        if linking_hint:
            parts += [linking_hint, ""]
        if few_shot:
            parts += [few_shot]
        parts.append(question_line)

        return "\n".join(parts)

    # ── Internals ─────────────────────────────────────────────────────────

    def _build_linking_hint(
        self, linking_result: Optional[SchemaLinkingResult]
    ) -> str:
        if linking_result is None:
            return ""

        hints: List[str] = []
        if linking_result.q_tab_match:
            tables = ", ".join(m.table_name for m in linking_result.q_tab_match)
            hints.append(f"/* Relevant tables: {tables} */")
        if linking_result.q_col_match:
            cols = ", ".join(
                f"{m.table_name}.{m.col_name}" for m in linking_result.q_col_match
            )
            hints.append(f"/* Relevant columns: {cols} */")
        if linking_result.cell_match:
            cells = ", ".join(
                f"{m.table_name}.{m.col_name} = '{m.matched_span}'"
                for m in linking_result.cell_match
                if m.matched_span
            )
            if cells:
                hints.append(f"/* Cell value hints: {cells} */")

        return "\n".join(hints)

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