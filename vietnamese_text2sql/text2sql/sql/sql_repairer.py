"""
SQLRepairer – self-correction loop for failed SQL queries.

Flow:
  1. Execute generated SQL
  2. If failed → build repair prompt with error feedback
  3. Re-call LLM → re-execute
  4. Repeat up to max_attempts times
  5. Return best result (last successful or original)

Improvements over v1:
  - Repair prompt no longer ends with bare "SELECT " to avoid double-SELECT
  - Progressive context: each repair attempt includes previous failed SQLs
  - Schema truncation to avoid exceeding context window
  - Execution-guided best-attempt selection (picks first success, not just any)
  - Richer logging for debugging
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from text2sql.sql.sql_executor import ExecutionResult, ExecStatus, SQLExecutor

if TYPE_CHECKING:
    from text2sql.core.llm_client import OpenRouterClient

logger = logging.getLogger(__name__)

# Max characters of schema DDL to include in repair prompt.
# Prevents blowing up context window on wide schemas.
_MAX_SCHEMA_CHARS = 4_000


@dataclass
class RepairAttempt:
    attempt_num: int
    sql: str               # SQL tiếng Việt (LLM output)
    translated_sql: str    # SQL tiếng Anh (sau schema_matching)
    exec_result: ExecutionResult
    repair_prompt_snippet: str = ""


@dataclass
class RepairOutcome:
    final_sql: str
    success: bool
    attempts: List[RepairAttempt] = field(default_factory=list)

    @property
    def num_repairs(self) -> int:
        return max(0, len(self.attempts) - 1)


class SQLRepairer:
    """
    Wraps the generate → execute → repair loop.

    Args:
        executor:     SQLExecutor instance.
        llm:          OpenRouterClient instance.
        max_attempts: Total attempts including the first generation.
    """

    # GAT Corrector style (paper: Almohaimeed et al., IJCNN 2025):
    # Detect + fix errors in ONE step — no intermediate True/False classification.
    # Ablation study shows this is far superior for non-English languages
    # (97/100 vs 67/100 correct on Arabic; same principle applies to Vietnamese).
    #
    # Key difference from old _REPAIR_TEMPLATE:
    #   OLD: "there is an error, fix it" → model knows it MUST change something
    #   NEW: "output the SQL whether correct or not" → model decides if fix needed
    #        This prevents the model from over-correcting correct-but-unexecutable SQL.

    _CORRECTOR_SYSTEM = (
        "Bạn là trợ lý kiểm tra và sửa SQL.\n"
        "Nhận vào: schema cơ sở dữ liệu, câu hỏi tiếng Việt, và câu SQL được sinh ra.\n"
        "Quy tắc:\n"
        "- Nếu SQL đúng và trả lời đúng câu hỏi: xuất nguyên câu SQL đó.\n"
        "- Nếu SQL sai (lỗi cú pháp, sai bảng/cột, sai logic): xuất câu SQL đã sửa.\n"
        "Chỉ xuất SQL duy nhất. Không giải thích, không markdown."
    )

    # Used when executor returns an error message (syntax/runtime errors).
    _CORRECTOR_TEMPLATE_WITH_ERROR = """\
{schema}

Câu hỏi: {question}
SQL: {failed_sql}
Lỗi khi thực thi: {error}
{value_hint_block}{prev_failures_block}"""

    # Used for logic errors (wrong result, no error message from executor).
    _CORRECTOR_TEMPLATE_NO_ERROR = """\
{schema}

Câu hỏi: {question}
SQL: {failed_sql}
{value_hint_block}{prev_failures_block}"""

    _PREV_FAILURE_TEMPLATE = """\
Các lần thử trước đã thất bại (không lặp lại các lỗi này):
{failures}
"""

    # Dùng khi _fix_value_mismatch KHÔNG tìm/sửa được giá trị nào (cột không
    # tồn tại ở bảng dự đoán, hoặc distinct values rỗng). Thay vì bỏ qua,
    # dump toàn bộ giá trị mẫu của các cột dạng text trong TẤT CẢ các bảng
    # để GAT Corrector có đủ ngữ cảnh "rethink" và tự chọn giá trị đúng.
    _VALUE_DUMP_TEMPLATE = """\
Giá trị thực tế có trong CSDL, theo từng cột (table.column: "v1", "v2", ...):
{values}
BẮT BUỘC: với mỗi điều kiện WHERE trên một cột có trong danh sách trên, PHẢI \
chọn CHÍNH XÁC MỘT giá trị có trong danh sách đó (giữ nguyên chính tả/hoa \
thường), không được tự bịa hay giữ nguyên giá trị cũ nếu nó không khớp.
"""

    def __init__(
        self,
        executor: SQLExecutor,
        llm: "OpenRouterClient",
        max_attempts: int = 3,
        translate_fn: Optional[Callable[[str, str], str]] = None,
        db_dir: Optional[str] = None,
        value_dump_max_values_per_col: int = 10,
        value_dump_max_chars: int = 2000,
    ) -> None:
        self._executor = executor
        self._llm = llm
        self._max_attempts = max_attempts
        self._translate = translate_fn
        self._db_dir = db_dir  # dùng để query DISTINCT values cho value matching
        self._value_dump_max_values_per_col = value_dump_max_values_per_col
        self._value_dump_max_chars = value_dump_max_chars

    # ── Public API ─────────────────────────────────────────────────────────

    def repair(
        self,
        initial_sql: str,
        db_id: str,
        question: str,
        schema_text: str,
    ) -> RepairOutcome:
        """
        Try to execute *initial_sql*; repair if it fails.

        Args:
            initial_sql:  First-pass generated SQL.
            db_id:        Target database identifier.
            question:     Original NL question (for repair prompt context).
            schema_text:  DDL string (for repair prompt context).

        Returns:
            RepairOutcome with the best SQL found and all attempt records.
        """
        attempts: List[RepairAttempt] = []
        current_sql = initial_sql
        schema_text = self._truncate_schema(schema_text)

        for attempt_num in range(1, self._max_attempts + 1):
            # Translate VI → EN trước khi execute.
            # LLM luôn gen SQL tiếng Việt; executor cần tên thật trong DB.
            translated = self._do_translate(current_sql, db_id)

            exec_result = self._executor.execute(translated, db_id)
            attempt = RepairAttempt(
                attempt_num=attempt_num,
                sql=current_sql,            # VI — dùng cho repair prompt
                translated_sql=translated,  # EN — đã execute
                exec_result=exec_result,
            )
            attempts.append(attempt)

            # QUAN TRỌNG: exec_result.success (property của ExecutionResult) coi
            # cả SUCCESS lẫn EMPTY là "thành công" — nhưng ở đây ta CHỦ ĐỘNG coi
            # EMPTY (chạy được, 0 dòng) là SAI, vì nhiều khả năng giá trị lọc
            # trong WHERE không khớp dữ liệu thực tế trong DB. Chỉ chấp nhận
            # ExecStatus.SUCCESS (có dữ liệu thật) là thành công thực sự.
            if self._is_real_success(exec_result):
                logger.debug(
                    "[SQLRepairer] %s: success on attempt %d/%d",
                    db_id, attempt_num, self._max_attempts,
                )
                return RepairOutcome(
                    final_sql=current_sql, success=True, attempts=attempts
                )

            if attempt_num == self._max_attempts:
                break

            if exec_result.status == ExecStatus.EMPTY:
                error_msg = (
                    "Câu lệnh thực thi THÀNH CÔNG về mặt cú pháp nhưng KHÔNG trả về "
                    "dòng nào (kết quả rỗng). Nhiều khả năng WHERE đang dùng SAI giá "
                    "trị so với dữ liệu thực tế trong CSDL — hãy kiểm tra lại giá trị "
                    "lọc (không phải cú pháp)."
                )
            else:
                error_msg = exec_result.short_error()
            logger.info(
                "[SQLRepairer] %s: attempt %d/%d failed — %s",
                db_id, attempt_num, self._max_attempts, error_msg,
            )

            # ── Bước value matching: chạy TRƯỚC GAT Corrector ──────────────
            # Thử fix literal values trong SQL (vd: 'nữ' → 'F') bằng cách:
            # 1. Extract string literals từ translated SQL
            # 2. Query DISTINCT values của cột tương ứng trong DB
            # 3. Nếu literal không khớp → dùng LLM map sang giá trị đúng
            # 4. Nếu fixed SQL execute được → return luôn, không cần LLM repair
            #
            # Nếu bước này KHÔNG tìm/sửa được giá trị nào (cột không tồn tại
            # ở bảng dự đoán, distinct values rỗng, hoặc LLM map thất bại) →
            # dump toàn bộ giá trị mẫu của các cột text trong CSDL, đưa vào
            # prompt GAT Corrector để LLM tự "rethink" và chọn giá trị đúng.
            value_hint_block = ""
            if self._db_dir:
                fixed_sql = self._fix_value_mismatch(
                    vi_sql=current_sql,
                    en_sql=translated,
                    db_id=db_id,
                )
                if fixed_sql and fixed_sql != translated:
                    fixed_result = self._executor.execute(fixed_sql, db_id)
                    if self._is_real_success(fixed_result):
                        logger.info(
                            "[SQLRepairer] %s: value matching fixed SQL on attempt %d",
                            db_id, attempt_num,
                        )
                        # Cập nhật VI SQL để trả về nhất quán
                        current_sql = fixed_sql
                        attempt.translated_sql = fixed_sql
                        attempt.exec_result = fixed_result
                        return RepairOutcome(
                            final_sql=current_sql, success=True, attempts=attempts
                        )
                    logger.debug(
                        "[SQLRepairer] %s: value matching attempted but still failed",
                        db_id,
                    )
                    # Vẫn sửa được literal nhưng chưa chạy được → vẫn cho thêm
                    # value dump để corrector có thêm lựa chọn khác nếu cần.
                    value_hint_block = self._dump_all_table_values(db_id)
                else:
                    # Không tìm/sửa được giá trị nào → dump toàn bộ giá trị bảng
                    logger.debug(
                        "[SQLRepairer] %s: no value fix found, falling back to "
                        "full table value dump for GAT Corrector",
                        db_id,
                    )
                    value_hint_block = self._dump_all_table_values(db_id)

            # prior_failed: attempts chưa THỰC SỰ thành công (loại cả EMPTY, không
            # chỉ lỗi cú pháp/runtime) — dùng _is_real_success thay vì .success
            prior_failed = [a for a in attempts if not self._is_real_success(a.exec_result)]
            # GAT Corrector: returns (system, user) tuple
            system_prompt, user_prompt = self._build_repair_prompt(
                failed_sql=current_sql,
                error=error_msg,
                question=question,
                schema=schema_text,
                prior_failed=prior_failed[:-1],
                value_hint_block=value_hint_block,
            )
            attempt.repair_prompt_snippet = user_prompt[:300]

            # llm.generate_with_system nếu LLMClient hỗ trợ,
            # fallback về generate(system + "\n\n" + user) nếu không
            if hasattr(self._llm, "generate_with_system"):
                raw_output = self._llm.generate_with_system(system_prompt, user_prompt)
            else:
                raw_output = self._llm.generate(system_prompt + "\n\n" + user_prompt)
            current_sql = self._parse_sql(raw_output)

        # All attempts failed – return last VI SQL
        logger.warning(
            "[SQLRepairer] %s: all %d attempts failed.\n  Last VI SQL: %s\n  Last EN SQL: %s",
            db_id, self._max_attempts, current_sql, attempts[-1].translated_sql,
        )
        return RepairOutcome(
            final_sql=current_sql, success=False, attempts=attempts
        )

    # ── Internals ─────────────────────────────────────────────────────────

    def _build_repair_prompt(
        self,
        failed_sql: str,
        error: str,
        question: str,
        schema: str,
        prior_failed: List[RepairAttempt],
        value_hint_block: str = "",
    ) -> tuple:
        """
        Returns (system_prompt, user_prompt) for GAT Corrector style.

        Two templates:
        - With error:    executor returned an error message → include it
        - Without error: SQL ran but result wrong (logic error) → no error line

        value_hint_block: khi non-empty, chèn thêm khối "Giá trị thực tế có
        trong CSDL" (dump toàn bộ giá trị mẫu các cột text) để LLM tự chọn
        lại giá trị đúng, dùng cho cả trường hợp lỗi lẫn kết quả rỗng.
        """
        if prior_failed:
            failure_lines = "\n".join(
                f"Lần {a.attempt_num}: {a.sql}"
                + (f"  -- lỗi: {a.exec_result.short_error()}" if a.exec_result.short_error() else "")
                for a in prior_failed
            )
            prev_block = self._PREV_FAILURE_TEMPLATE.format(failures=failure_lines)
        else:
            prev_block = ""

        value_block = (
            self._VALUE_DUMP_TEMPLATE.format(values=value_hint_block)
            if value_hint_block
            else ""
        )

        template = (
            self._CORRECTOR_TEMPLATE_WITH_ERROR if error
            else self._CORRECTOR_TEMPLATE_NO_ERROR
        )
        user_prompt = template.format(
            schema=schema,
            question=question,
            failed_sql=failed_sql,
            error=error,
            value_hint_block=value_block,
            prev_failures_block=prev_block,
        )
        return self._CORRECTOR_SYSTEM, user_prompt

    @staticmethod
    def _parse_sql(raw: str) -> str:
        """
        Normalise LLM output to a clean SQL string.

        Handles:
          - Markdown fences (```sql ... ```)
          - Leading/trailing whitespace
          - Duplicate SELECT prefix (e.g. "SELECT SELECT ...")
        """
        # Strip markdown fences
        raw = re.sub(r"```(?:sql)?", "", raw, flags=re.IGNORECASE).replace("```", "")
        sql = raw.strip().rstrip(";").strip()

        # If the model echoed "SELECT" twice (artifact of old SELECT-prefix prompts)
        if re.match(r"(?i)^select\s+select\b", sql):
            sql = sql[len("select"):].strip()

        return sql

    @staticmethod
    def _truncate_schema(schema_text: str, max_chars: int = _MAX_SCHEMA_CHARS) -> str:
        """Truncate schema DDL if it would overflow the context window."""
        if len(schema_text) <= max_chars:
            return schema_text
        truncated = schema_text[:max_chars]
        logger.debug(
            "[SQLRepairer] Schema truncated from %d to %d chars",
            len(schema_text), max_chars,
        )
        return truncated + "\n-- [schema truncated for length]"

    def _do_translate(self, vi_sql: str, db_id: str) -> str:
        """
        Translate VI SQL → EN SQL nếu translate_fn được inject.
        Fallback: trả nguyên vi_sql nếu không có translate_fn.
        """
        if self._translate is None:
            return vi_sql
        try:
            return self._translate(vi_sql, db_id)
        except Exception as exc:
            logger.warning(
                "[SQLRepairer] translate_fn failed for %s: %s. Using original SQL.",
                db_id, exc,
            )
            return vi_sql

    @staticmethod
    def _is_real_success(exec_result: ExecutionResult) -> bool:
        """
        True chỉ khi SQL chạy được VÀ có dữ liệu trả về (ExecStatus.SUCCESS).
        Khác với ExecutionResult.success (property gốc), ở đây ExecStatus.EMPTY
        KHÔNG được coi là thành công — chủ đích để trigger value-matching/repair
        khi kết quả rỗng, vì rỗng thường do sai giá trị lọc chứ không phải do
        câu hỏi thực sự không có đáp án.
        """
        return exec_result.status == ExecStatus.SUCCESS

    def _dump_all_table_values(
        self,
        db_id: str,
        max_values_per_col: Optional[int] = None,
        max_total_chars: Optional[int] = None,
    ) -> str:
        """
        Dump giá trị mẫu (distinct) của các cột dạng text trong TẤT CẢ các
        bảng của DB — dùng làm fallback khi:
          - _fix_value_mismatch không tìm/sửa được literal nào (cột không
            tồn tại ở bảng dự đoán, hoặc distinct values rỗng), hoặc
          - SQL chạy được nhưng trả về rỗng (ExecStatus.EMPTY) và không có
            literal nào để value-match (vd: SQL không có WHERE dạng đơn giản).

        Bị giới hạn bởi max_total_chars để không phá vỡ context window của
        repair prompt. Chỉ lấy cột kiểu CHAR/TEXT/CLOB — cột số ít khi cần
        "rethink" giá trị.
        """
        import os
        if not self._db_dir:
            return ""

        max_values_per_col = (
            max_values_per_col
            if max_values_per_col is not None
            else self._value_dump_max_values_per_col
        )
        max_total_chars = (
            max_total_chars
            if max_total_chars is not None
            else self._value_dump_max_chars
        )

        db_path = os.path.join(self._db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            return ""

        lines: List[str] = []
        used_chars = 0

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cursor.fetchall()]

            for table in tables:
                try:
                    cursor.execute(f'PRAGMA table_info("{table}")')
                    cols_info = cursor.fetchall()
                except sqlite3.OperationalError:
                    continue

                for col_info in cols_info:
                    col_name = col_info[1]
                    col_type = (col_info[2] or "").upper()
                    # Chỉ lấy cột dạng text/varchar/char
                    if col_type and not any(t in col_type for t in ("CHAR", "TEXT", "CLOB")):
                        continue

                    try:
                        cursor.execute(
                            f'SELECT DISTINCT "{col_name}" FROM "{table}" '
                            f'WHERE "{col_name}" IS NOT NULL LIMIT ?',
                            (max_values_per_col,),
                        )
                        rows = cursor.fetchall()
                    except sqlite3.OperationalError:
                        continue

                    if not rows:
                        continue

                    vals = ", ".join(f'"{r[0]}"' for r in rows)
                    line = f"{table}.{col_name}: {vals}"

                    if used_chars + len(line) > max_total_chars:
                        lines.append("-- [đã cắt bớt vì giới hạn độ dài]")
                        conn.close()
                        return "\n".join(lines)

                    lines.append(line)
                    used_chars += len(line)

            conn.close()
        except Exception as exc:
            logger.debug(
                "[SQLRepairer] Failed to dump table values for %s: %s", db_id, exc
            )
            return ""

        return "\n".join(lines)

    # ── Value Matching ─────────────────────────────────────────────────────

    def _fix_value_mismatch(
        self,
        vi_sql: str,
        en_sql: str,
        db_id: str,
    ) -> Optional[str]:
        """
        Cố gắng sửa lỗi value mismatch trong EN SQL bằng cách:
          1. Extract cặp (column, literal) từ WHERE clause của EN SQL.
          2. Query DISTINCT values của column đó trong DB.
          3. Nếu literal không có trong DISTINCT → dùng LLM map sang giá trị đúng.
          4. Replace literal trong EN SQL và trả về SQL đã sửa.

        Returns None nếu không tìm được mapping nào.
        """
        pairs = self._extract_literals_with_columns(en_sql)
        if not pairs:
            return None

        fixed_sql = en_sql
        changed = False

        for col, literal in pairs:
            distinct_vals = self._get_distinct_values(db_id, col)
            if not distinct_vals:
                continue

            # Nếu literal đã có trong distinct values (case-insensitive) → bỏ qua
            if any(literal.lower() == v.lower() for v in distinct_vals):
                continue

            # LLM map literal → giá trị đúng trong distinct_vals
            mapped = self._llm_map_value(literal, distinct_vals)
            if mapped and mapped != literal:
                # Replace trong SQL (bảo vệ quote)
                fixed_sql = fixed_sql.replace(f"'{literal}'", f"'{mapped}'")
                fixed_sql = fixed_sql.replace(f'"{literal}"', f'"{mapped}"')
                logger.info(
                    "[ValueMatching] '%s' → '%s' (col: %s)",
                    literal, mapped, col,
                )
                changed = True

        return fixed_sql if changed else None

    @staticmethod
    def _extract_literals_with_columns(sql: str) -> List[Tuple[str, str]]:
        """
        Extract cặp (column_name, string_literal) từ WHERE clause.
        Chỉ xét dạng: column = 'value' hoặc column = "value"
        hoặc column != / LIKE / IN (...) dạng đơn giản.

        Ví dụ:
            WHERE sex = 'nữ'        → [('sex', 'nữ')]
            WHERE country = 'Hoa Kỳ' → [('country', 'Hoa Kỳ')]
        """
        pairs: List[Tuple[str, str]] = []
        # Match: identifier = 'value' hoặc identifier = "value"
        pattern = re.compile(
            r'\b([\w.]+)\s*(?:=|!=|LIKE)\s*[\'"]([^\'"]+)[\'"]',
            re.IGNORECASE,
        )
        for m in pattern.finditer(sql):
            col_full = m.group(1)
            # Bỏ table alias prefix (t1.col → col)
            col = col_full.split(".")[-1]
            literal = m.group(2)
            pairs.append((col, literal))
        return pairs

    def _get_distinct_values(self, db_id: str, column: str) -> List[str]:
        """
        Query DISTINCT values của một column trong DB.
        Thử tìm column name trên tất cả các bảng trong DB.
        Trả về list string (tối đa 50 giá trị để tránh prompt quá dài).

        LƯU Ý QUAN TRỌNG (bug đã fix): SQLite có quirk tương thích ngược —
        khi định danh trong DẤU NHÁY KÉP không khớp tên cột nào, thay vì báo
        lỗi "no such column", nó âm thầm coi định danh đó như một STRING
        LITERAL. Vd `SELECT DISTINCT "gioi_tinh" FROM "mon_hoc"` khi bảng
        mon_hoc KHÔNG có cột gioi_tinh sẽ KHÔNG raise OperationalError, mà
        trả về đúng 1 dòng: ('gioi_tinh',) — biến tên cột thành "giá trị".
        Nếu bảng này được duyệt trước bảng thật sự có cột đó, hàm sẽ trả về
        sai (['gioi_tinh'] thay vì ['F', 'M']) mà không hề báo lỗi.
        Cách fix: kiểm tra cột có THẬT SỰ tồn tại qua PRAGMA table_info
        trước khi query, thay vì dựa vào except OperationalError.
        """
        import os
        if not self._db_dir:
            return []

        db_path = os.path.join(self._db_dir, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            return []

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Lấy danh sách bảng
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [r[0] for r in cursor.fetchall()]

            for table in tables:
                # Kiểm tra cột có thực sự tồn tại trong bảng này không,
                # TRƯỚC khi query — tránh SQLite quirk coi "column" không
                # tồn tại như string literal thay vì raise lỗi.
                try:
                    cursor.execute(f'PRAGMA table_info("{table}")')
                    real_cols = {row[1].lower() for row in cursor.fetchall()}
                except sqlite3.OperationalError:
                    continue
                if column.lower() not in real_cols:
                    continue

                try:
                    cursor.execute(
                        f'SELECT DISTINCT "{column}" FROM "{table}" '
                        f'WHERE "{column}" IS NOT NULL LIMIT 50'
                    )
                    rows = cursor.fetchall()
                    if rows:
                        conn.close()
                        return [str(r[0]) for r in rows]
                except sqlite3.OperationalError:
                    # Trường hợp hiếm khác (vd tên bảng/cột chứa ký tự đặc biệt)
                    continue

            conn.close()
        except Exception as exc:
            logger.debug("[ValueMatching] DB query failed for %s.%s: %s", db_id, column, exc)

        return []

    def _llm_map_value(self, literal: str, distinct_vals: List[str]) -> Optional[str]:
        """
        Dùng LLM để map một literal sang giá trị đúng trong distinct_vals.
        Prompt ngắn gọn, yêu cầu chỉ trả về giá trị, không giải thích.
        """
        vals_str = ", ".join(f"'{v}'" for v in distinct_vals[:20])
        prompt = (
            f"Cho danh sách giá trị THỰC TẾ có trong cơ sở dữ liệu cho cột này: [{vals_str}]\n"
            f"Giá trị người dùng nhập: '{literal}'\n"
            f"Hãy tìm giá trị trong danh sách trên tương ứng với nghĩa của '{literal}' "
            f"(có thể là bản dịch, viết tắt, hoặc cách viết khác của cùng một khái niệm).\n"
            f"BẮT BUỘC chỉ trả về CHÍNH XÁC MỘT giá trị lấy nguyên văn từ danh sách trên, "
            f"không giải thích, không thêm dấu ngoặc kép hay ký tự khác.\n"
            f"Chỉ trả về NONE nếu bạn chắc chắn không có giá trị nào trong danh sách liên quan "
            f"đến '{literal}'."
        )
        try:
            if hasattr(self._llm, "generate_with_system"):
                raw = self._llm.generate_with_system(
                    "Bạn là trợ lý ánh xạ giá trị ngôn ngữ tự nhiên sang giá trị cơ sở dữ liệu.",
                    prompt,
                )
            else:
                raw = self._llm.generate(prompt)

            mapped = raw.strip().strip("'\"")
            if mapped.upper() == "NONE" or not mapped:
                return None
            # Kiểm tra mapped có thực sự nằm trong distinct_vals không
            for v in distinct_vals:
                if v.lower() == mapped.lower():
                    return v  # trả về đúng case từ DB
            return None
        except Exception as exc:
            logger.debug("[ValueMatching] LLM map failed: %s", exc)
            return None