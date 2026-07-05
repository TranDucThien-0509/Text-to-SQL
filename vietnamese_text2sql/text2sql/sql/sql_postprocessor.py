"""
SQLPostProcessor – post-process LLM output trước khi evaluate/execute.

Các vấn đề được fix:
  1. Strip markdown artifacts: ```sql ... ``` → clean SQL
  2. Repetition detection: cắt SQL bị lặp vô tận
  3. Quote normalization: single quote → double quote trong WHERE values
  4. Bare string values: female → "female", Grondzeiler → "Grondzeiler"
  5. Comma spacing: 't1.tên, t1.quốc_tịch' → 't1.tên , t1.quốc_tịch'
  6. Normalize whitespace

Không fix được ở postprocessor (lỗi logic LLM):
  - Sai tên cột (quốc_gia vs quốc_tịch)
  - Sai thứ tự ON clause (t2.col = t1.col thay vì t1.col = t2.col)
  - Sai approach (ORDER BY LIMIT 1 thay vì MAX())
"""
from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)

# ── Compiled patterns ────────────────────────────────────────────────────────

# Strip markdown code fences
_MARKDOWN_PAT = re.compile(r"```(?:sql)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

# Detect repetition loop: AND <table>.<col> IN ( SELECT ... xuất hiện >= 3 lần
_REPEAT_IN_SELECT_PAT = re.compile(r'\band\s+\w+[\w.]*\s+in\s*\(\s*select\b', re.IGNORECASE)

# Single-quoted string literals → double-quoted
# Chỉ match 'value' không phải là SQL identifier (không có dấu chấm trước)
_SINGLE_QUOTE_PAT = re.compile(r"'([^']*)'")

# Bare unquoted string value sau = hoặc != hoặc <>
# Match: = female, = Grondzeiler, != male
# Không match: số, keyword SQL, column reference (có dấu chấm)
_BARE_VALUE_PAT = re.compile(
    r"(?<=[=!<>])\s*"                                           # sau toán tử so sánh
    r"([A-Za-z\u00C0-\u024F\u1E00-\u1EFF]"                     # bắt đầu bằng chữ cái (kể cả Unicode)
    r"[A-Za-z\u00C0-\u024F\u1E00-\u1EFF0-9_]*)"                # phần còn lại (không có dấu cách)
    r"(?=\s*(?:and\b|or\b|order\b|group\b|having\b|limit\b|\)|$))",  # theo sau là keyword/end
    re.IGNORECASE | re.UNICODE,
)

# SQL keywords – không quote những từ này dù match pattern trên
_SQL_KEYWORDS = frozenset({
    "select", "from", "where", "and", "or", "not", "in", "is", "null",
    "join", "left", "right", "inner", "outer", "cross", "on", "as",
    "group", "by", "order", "having", "limit", "distinct", "union", "all",
    "count", "sum", "avg", "max", "min", "exists", "between", "like",
    "asc", "desc", "offset", "true", "false", "case", "when", "then",
    "else", "end", "insert", "update", "delete", "set", "values", "into",
})


class SQLPostProcessor:
    """
    Stateless post-processor cho LLM-generated SQL.
    Gọi SQLPostProcessor.process(sql) trước khi evaluate.
    """

    @classmethod
    def process(cls, sql: str) -> str:
        """
        Full post-processing pipeline theo thứ tự:
          1. strip_markdown
          2. cut_repetition
          3. normalize_quotes      (single → double)
          4. fix_bare_values       (female → "female")
          5. normalize_comma_spacing
          6. normalize_whitespace
        """
        sql = sql.strip()
        sql = cls.strip_markdown(sql)
        sql = cls.cut_repetition(sql)
        sql = cls.normalize_quotes(sql)
        sql = cls.fix_bare_values(sql)
        sql = cls.normalize_comma_spacing(sql)
        sql = cls.normalize_whitespace(sql)
        return sql

    # ── Step 1 ────────────────────────────────────────────────────────────

    @classmethod
    def strip_markdown(cls, sql: str) -> str:
        """Remove ```sql ... ``` fences nếu có."""
        m = _MARKDOWN_PAT.search(sql)
        if m:
            sql = m.group(1)
            logger.debug("[PostProcessor] Stripped markdown fences.")
        return sql.strip()

    # ── Step 2 ────────────────────────────────────────────────────────────

    @classmethod
    def cut_repetition(cls, sql: str) -> str:
        """
        Phát hiện và cắt SQL bị lặp vô tận.
        Heuristic: 'AND <table>.<col> IN (SELECT ...)' xuất hiện >= 3 lần
        → cắt từ lần lặp thứ 2 trở đi và đóng ngoặc còn mở.
        """
        matches = list(_REPEAT_IN_SELECT_PAT.finditer(sql))
        if len(matches) >= 3:
            cut_pos = matches[1].start()
            sql_cut = sql[:cut_pos].rstrip()
            open_count = sql_cut.count('(') - sql_cut.count(')')
            sql_cut += ')' * max(0, open_count)
            logger.warning(
                "[PostProcessor] Detected repetition loop (%d matches). Truncated SQL.",
                len(matches),
            )
            return sql_cut
        return sql

    # ── Step 3 ────────────────────────────────────────────────────────────

    @classmethod
    def normalize_quotes(cls, sql: str) -> str:
        """
        Chuyển single-quoted string literals → double-quoted.
        'female' → "female",  'Grondzeiler' → "Grondzeiler"

        Không escape double quote bên trong vì trong SQL context
        double quote là identifier delimiter, không phải escape sequence.
        Giữ nguyên case của value.
        """
        def _to_double(m: re.Match) -> str:
            return f'"{m.group(1)}"'

        original = sql
        sql = _SINGLE_QUOTE_PAT.sub(_to_double, sql)
        if sql != original:
            logger.debug("[PostProcessor] Normalized single quotes → double quotes.")
        return sql

    # ── Step 4 ────────────────────────────────────────────────────────────

    @classmethod
    def fix_bare_values(cls, sql: str) -> str:
        """
        Thêm double quotes cho bare (unquoted) string values trong WHERE.
        Ví dụ:
          where giới_tính = female       → where giới_tính = "female"
          where loại = Grondzeiler       → where loại = "Grondzeiler"
          where quốc_gia = American      → where quốc_gia = "American"

        Skip: SQL keywords, số nguyên/thực, giá trị đã có quotes.
        """
        def _maybe_quote(m: re.Match) -> str:
            val = m.group(1)
            if val.lower() in _SQL_KEYWORDS:
                return m.group(0)  # giữ nguyên keyword
            # Thay thế đúng vị trí: giữ khoảng trắng sau toán tử
            full = m.group(0)
            return full.replace(val, f'"{val}"', 1)

        original = sql
        sql = _BARE_VALUE_PAT.sub(_maybe_quote, sql)
        if sql != original:
            logger.debug("[PostProcessor] Fixed bare string values → double-quoted.")
        return sql

    # ── Step 5 ────────────────────────────────────────────────────────────

    @classmethod
    def normalize_comma_spacing(cls, sql: str) -> str:
        """
        Chuẩn hóa spacing quanh dấu phẩy: 'a, b' và 'a ,b' → 'a , b'.
        Đây là format chuẩn của gold SQL trong dataset này.
        """
        return re.sub(r'\s*,\s*', ' , ', sql)

    # ── Step 6 ────────────────────────────────────────────────────────────

    @classmethod
    def normalize_whitespace(cls, sql: str) -> str:
        """Collapse multiple spaces, strip leading/trailing."""
        return re.sub(r'\s+', ' ', sql).strip()


# ── Convenience function ──────────────────────────────────────────────────────

def post_process_sql(sql: str) -> str:
    """Shortcut: SQLPostProcessor.process(sql)"""
    return SQLPostProcessor.process(sql)


# ── Tests ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        # ── Quote normalization ──────────────────────────────────────────
        ("select * from t where giới_tính = 'female'",
         'select * from t where giới_tính = "female"'),

        ("select * from t where loại = 'Grondzeiler'",
         'select * from t where loại = "Grondzeiler"'),

        # ── Bare values (từ results.json) ────────────────────────────────
        ("select count ( * ) from kiến_trúc_sư where giới_tính = female",
         'select count ( * ) from kiến_trúc_sư where giới_tính = "female"'),

        ("select tên , quốc_tịch , id from kiến_trúc_sư where giới_tính = male order by tên",
         'select tên , quốc_tịch , id from kiến_trúc_sư where giới_tính = "male" order by tên'),

        ("select tên , năm xây_dựng from nhà_máy where loại = Grondzeiler",
         'select tên , năm xây_dựng from nhà_máy where loại = "Grondzeiler"'),

        # ── Đã đúng rồi → không được làm hỏng ───────────────────────────
        ('select * from t where giới_tính = "female"',
         'select * from t where giới_tính = "female"'),

        # ── Số → không quote ─────────────────────────────────────────────
        ("select * from t where id = 5",
         "select * from t where id = 5"),

        # ── Comma spacing ────────────────────────────────────────────────
        ("select distinct t1.tên, t1.quốc_tịch from kiến_trúc_sư as t1 join nhà_máy as t2 on t1.id = t2.id kiến_trúc_sư",
         "select distinct t1.tên , t1.quốc_tịch from kiến_trúc_sư as t1 join nhà_máy as t2 on t1.id = t2.id kiến_trúc_sư"),

        ("select t1.id, t1.tên from kiến_trúc_sư as t1 join cầu as t2 on t1.id = t2.id kiến_trúc_sư group by t1.id having count ( * ) >= 3",
         "select t1.id , t1.tên from kiến_trúc_sư as t1 join cầu as t2 on t1.id = t2.id kiến_trúc_sư group by t1.id having count ( * ) >= 3"),

        # ── Markdown strip ───────────────────────────────────────────────
        ("```sql\nselect count ( * ) from singer\n```",
         "select count ( * ) from singer"),

        # ── Repetition detection (no-crash) ──────────────────────────────
        ("select a from t where a in (select b from c) "
         "and t.id in (select id from c) "
         "and t.id in (select id from c) "
         "and t.id in (select id from c)",
         None),

        # ── OR với multiple bare values ───────────────────────────────────
        ("select distinct loại from nhà_máy join kiến_trúc_sư on nhà_máy.id kiến_trúc_sư = kiến_trúc_sư.id "
         "where quốc_gia = American or quốc_gia = Canadian",
         'select distinct loại from nhà_máy join kiến_trúc_sư on nhà_máy.id kiến_trúc_sư = kiến_trúc_sư.id '
         'where quốc_gia = "American" or quốc_gia = "Canadian"'),
    ]

    print("=== SQLPostProcessor Tests ===\n")
    all_pass = True
    for i, (inp, expected) in enumerate(test_cases):
        result = SQLPostProcessor.process(inp)
        if expected is not None:
            ok = result == expected
            status = "✓" if ok else "✗"
            if not ok:
                all_pass = False
                print(f"[{status}] Test {i+1} FAILED")
                print(f"  Input:    {inp}")
                print(f"  Expected: {expected}")
                print(f"  Got:      {result}")
            else:
                print(f"[{status}] Test {i+1} passed")
        else:
            print(f"[~] Test {i+1} (no-crash): {result[:80]}...")

    print()
    print("All tests passed!" if all_pass else "Some tests FAILED.")