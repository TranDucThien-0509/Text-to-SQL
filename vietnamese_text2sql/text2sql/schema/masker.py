"""
masker.py – Sinh Masked Question theo chuẩn DAIL-SQL (DAIL Selection).

DAIL-SQL chỉ cần MỘT thứ từ schema linking: vị trí các token trong câu hỏi
trùng với tên cột / tên bảng / giá trị cell / số / ngày → thay bằng một
mask token chung (mặc định "_"). Không cần giữ score, match_type chi tiết
(CEM/CPM/TEM/TPM/cell/num_date) ở bước này — những nhãn đó chỉ hữu ích cho
GNN schema linking, không phải cho việc tính độ tương đồng giữa các câu hỏi
trong DAIL Selection.

    from text2sql.schema.schema_linker import SchemaLinker
    from text2sql.schema.masker import mask_question

    linker = SchemaLinker()
    result = linker.link(question, schema, cell_retriever)
    masked = mask_question(question, result)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from text2sql.schema.schema_linker import (
    SchemaLinker,
    SchemaLinkingResult,
    _NUM_PAT,
    _DATE_PAT,
)


def mask_question(
    question: str,
    result: SchemaLinkingResult,
    mask_tag: str = "_",
    collapse_consecutive: bool = True,
) -> str:
    """
    Sinh masked question từ kết quả SchemaLinker.link().

    Args:
        question: câu hỏi gốc (chỉ dùng để re-tokenize cho đồng bộ).
        result: SchemaLinkingResult trả về từ SchemaLinker.link().
        mask_tag: token dùng để thay thế (DAIL-SQL paper dùng "_").
        collapse_consecutive: nếu True, các mask liền kề được gộp thành
            1 mask_tag duy nhất (đúng hành vi gốc của DAIL-SQL, tránh
            "_ _ _" lặp lại làm sai lệch độ tương đồng).

    Returns:
        Chuỗi câu hỏi đã tokenize + mask, ví dụ:
            "_ sinh_viên có _ là bao_nhiêu"
    """
    tokens = SchemaLinker._tokenize(question)
    n = len(tokens)
    if n == 0:
        return ""

    mask_flags = [False] * n

    def _try_mask(span_text: str) -> None:
        if not span_text:
            return
        span_toks = span_text.split()
        L = len(span_toks)
        if L == 0 or L > n:
            return
        i = 0
        while i <= n - L:
            if (
                not any(mask_flags[i : i + L])
                and tokens[i : i + L] == span_toks
            ):
                for k in range(L):
                    mask_flags[i + k] = True
                i += L
            else:
                i += 1

    # 1) Column matches (exact + partial)
    spans: List[str] = []
    for m in result.q_col_match:
        if m.matched_span:
            spans.append(m.matched_span)

    # 2) Table matches
    for m in result.q_tab_match:
        if m.matched_span:
            spans.append(m.matched_span)

    # 3) Cell value matches
    for m in result.cell_match:
        if m.matched_span:
            spans.append(m.matched_span.lower())

    # Mask theo span dài nhất trước để tránh mask nhầm subset của 1 cụm dài
    for span in sorted(set(spans), key=lambda s: -len(s.split())):
        _try_mask(span)

    # 4) Số / ngày — match trực tiếp trên từng token (không qua matched_span
    #    vì _match_num_date không lưu span gốc)
    for i, tok in enumerate(tokens):
        if mask_flags[i]:
            continue
        if _NUM_PAT.fullmatch(tok) or _DATE_PAT.fullmatch(tok):
            mask_flags[i] = True

    # Build output
    out_tokens: List[str] = []
    prev_masked = False
    for tok, flagged in zip(tokens, mask_flags):
        if flagged:
            if collapse_consecutive and prev_masked:
                continue
            out_tokens.append(mask_tag)
        else:
            out_tokens.append(tok)
        prev_masked = flagged

    return " ".join(out_tokens)


def mask_question_typed(
    question: str,
    result: SchemaLinkingResult,
    tag_col: str = "COLUMN",
    tag_tab: str = "TABLE",
    tag_cell: str = "VALUE",
    tag_num: str = "NUM",
    tag_date: str = "DATE",
    collapse_consecutive: bool = True,
) -> str:
    """
    Biến thể của mask_question() — che theo LOẠI thực thể (cột/bảng/giá trị/
    số/ngày) bằng tag riêng, thay vì 1 tag "_" chung cho tất cả.

    KHÔNG dùng cho DAIL-Selection gốc (dùng mask_question() ở trên — DAILSQL chỉ cần 1 mask token, không cần phân loại, xem docstring đầu file).
    Hàm này dùng cho các use case cần giữ tín hiệu cấu trúc chi tiết hơn,
    ví dụ: tính similarity giữa câu hỏi để chọn few-shot example theo đúng
    "khung cấu trúc" (bảng/cột nào, không chỉ có mặt cột/bảng hay không).

    Ví dụ:
        "Hiển_thị tên của các thành_viên đến từ ' Hoa_Kỳ ' hoặc ' Canada ' ."
      → "Hiển_thị COLUMN của các TABLE đến từ VALUE hoặc VALUE"

    Lưu ý: SchemaLinker._tokenize() bóc hết dấu câu (string.punctuation)
    trước khi tokenize, nên dấu nháy ' ' và dấu . cuối câu KHÔNG còn trong
    output — đây là hành vi có sẵn của _tokenize(), không phải lỗi riêng
    của hàm này.

    Args:
        question: câu hỏi gốc (chỉ dùng để re-tokenize cho đồng bộ).
        result: SchemaLinkingResult trả về từ SchemaLinker.link().
        tag_col/tag_tab/tag_cell/tag_num/tag_date: tag dùng cho mỗi loại.
        collapse_consecutive: nếu True, các token liên tiếp CÙNG loại được
            gộp thành 1 tag duy nhất (token liên tiếp nhưng KHÁC loại thì
            không gộp, để giữ ranh giới giữa các thực thể khác nhau).

    Returns:
        Chuỗi câu hỏi đã tokenize + mask theo loại, ví dụ:
            "COLUMN của TABLE có VALUE"
    """
    tokens = SchemaLinker._tokenize(question)
    n = len(tokens)
    if n == 0:
        return ""

    # None = chưa mask; ngược lại là tag của loại đã mask token đó
    token_tags: List[Optional[str]] = [None] * n

    def _try_mask(span_text: str, tag: str) -> None:
        if not span_text:
            return
        span_toks = span_text.split()
        L = len(span_toks)
        if L == 0 or L > n:
            return
        i = 0
        while i <= n - L:
            if (
                all(token_tags[k] is None for k in range(i, i + L))
                and tokens[i : i + L] == span_toks
            ):
                for k in range(L):
                    token_tags[i + k] = tag
                i += L
            else:
                i += 1

    # Gom span kèm tag loại — KHÔNG dùng set() thô trên toàn bộ vì cần giữ
    # tag đi kèm; dedup theo (span, tag) là đủ vì cùng span cùng tag chỉ
    # cần thử match 1 lần.
    spans_typed: List[Tuple[str, str]] = []
    for m in result.q_col_match:
        if m.matched_span:
            spans_typed.append((m.matched_span, tag_col))
    for m in result.q_tab_match:
        if m.matched_span:
            spans_typed.append((m.matched_span, tag_tab))
    for m in result.cell_match:
        if m.matched_span:
            spans_typed.append((m.matched_span.lower(), tag_cell))

    # Mask span dài nhất trước, giống mask_question gốc, để tránh mask
    # nhầm subset của 1 cụm dài hơn đã được nhận diện đúng.
    for span, tag in sorted(set(spans_typed), key=lambda st: -len(st[0].split())):
        _try_mask(span, tag)

    # Số / ngày — match trực tiếp trên từng token còn trống (không qua
    # matched_span vì _match_num_date không lưu span gốc).
    for i, tok in enumerate(tokens):
        if token_tags[i] is not None:
            continue
        if _NUM_PAT.fullmatch(tok):
            token_tags[i] = tag_num
        elif _DATE_PAT.fullmatch(tok):
            token_tags[i] = tag_date

    # Build output
    out_tokens: List[str] = []
    prev_tag: Optional[str] = None
    for tok, tag in zip(tokens, token_tags):
        if tag is not None:
            if collapse_consecutive and tag == prev_tag:
                continue
            out_tokens.append(tag)
        else:
            out_tokens.append(tok)
        prev_tag = tag

    return " ".join(out_tokens)


def mask_for_dail_sql(
    question: str,
    schema,
    cell_retriever: Optional["CellValueRetriever"] = None,  # type: ignore[name-defined]
    linker: Optional[SchemaLinker] = None,
    mask_tag: str = "_",
) -> str:
    """
    Hàm tiện ích: chạy SchemaLinker.link() + mask_question() trong 1 lần gọi.
    Dùng khi bạn chưa cần giữ lại SchemaLinkingResult chi tiết cho việc khác.
    """
    linker = linker or SchemaLinker()
    result = linker.link(question, schema, cell_retriever)
    return mask_question(question, result, mask_tag=mask_tag)