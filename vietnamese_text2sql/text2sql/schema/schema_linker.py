"""
SchemaLinker – identifies which tables and columns are relevant to a question.

Ported from DAIL-SQL / IRNet spider_match_utils.compute_schema_linking logic.
Implements five match types:
  q_col_match    – question tokens ↔ column names   (exact / partial via n-gram,
                   hoặc semantic nếu không khớp string nhưng đồng nghĩa,
                   ví dụ: "quê hương" ↔ cột "quê_quán")
  q_tab_match    – question tokens ↔ table names    (exact / partial via n-gram,
                   hoặc semantic tương tự)
  cell_match     – question tokens ↔ actual DB cell values  (via CellValueRetriever)
  num_date_match – numeric / date literals in the question

Semantic matching (embedding-based) được dùng làm bước bổ sung, chạy SONG SONG
với n-gram matching cho TẤT CẢ cột/bảng, sau đó lấy max(score n-gram, score
semantic) cho mỗi cột/bảng — không chỉ fallback cho phần chưa match được.
"""
from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np
from pyvi import ViTokenizer

from text2sql.schema.schema_processor import DatabaseSchema
from text2sql.schema.value_hint_builder import build_value_hints, format_hints_for_prompt, ValueHint

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Semantic (embedding) matching config
# ─────────────────────────────────────────────────────────────
_DEFAULT_EMBED_MODEL = "moka-ai/m3e-base"
_DEFAULT_SEMANTIC_THRESHOLD = 0.55

_embed_model_cache: dict = {}  # model_name -> SentenceTransformer instance


def _get_embed_model(model_name: str) -> "SentenceTransformer":  # type: ignore[name-defined]
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers chưa được cài. Chạy: "
            "pip install sentence-transformers --break-system-packages"
        )
    if model_name not in _embed_model_cache:
        logger.info("[SchemaLinker] Loading embedding model: %s", model_name)
        _embed_model_cache[model_name] = SentenceTransformer(model_name)
    return _embed_model_cache[model_name]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)

# ─────────────────────────────────────────────────────────────
# Stop words (Vietnamese + English)
# ─────────────────────────────────────────────────────────────
_VI_STOPWORDS: Set[str] = {
    "là", "của", "các", "những", "một", "có", "cho", "và", "hoặc", "nhưng",
    "không", "nào", "gì", "ai", "hãy", "liệt_kê", "hiển_thị", "tìm", "đưa",
    "trả_về", "lấy", "cho_biết", "tôi", "này", "kia", "rằng", "thì", "mà",
    "để", "với", "tại", "từ", "đến", "trong", "ngoài", "dưới", "trên",
    "khi", "lúc", "đang", "đã", "sẽ", "nhé", "nha", "ạ", "liệt", "kê",
}

_PUNKS: Set[str] = set(string.punctuation)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

# Number / date patterns
_NUM_PAT  = re.compile(r"\b\d+([.,]\d+)?\b")
_DATE_PAT = re.compile(
    r"\b\d{4}[-/]\d{2}[-/]\d{2}\b"
    r"|\b\d{2}/\d{2}/\d{4}\b"
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
    match_type: str        # "exact" | "partial" | "semantic" | "cell" | "num_date"
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
    q_col_match: List[ColumnMatch]   = field(default_factory=list)
    q_tab_match: List[TableMatch]    = field(default_factory=list)
    cell_match:  List[ColumnMatch]   = field(default_factory=list)
    num_date_match: List[ColumnMatch] = field(default_factory=list)
    # table.column -> ValueHint (giá trị thật trong DB, khớp nhất với câu hỏi).
    # Được build ngay trong link() nếu cell_retriever được truyền vào, dùng
    # chung 1 lần truy vấn DB cho cả cell_match lẫn value_hints.
    value_hints: dict = field(default_factory=dict)  # Dict[str, ValueHint]

    @property
    def prompt_value_hint_block(self) -> str:
        """Text block sẵn sàng nhét vào prompt — xem value_hint_builder.format_hints_for_prompt."""
        return format_hints_for_prompt(self.value_hints)

    @property
    def relevant_table_indices(self) -> Set[int]:
        indices: Set[int] = set()
        for m in self.q_tab_match:
            indices.add(m.table_idx)
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
                "col":     m.col_name,
                "table":   m.table_name,
                "type":    m.match_type,
                "score":   round(m.score, 4),
                "span":    m.matched_span,
            }

        def _tab(m: TableMatch) -> dict:
            return {
                "table_idx": m.table_idx,
                "table":     m.table_name,
                "type":      m.match_type,
                "score":     round(m.score, 4),
                "span":      m.matched_span,
            }

        return {
            "db_id":         self.db_id,
            "question":      self.question,
            "q_col_match":   [_col(m) for m in self.q_col_match],
            "q_tab_match":   [_tab(m) for m in self.q_tab_match],
            "cell_match":    [_col(m) for m in self.cell_match],
            "num_date_match":[_col(m) for m in self.num_date_match],
        }


# ─────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────

class SchemaLinker:
    """
    Reimplemented theo logic gốc của DAIL-SQL spider_match_utils.
    Dùng n-gram (tối đa 5) trên token list, so sánh exact / partial
    với word-boundary regex — không phụ thuộc vào ViTokenizer cho schema name.
    """

    def __init__(
        self,
        max_ngram: int = 5,
        use_semantic: bool = True,
        semantic_threshold: float = _DEFAULT_SEMANTIC_THRESHOLD,
        embed_model_name: str = _DEFAULT_EMBED_MODEL,
    ) -> None:
        self._max_ngram = max_ngram
        self._use_semantic = use_semantic
        self._semantic_threshold = semantic_threshold
        self._embed_model_name = embed_model_name

    # ── Public entry point ────────────────────────────────────

    def link(
        self,
        question: str,
        schema: DatabaseSchema,
        cell_retriever: Optional["CellValueRetriever"] = None,  # type: ignore[name-defined]
    ) -> SchemaLinkingResult:
        result = SchemaLinkingResult(db_id=schema.db_id, question=question)

        # Tokenize câu hỏi → list[str]
        q_tokens = self._tokenize(question)

        if not q_tokens:
            return result

        # Chuẩn bị col/tab token lists theo đúng cách DAIL-SQL làm
        # (tách _ thành space, lowercase, không qua ViTokenizer)
        col_token_lists = self._prepare_schema_token_lists(schema)
        tab_token_lists = self._prepare_table_token_lists(schema)

        # Step 1 & 2 – schema linking (port từ compute_schema_linking)
        raw_col, raw_tab = self._compute_schema_linking(
            q_tokens, col_token_lists, tab_token_lists
        )

        # Step 1b – semantic (embedding) matching, chạy cho TẤT CẢ cột/bảng
        # song song với n-gram, dùng để bắt các trường hợp đồng nghĩa mà
        # string-matching bỏ lỡ (ví dụ: "quê hương" ↔ cột "quê_quán").
        sem_col, sem_tab = self._compute_semantic_matches(question, schema)

        # Chuyển raw dict → ColumnMatch / TableMatch (dedup, lấy best per col/tab,
        # kết hợp với semantic score theo nguyên tắc max(score n-gram, score semantic))
        result.q_col_match = self._build_col_matches(raw_col, schema, sem_col)
        result.q_tab_match = self._build_tab_matches(raw_tab, schema, sem_tab)

        # Step 3 – cell value matching
        if cell_retriever is not None:
            result.cell_match = self._match_cells(schema, question, cell_retriever)

            # Step 3b – truy cập DB để lấy distinct values + chọn giá trị khớp
            # nhất cho các cột đã được link (q_col_match ∪ cell_match), sẵn
            # sàng đưa vào prompt để LLM dùng đúng literal thật trong DB thay
            # vì tự chép nguyên văn từ câu hỏi (có thể đã bị hỏng bởi word
            # segmentation, ví dụ "Elaine_Lee" thay vì "Elaine Lee").
            linked_cols = {
                (m.table_name, m.col_name)
                for m in (*result.q_col_match, *result.cell_match)
            }
            result.value_hints = build_value_hints(
                question=question,
                db_id=schema.db_id,
                linked_columns=[{"table": t, "column": c} for t, c in linked_cols],
                cell_retriever=cell_retriever,
            )

        # Step 4 – num / date matching
        result.num_date_match = self._match_num_date(schema, question)

        logger.debug(
            "[SchemaLinker] %s: %d col | %d tab | %d cell | %d num/date",
            schema.db_id,
            len(result.q_col_match), len(result.q_tab_match),
            len(result.cell_match),  len(result.num_date_match),
        )
        return result

    # ── Schema linking core (port từ spider_match_utils) ─────

    def _compute_schema_linking(
        self,
        question: List[str],
        col_token_lists: dict,   # col_global_idx -> List[str]
        tab_token_lists: dict,   # tab_idx        -> List[str]
    ) -> Tuple[dict, dict]:
        """
        Port trực tiếp từ compute_schema_linking().
        q_col_match / q_tab_match: key = "q_id,schema_id", value = "CEM"/"CPM"/"TEM"/"TPM"
        """

        def partial_match(x_list: List[str], y_list: List[str]) -> bool:
            x_str = " ".join(x_list)
            y_str = " ".join(y_list)
            if not x_str or x_str in _VI_STOPWORDS or x_str in _PUNKS:
                return False
            return bool(re.search(rf"\b{re.escape(x_str)}\b", y_str))

        def exact_match(x_list: List[str], y_list: List[str]) -> bool:
            return " ".join(x_list) == " ".join(y_list)

        q_col_match: dict = {}
        q_tab_match: dict = {}

        n = self._max_ngram
        while n > 0:
            for i in range(len(question) - n + 1):
                ngram = question[i : i + n]
                ngram_str = " ".join(ngram)
                if not ngram_str.strip():
                    continue

                # exact match
                for col_id, col_toks in col_token_lists.items():
                    if exact_match(ngram, col_toks):
                        for q_id in range(i, i + n):
                            q_col_match[f"{q_id},{col_id}"] = ("CEM", ngram_str)

                for tab_id, tab_toks in tab_token_lists.items():
                    if exact_match(ngram, tab_toks):
                        for q_id in range(i, i + n):
                            q_tab_match[f"{q_id},{tab_id}"] = ("TEM", ngram_str)

                # partial match (only if not already exact)
                for col_id, col_toks in col_token_lists.items():
                    if partial_match(ngram, col_toks):
                        for q_id in range(i, i + n):
                            key = f"{q_id},{col_id}"
                            if key not in q_col_match:
                                q_col_match[key] = ("CPM", ngram_str)

                for tab_id, tab_toks in tab_token_lists.items():
                    if partial_match(ngram, tab_toks):
                        for q_id in range(i, i + n):
                            key = f"{q_id},{tab_id}"
                            if key not in q_tab_match:
                                q_tab_match[key] = ("TPM", ngram_str)

            n -= 1

        return q_col_match, q_tab_match

    # ── Semantic matching (embedding-based, port bổ sung) ────

    def _compute_semantic_matches(
        self, question: str, schema: DatabaseSchema
    ) -> Tuple[dict, dict]:
        """
        Tính cosine similarity giữa embedding của câu hỏi và embedding của
        từng tên cột/bảng (dạng đọc tự nhiên, thay "_" bằng khoảng trắng).
        Chỉ giữ lại các cặp có score >= self._semantic_threshold.

        Returns:
            col_scores: col_global_idx -> score
            tab_scores: tab_idx -> score
        """
        col_scores: dict = {}
        tab_scores: dict = {}

        if not self._use_semantic:
            return col_scores, tab_scores

        try:
            model = _get_embed_model(self._embed_model_name)
        except ImportError as e:
            logger.warning("[SchemaLinker] Bỏ qua semantic matching: %s", e)
            return col_scores, tab_scores

        q_emb = model.encode(question)

        if schema.columns:
            col_names = [c.name.replace("_", " ") for c in schema.columns]
            col_embs = model.encode(col_names)
            for col, emb in zip(schema.columns, col_embs):
                score = _cosine_sim(q_emb, emb)
                if score >= self._semantic_threshold:
                    col_scores[col.global_idx] = score

        if schema.table_names:
            tab_names = [name.replace("_", " ") for name in schema.table_names]
            tab_embs = model.encode(tab_names)
            for idx, emb in enumerate(tab_embs):
                score = _cosine_sim(q_emb, emb)
                if score >= self._semantic_threshold:
                    tab_scores[idx] = score

        return col_scores, tab_scores

    # ── Convert raw dicts → dataclasses ──────────────────────

    def _build_col_matches(
        self, raw: dict, schema: DatabaseSchema, semantic_scores: Optional[dict] = None
    ) -> List[ColumnMatch]:
        """
        raw key = "q_id,col_global_idx", value = "CEM"/"CPM"
        Dedup per col: ưu tiên CEM > CPM.
        Sau đó merge với semantic_scores (col_global_idx -> score): nếu
        semantic score cao hơn score hiện tại của cột đó, thay bằng match_type
        "semantic" — đúng nguyên tắc kết hợp max(score n-gram, score semantic).
        """
        best: dict = {}   # col_global_idx -> (flag, score, span)

        for key, (flag, span) in raw.items():
            _, col_idx = key.split(",")
            col_idx = int(col_idx)
            score = 1.0 if flag == "CEM" else 0.8
            if col_idx not in best or (flag == "CEM" and best[col_idx][0] != "CEM"):
                best[col_idx] = (flag, score, span)

        for col_idx, sem_score in (semantic_scores or {}).items():
            current = best.get(col_idx)
            if current is None or sem_score > current[1]:
                best[col_idx] = ("SEM", sem_score, None)

        matches = []
        col_map = {c.global_idx: c for c in schema.columns}
        match_type_map = {"CEM": "exact", "CPM": "partial", "SEM": "semantic"}
        for col_idx, (flag, score, span) in best.items():
            if col_idx not in col_map:
                continue
            col = col_map[col_idx]
            tbl_name = schema.table_names[col.table_idx]
            matches.append(ColumnMatch(
                col_global_idx=col_idx,
                col_name=col.name,
                table_name=tbl_name,
                match_type=match_type_map[flag],
                score=score,
                matched_span=span,
            ))

        return sorted(matches, key=lambda m: -m.score)

    def _build_tab_matches(
        self, raw: dict, schema: DatabaseSchema, semantic_scores: Optional[dict] = None
    ) -> List[TableMatch]:
        """
        raw key = "q_id,tab_idx", value = "TEM"/"TPM"
        Dedup per tab: ưu tiên TEM > TPM.
        Sau đó merge với semantic_scores (tab_idx -> score) theo nguyên tắc
        max(score n-gram, score semantic), giống _build_col_matches.
        """
        best: dict = {}   # tab_idx -> (flag, score, span)

        for key, (flag, span) in raw.items():
            _, tab_idx = key.split(",")
            tab_idx = int(tab_idx)
            score = 1.0 if flag == "TEM" else 0.8
            if tab_idx not in best or (flag == "TEM" and best[tab_idx][0] != "TEM"):
                best[tab_idx] = (flag, score, span)

        for tab_idx, sem_score in (semantic_scores or {}).items():
            current = best.get(tab_idx)
            if current is None or sem_score > current[1]:
                best[tab_idx] = ("SEM", sem_score, None)

        matches = []
        match_type_map = {"TEM": "exact", "TPM": "partial", "SEM": "semantic"}
        for tab_idx, (flag, score, span) in best.items():
            if tab_idx >= len(schema.table_names):
                continue
            matches.append(TableMatch(
                table_idx=tab_idx,
                table_name=schema.table_names[tab_idx],
                match_type=match_type_map[flag],
                score=score,
                matched_span=span,
            ))

        return sorted(matches, key=lambda m: -m.score)

    # ── Prepare token lists ───────────────────────────────────

    def _prepare_schema_token_lists(self, schema: DatabaseSchema) -> dict:
        """col_global_idx -> List[str] (lowercase, _ split, no stop words)"""
        result = {}
        for col in schema.columns:
            toks = self._tokenize_schema_name(col.name)
            if toks:
                result[col.global_idx] = toks
        return result

    def _prepare_table_token_lists(self, schema: DatabaseSchema) -> dict:
        """tab_idx -> List[str]"""
        result = {}
        for idx, name in enumerate(schema.table_names):
            toks = self._tokenize_schema_name(name)
            if toks:
                result[idx] = toks
        return result

    # ── Cell matching ─────────────────────────────────────────

    def _match_cells(
        self,
        schema: DatabaseSchema,
        question: str,
        cell_retriever: "CellValueRetriever",  # type: ignore[name-defined]
    ) -> List[ColumnMatch]:
        matches: List[ColumnMatch] = []
        seen: Set[int] = set()
        for col in schema.columns:
            if col.global_idx in seen:
                continue
            tbl_name = schema.table_names[col.table_idx]
            hits = cell_retriever.match(
                db_id=schema.db_id, table=tbl_name,
                column=col.name, question=question,
            )
            for cell_val, score in hits:
                if score > 0:
                    seen.add(col.global_idx)
                    matches.append(ColumnMatch(
                        col_global_idx=col.global_idx,
                        col_name=col.name,
                        table_name=tbl_name,
                        match_type="cell",
                        score=score,
                        matched_span=cell_val,
                    ))
                    break
        return matches

    # ── Num/date matching ─────────────────────────────────────

    def _match_num_date(
        self, schema: DatabaseSchema, question: str
    ) -> List[ColumnMatch]:
        num_hits  = _NUM_PAT.findall(question)
        date_hits = _DATE_PAT.findall(question)
        if not num_hits and not date_hits:
            return []

        numeric_types = {"NUMBER", "INT", "INTEGER", "FLOAT", "REAL", "DOUBLE", "DECIMAL"}
        date_types    = {"DATE", "DATETIME", "TIME", "TIMESTAMP", "YEAR"}

        matches: List[ColumnMatch] = []
        seen: Set[int] = set()
        for col in schema.columns:
            if col.global_idx in seen:
                continue
            ctype = col.col_type.upper()
            relevant = (num_hits and ctype in numeric_types) or \
                       (date_hits and any(dt in ctype for dt in date_types))
            if relevant:
                seen.add(col.global_idx)
                matches.append(ColumnMatch(
                    col_global_idx=col.global_idx,
                    col_name=col.name,
                    table_name=schema.table_names[col.table_idx],
                    match_type="num_date",
                    score=0.8,
                ))
        return matches

    # ── Token utilities ───────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Tokenize câu hỏi bằng ViTokenizer.
        Giữ nguyên _ để ViTokenizer xử lý đúng từ ghép tiếng Việt.
        Ví dụ: "sinh viên" -> "sinh_viên", "nhà_xuất_bản" -> "nhà_xuất_bản"
        """
        # Bỏ dấu câu nhưng GIỮ _ (giống _tokenize_schema_name)
        punct_no_underscore = str.maketrans("", "", string.punctuation.replace("_", ""))
        text = text.lower().translate(punct_no_underscore)
        tokenized = ViTokenizer.tokenize(text)
        tokens = tokenized.split()
        return [t for t in tokens if t not in _VI_STOPWORDS and len(t) > 1]

    @staticmethod
    def _tokenize_schema_name(name: str) -> List[str]:
        """
        Tokenize tên bảng/cột để khớp với output của ViTokenizer.

        ViTokenizer ghép từ ghép bằng _ nên câu hỏi cho ra token như:
            "sinh_viên", "giới_tính"  (có dấu _)

        Schema name cần giữ nguyên _ để khớp:
            "sinh_viên" -> ["sinh_viên"]   khớp q_token "sinh_viên"
            "giới_tính" -> ["giới_tính"]   khớp q_token "giới_tính"
            "tên khoa"  -> ["tên", "khoa"] (space vẫn tách bình thường)
        """
        # lowercase + bỏ dấu câu, GIỮ NGUYÊN dấu _ (không dùng _PUNCT_TABLE vì _ bị xóa)
        punct_no_underscore = str.maketrans("", "", string.punctuation.replace("_", ""))
        normalized = name.lower().translate(punct_no_underscore)
        tokens = normalized.split()
        return [t for t in tokens if t not in _VI_STOPWORDS and len(t) > 1]