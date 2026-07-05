"""
FewShotRetriever – DAIL_S + DAIL_O with MMR and hybrid retrieval.

Retrieval score (configurable):
    score = alpha  * question_similarity
          + beta   * sql_skeleton_similarity
          + gamma  * schema_similarity   (future hook)

Supports:
  - Masked question similarity (DAIL_S) — mask CẢ số/chuỗi literal VÀ
    tên cột/bảng (qua SchemaLinker + masker.mask_question), để hai câu
    hỏi cùng "khung cấu trúc" nhưng khác cột/bảng/giá trị cụ thể vẫn
    được nhận diện là tương tự nhau khi tính embedding. Ví dụ:
        "Liệt kê sinh viên có lương trên 5000"
        "Liệt kê giảng viên có tuổi trên 30"
      → cả hai ra cùng khung: "liệt_kê _ có _ trên NUM"
  - SQL skeleton similarity
  - MMR (Maximal Marginal Relevance) for diversity
  - Token-budget-aware selection
  - Duplicate filtering

LƯU Ý: việc mask tên cột/bảng cần biết schema của từng câu hỏi (qua
db_id). Nếu không truyền `schemas` vào constructor, hoặc không truyền
`db_id` khi gọi retrieve(), hệ thống tự fallback về mask số/chuỗi như
bản gốc (không che cột/bảng) — không phá vỡ code cũ đang gọi retrieve()
mà chưa có db_id.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer, util

from text2sql.core.config import PipelineConfig
from text2sql.utils.embedding_cache import EmbeddingCache
from text2sql.schema.schema_linker import SchemaLinker
from text2sql.schema.masker import mask_question_typed

logger = logging.getLogger(__name__)

# ── Masking patterns (số/chuỗi literal) ────────────────────────────────────────
_NUM_PAT = re.compile(r"\b\d+(\.\d+)?\b")
_STR_PAT = re.compile(r"'[^']*'|\"[^\"]*\"")


def _mask_literals(text: str) -> str:
    """Che số và chuỗi trong ngoặc nháy — chạy TRƯỚC schema masking, vì
    _STR_PAT cần dấu nháy còn nguyên (SchemaLinker._tokenize sẽ bóc dấu
    câu, nên nếu chạy sau thì _STR_PAT sẽ không còn gì để khớp)."""
    text = _NUM_PAT.sub("NUM", text)
    return _STR_PAT.sub("STR", text)


def _sql_skeleton(sql: str) -> str:
    sql = _NUM_PAT.sub("NUM", sql)
    sql = _STR_PAT.sub("STR", sql)
    return " ".join(sql.split())


# ── Data container ────────────────────────────────────────────────────────────
@dataclass
class TrainingExample:
    question: str           # original (for display)
    question_masked: str    # for embedding — literal + schema đã che
    sql: str
    sql_skeleton: str
    db_id: str


# ── Retriever ─────────────────────────────────────────────────────────────────
class FewShotRetriever:
    """
    Retrieves the most relevant few-shot examples for a query.

    DAIL_O output format (Question + SQL only, no schema):
        /* Answer the following: {question} */
        {sql}
    """

    EXAMPLE_TEMPLATE = "/* Answer the following: {question} */\n{sql}\n\n"
    _APPROX_CHARS_PER_TOKEN = 4

    def __init__(
        self,
        config: PipelineConfig,
        schemas: Optional[Dict[str, object]] = None,  # db_id -> DatabaseSchema
    ) -> None:
        self.config = config
        self._examples: List[TrainingExample] = []
        self._q_embeddings: Optional[torch.Tensor] = None
        self._sql_embeddings: Optional[torch.Tensor] = None
        self._model: Optional[SentenceTransformer] = None
        self._cache = EmbeddingCache(config.base_dir / ".cache" / "embeddings")

        # Schema cho từng db_id, dùng để che tên cột/bảng khi mask câu hỏi.
        # Không truyền vào -> hệ thống chỉ che số/chuỗi như hành vi gốc.
        self._schemas: Dict[str, object] = schemas or {}
        self._linker = SchemaLinker()

    # ── Public API ─────────────────────────────────────────────────────────

    def load(self) -> "FewShotRetriever":
        self._load_examples()
        self._load_model()
        self._load_or_build_embeddings()
        return self

    def retrieve(
        self,
        question: str,
        db_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Return formatted few-shot block ready to insert into a prompt.

        Args:
            question: The natural-language question to find examples for.
            db_id:    Database mà câu hỏi này thuộc về — cần để che tên
                      cột/bảng khi tính similarity. Nếu không truyền, chỉ
                      số/chuỗi literal được che (không che cột/bảng).
            top_k:    Number of examples. Falls back to config.top_k_examples.
        """
        k = top_k or self.config.top_k_examples
        selected = self._select_examples(question, db_id, k)

        header = (
            "/* Some example questions and corresponding SQL queries "
            "are provided based on similar problems: */\n"
        )
        body = "".join(
            self.EXAMPLE_TEMPLATE.format(question=ex.question, sql=ex.sql)
            for ex in selected
        )
        return header + body

    def retrieve_with_scores(
        self,
        question: str,
        db_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Tuple[TrainingExample, float]]:
        """Return (example, score) pairs – useful for experiment logging."""
        k = top_k or self.config.top_k_examples
        return self._rank_examples(question, db_id, k * 3)[:k]

    # ── Masking: số/chuỗi literal + tên cột/bảng ────────────────────────────

    def _mask_structural(self, question: str, db_id: Optional[str]) -> str:
        """
        Che số/chuỗi literal trước, sau đó che tiếp tên cột/bảng nếu xác
        định được schema của db_id — chỉ giữ lại "khung cấu trúc" của câu
        hỏi (không che cell value, vì mục đích ở đây là so khớp CẤU TRÚC
        câu hỏi, không phải giá trị lọc cụ thể).
        """
        masked = _mask_literals(question)

        schema = self._schemas.get(db_id) if db_id else None
        if schema is None:
            # Không có schema cho db_id này -> giữ hành vi gốc (chỉ literal)
            return masked

        # cell_retriever=None: không cần truy cập DB, không che cell value,
        # chỉ lấy q_col_match / q_tab_match để che tên cột/bảng.
        result = self._linker.link(masked, schema, cell_retriever=True)
        # Dùng bản typed (COLUMN/TABLE/VALUE riêng biệt) thay vì mask_question
        # gốc (1 tag "_" chung) — giữ thêm tín hiệu loại thực thể cho similarity.
        return mask_question_typed(masked, result, collapse_consecutive=True)

    # ── Core selection logic ────────────────────────────────────────────────

    def _select_examples(
        self, question: str, db_id: Optional[str], k: int
    ) -> List[TrainingExample]:
        if self.config.use_mmr:
            return self._select_mmr(question, db_id, k)
        pairs = self._rank_examples(question, db_id, k)
        return [ex for ex, _ in pairs]

    def _rank_examples(
        self, question: str, db_id: Optional[str], k: int
    ) -> List[Tuple[TrainingExample, float]]:
        """Return top-k examples by hybrid score, deduplicated."""
        masked_q = self._mask_structural(question, db_id)
        q_emb = self._model.encode(masked_q, convert_to_tensor=True)

        # Question similarity (alpha weight)
        q_scores = util.cos_sim(q_emb, self._q_embeddings)[0]
        combined = self.config.retrieval_alpha * q_scores

        # SQL skeleton similarity (beta weight)
        if self.config.retrieval_beta > 0 and self._sql_embeddings is not None:
            skel_emb = self._model.encode(masked_q, convert_to_tensor=True)
            sql_scores = util.cos_sim(skel_emb, self._sql_embeddings)[0]
            combined = combined + self.config.retrieval_beta * sql_scores

        top = torch.topk(combined, k=min(k * 2, len(self._examples)))
        seen_sqls: set = set()
        results: List[Tuple[TrainingExample, float]] = []

        for score_t, idx_t in zip(top.values, top.indices):
            idx = idx_t.item()
            ex = self._examples[idx]
            # Duplicate SQL filter
            if ex.sql_skeleton in seen_sqls:
                continue
            seen_sqls.add(ex.sql_skeleton)
            results.append((ex, float(score_t)))
            if len(results) >= k:
                break

        return results

    def _select_mmr(
        self, question: str, db_id: Optional[str], k: int
    ) -> List[TrainingExample]:
        """
        Maximal Marginal Relevance selection for diversity-aware retrieval.

        Score(d) = lambda * relevance(d, q)
                 - (1 - lambda) * max_similarity(d, selected)
        """
        lam = self.config.mmr_lambda
        pool_size = min(k * 5, len(self._examples))
        pool = self._rank_examples(question, db_id, pool_size)  # (ex, score)
        if not pool:
            return []

        pool_exs = [ex for ex, _ in pool]
        pool_scores = torch.tensor([s for _, s in pool])

        # Encode all pool examples (đã có question_masked che sẵn từ lúc load)
        pool_texts = [ex.question_masked for ex in pool_exs]
        pool_embs = self._model.encode(pool_texts, convert_to_tensor=True)

        selected_indices: List[int] = []
        selected_embs: List[torch.Tensor] = []

        for _ in range(k):
            if not selected_embs:
                # First: pick highest relevance
                best = int(torch.argmax(pool_scores).item())
            else:
                stacked = torch.stack(selected_embs)
                sim_to_selected = util.cos_sim(pool_embs, stacked).max(dim=1).values
                mmr_scores = lam * pool_scores - (1 - lam) * sim_to_selected
                # Mask already selected
                for si in selected_indices:
                    mmr_scores[si] = -1e9
                best = int(torch.argmax(mmr_scores).item())

            selected_indices.append(best)
            selected_embs.append(pool_embs[best])

        return [pool_exs[i] for i in selected_indices]

    # ── Data loading ────────────────────────────────────────────────────────

    def _load_examples(self) -> None:
        import json
        logger.info("Loading training examples from %s", self.config.train_path)
        with open(self.config.train_path, "r", encoding="utf-8") as fh:
            data: List[dict] = json.load(fh)

        self._examples = [
            TrainingExample(
                question=item["question"],
                question_masked=self._mask_structural(item["question"], item["db_id"]),
                sql=item["query"],
                sql_skeleton=_sql_skeleton(item["query"]),
                db_id=item["db_id"],
            )
            for item in data
        ]
        logger.info("Loaded %d training examples.", len(self._examples))

    def _load_model(self) -> None:
        logger.info("Loading embedding model '%s'", self.config.embedding_model)
        self._model = SentenceTransformer(self.config.embedding_model)

    def _load_or_build_embeddings(self) -> None:
        q_texts = [ex.question_masked for ex in self._examples]
        self._q_embeddings = self._cache.get("train_questions", q_texts)
        if self._q_embeddings is None:
            logger.info("Encoding %d masked questions...", len(q_texts))
            self._q_embeddings = self._model.encode(
                q_texts, convert_to_tensor=True, show_progress_bar=True
            )
            self._cache.put("train_questions", q_texts, self._q_embeddings)

        if self.config.retrieval_beta > 0:
            sql_texts = [ex.sql_skeleton for ex in self._examples]
            self._sql_embeddings = self._cache.get("train_sql_skeletons", sql_texts)
            if self._sql_embeddings is None:
                logger.info("Encoding %d SQL skeletons...", len(sql_texts))
                self._sql_embeddings = self._model.encode(
                    sql_texts, convert_to_tensor=True, show_progress_bar=True
                )
                self._cache.put("train_sql_skeletons", sql_texts, self._sql_embeddings)