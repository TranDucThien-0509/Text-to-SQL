"""
Text-to-SQL Pipeline - DAIL-SQL Architecture
======================================================
Cấu trúc chuẩn DAIL-SQL:
    1. CR_P (Code Representation): Schema dạng CREATE TABLE + PK + FK.
    2. DAIL_S (Masked Question Similarity): Retrieval bằng câu hỏi đã mask literal values.
    3. DAIL_O (DAIL Organization): Ví dụ Few-shot chỉ giữ lại Question & SQL (không schema).
    4. Rule Implication: Ép output chặt chẽ ("with no explanation").
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
import re
import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import torch
from sentence_transformers import SentenceTransformer, util

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────
@dataclass
class PipelineConfig:
    # ── Đường dẫn dữ liệu ──────────────────────
    base_dir: Path = Path("D:/Uni/DS319/Project/word-level")
    tables_file: str = "tables.json"
    train_file: str = "train.json"
    test_file: str = "test.json"
    output_json_file: str = "test_results_dail.json"
    output_sql_file: str = "predicted_dail.sql"

    # ── Mô hình & API ──────────────────────────
    embedding_model: str = "moka-ai/m3e-base"
    llm_model: str = "qwen/qwen3-next-80b-a3b-instruct:free"
    
    # Lấy API Key từ biến môi trường (Bảo mật)
    openrouter_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "")
    )

    # ── Tham số pipeline ───────────────────────
    top_k_examples: int = 3
    test_limit: Optional[int] = 20
    max_workers: int = 5
    api_retries: int = 3
    api_retry_delay: float = 2.0
    api_timeout: int = 40
    llm_temperature: float = 0.0

    @property
    def tables_path(self) -> Path: return self.base_dir / self.tables_file
    @property
    def train_path(self) -> Path: return self.base_dir / self.train_file
    @property
    def test_path(self) -> Path: return self.base_dir / self.test_file
    @property
    def output_json_path(self) -> Path: return self.base_dir / self.output_json_file
    @property
    def output_sql_path(self) -> Path: return self.base_dir / self.output_sql_file


# ─────────────────────────────────────────────
# BƯỚC 1 – SCHEMA PROCESSOR (DAIL-SQL CR_P)
# ─────────────────────────────────────────────
class SchemaProcessor:
    """
    CR_P: sinh CREATE TABLE với PRIMARY KEY + FOREIGN KEY.
    Giúp LLM hiểu sâu về cấu trúc và cách JOIN bảng.
    """

    def __init__(self, tables_path: Path) -> None:
        self.tables_path = tables_path
        self._schema_dict: Dict[str, str] = {}

    def load(self) -> "SchemaProcessor":
        logger.info("Đang đọc schema (CR_P Format) từ %s ...", self.tables_path)
        try:
            with open(self.tables_path, "r", encoding="utf-8") as fh:
                tables_data: List[dict] = json.load(fh)
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy file schema: {self.tables_path}")

        for db in tables_data:
            self._schema_dict[db["db_id"]] = self._format_schema(db)

        logger.info("Đã load %d database schemas.", len(self._schema_dict))
        return self

    @staticmethod
    def _format_schema(db: dict) -> str:
        table_names: List[str] = db["table_names_original"]
        columns: List[Tuple[int, str]] = db["column_names_original"]
        col_types: List[str] = db.get("column_types", [])
        primary_keys: List[int] = db.get("primary_keys", [])
        foreign_keys: List[List[int]] = db.get("foreign_keys", [])

        # Map: col_global_idx -> table_idx
        col_to_table: Dict[int, int] = {}
        # Map: table_idx -> list of (col_global_idx, col_name, col_type)
        table_cols: Dict[int, List[Tuple[int, str, str]]] = {
            i: [] for i in range(len(table_names))
        }
        for global_idx, (table_idx, col_name) in enumerate(columns):
            if table_idx < 0:   # bỏ qua cột ảo "*"
                continue
            col_to_table[global_idx] = table_idx
            col_type = col_types[global_idx] if global_idx < len(col_types) else "TEXT"
            table_cols[table_idx].append((global_idx, col_name, col_type.upper()))

        # Map: table_idx -> list of FK clauses liên quan đến bảng đó
        fk_clauses: Dict[int, List[str]] = {i: [] for i in range(len(table_names))}
        for src_idx, dst_idx in foreign_keys:
            src_table = col_to_table.get(src_idx)
            dst_table = col_to_table.get(dst_idx)
            if src_table is None or dst_table is None:
                continue
            src_col = columns[src_idx][1]
            dst_col = columns[dst_idx][1]
            dst_tbl_name = table_names[dst_table]
            fk_clauses[src_table].append(
                f"  FOREIGN KEY ({src_col}) REFERENCES {dst_tbl_name}({dst_col})"
            )

        pk_set = set(primary_keys)
        blocks: List[str] = []

        for tbl_idx, tbl_name in enumerate(table_names):
            col_defs: List[str] = []
            for global_idx, col_name, col_type in table_cols[tbl_idx]:
                pk_marker = " PRIMARY KEY" if global_idx in pk_set else ""
                col_defs.append(f"  {col_name} {col_type}{pk_marker}")

            col_defs.extend(fk_clauses[tbl_idx])
            body = ",\n".join(col_defs)
            blocks.append(f"CREATE TABLE {tbl_name} (\n{body}\n);")

        return "\n\n".join(blocks)

    def get(self, db_id: str) -> str:
        schema = self._schema_dict.get(db_id, "")
        if not schema:
            logger.warning("Không tìm thấy schema cho db_id='%s'", db_id)
        return schema


# ─────────────────────────────────────────────
# BƯỚC 2 – FEW-SHOT RETRIEVER (DAIL_S + DAIL_O)
# ─────────────────────────────────────────────
class FewShotRetriever:
    """
    Kết hợp Masked Question Similarity và DAIL Organization.
    """

    # DAIL Organization format: Bỏ schema, giữ ánh xạ Q-SQL
    EXAMPLE_TEMPLATE = "/* Answer the following: {question} */\n{sql}\n\n"

    # Value-masking patterns
    _NUM_PAT = re.compile(r"\b\d+(\.\d+)?\b")
    _STR_PAT = re.compile(r"'[^']*'|\"[^\"]*\"")

    def __init__(self, config: PipelineConfig, alpha: float = 1.0) -> None:
        self.config = config
        self.alpha = alpha
        self._questions: List[str] = []         # MASKED questions (để encode)
        self._questions_display: List[str] = [] # GỐC questions (để hiển thị prompt)
        self._sqls: List[str] = []
        self._q_embeddings: Optional[torch.Tensor] = None
        self._sql_embeddings: Optional[torch.Tensor] = None
        self._model: Optional[SentenceTransformer] = None

    @classmethod
    def _mask_question(cls, text: str) -> str:
        text = cls._NUM_PAT.sub("NUM", text)
        text = cls._STR_PAT.sub("STR", text)
        return text

    @classmethod
    def _sql_skeleton(cls, sql: str) -> str:
        sql = cls._NUM_PAT.sub("NUM", sql)
        sql = cls._STR_PAT.sub("STR", sql)
        return " ".join(sql.split())

    def load(self) -> "FewShotRetriever":
        self._load_train_data()
        self._load_model()
        self._load_or_build_embeddings()
        return self

    def retrieve(self, question: str, top_k: int) -> str:
        masked_q = self._mask_question(question)
        q_emb = self._model.encode(masked_q, convert_to_tensor=True)
        q_scores = util.cos_sim(q_emb, self._q_embeddings)[0]

        if self.alpha < 1.0 and self._sql_embeddings is not None:
            sql_query_emb = self._model.encode(masked_q, convert_to_tensor=True)
            sql_scores = util.cos_sim(sql_query_emb, self._sql_embeddings)[0]
            combined = self.alpha * q_scores + (1 - self.alpha) * sql_scores
        else:
            combined = q_scores

        top_results = torch.topk(combined, k=top_k)
        header = "/* Some example questions and corresponding SQL queries are provided based on similar problems: */\n"
        
        # SỬA LỖI BUG: Dùng _questions_display để hiển thị câu hỏi chưa mask vào prompt
        examples = "".join(
            self.EXAMPLE_TEMPLATE.format(
                question=self._questions_display[idx],
                sql=self._sqls[idx],
            )
            for _, idx in zip(top_results[0], top_results[1])
        )
        return header + examples

    def _load_train_data(self) -> None:
        logger.info("Đang đọc tập Train từ %s ...", self.config.train_path)
        with open(self.config.train_path, "r", encoding="utf-8") as fh:
            data: List[dict] = json.load(fh)
            
        self._questions = [self._mask_question(item["question"]) for item in data]
        self._questions_display = [item["question"] for item in data]
        self._sqls_raw = [item["query"] for item in data]
        self._sql_skeletons = [self._sql_skeleton(q) for q in self._sqls_raw]
        self._sqls = self._sqls_raw
        logger.info("Tập Train: %d mẫu.", len(self._questions))

    def _load_model(self) -> None:
        logger.info("Đang tải embedding model '%s' ...", self.config.embedding_model)
        self._model = SentenceTransformer(self.config.embedding_model)

    def _load_or_build_embeddings(self) -> None:
        q_cache = self.config.base_dir / "train_q_embeddings.pkl"
        sql_cache = self.config.base_dir / "train_sql_embeddings.pkl"

        if q_cache.exists():
            logger.info("Load question embeddings cache từ %s", q_cache)
            with open(q_cache, "rb") as fh:
                self._q_embeddings = pickle.load(fh)
        else:
            logger.info("Encode %d câu hỏi (masked) ...", len(self._questions))
            self._q_embeddings = self._model.encode(
                self._questions, convert_to_tensor=True, show_progress_bar=True
            )
            with open(q_cache, "wb") as fh:
                pickle.dump(self._q_embeddings, fh)

        if self.alpha < 1.0:
            if sql_cache.exists():
                logger.info("Load SQL skeleton embeddings cache từ %s", sql_cache)
                with open(sql_cache, "rb") as fh:
                    self._sql_embeddings = pickle.load(fh)
            else:
                logger.info("Encode %d SQL skeletons ...", len(self._sql_skeletons))
                self._sql_embeddings = self._model.encode(
                    self._sql_skeletons, convert_to_tensor=True, show_progress_bar=True
                )
                with open(sql_cache, "wb") as fh:
                    pickle.dump(self._sql_embeddings, fh)


# ─────────────────────────────────────────────
# BƯỚC 3 – PROMPT BUILDER
# ─────────────────────────────────────────────
class PromptBuilder:
    """
    Xây dựng prompt với Rule Implication bắt buộc từ DAIL-SQL.
    """

    _SYSTEM_INSTRUCTION = """You are an expert Text-to-SQL system for the Spider benchmark.
Complete sqlite SQL query only and with no explanation.

Rules:
1. Output ONLY a single valid SQLite SQL query.
2. Use table aliases (e.g., T1, T2) when doing JOINs.
3. Always respect foreign-key relationships implied by the schema.
4. Do NOT wrap identifiers in quotes or backticks."""

    def build(
        self,
        schema_text: str,
        few_shot_examples: str,
        question: str,
    ) -> str:
        return (
            f"{self._SYSTEM_INSTRUCTION}\n\n"
            f"/* Given the following database schema: */\n"
            f"{schema_text}\n\n"
            f"{few_shot_examples}"
            f"/* Answer the following: {question} */\n"
            f"SELECT "
        )


# ─────────────────────────────────────────────
# BƯỚC 4 – LLM CLIENT
# ─────────────────────────────────────────────
class OpenRouterClient:
    _API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        if not self.config.openrouter_api_key:
            raise ValueError("Chưa cấu hình OPENROUTER_API_KEY. Vui lòng set biến môi trường.")

        self._headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "Content-Type": "application/json",
        }

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.config.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.llm_temperature,
        }

        for attempt in range(1, self.config.api_retries + 1):
            try:
                resp = requests.post(
                    self._API_URL,
                    headers=self._headers,
                    json=payload,
                    timeout=self.config.api_timeout,
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()
                return self._post_process(raw)

            except Exception as exc:
                logger.warning("API lỗi (lần %d/%d): %s", attempt, self.config.api_retries, exc)
                if attempt < self.config.api_retries:
                    time.sleep(self.config.api_retry_delay)

        return "SELECT 1"

    @staticmethod
    def _post_process(text: str) -> str:
        # Cắt giải thích thừa (nếu mô hình cố tình giải thích dù đã cấm)
        if ";" in text:
            text = text.split(";")[0]

        if "```" in text:
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else parts[0]
            if text.lower().startswith("sql"):
                text = text[3:]

        text = text.replace('"', "").replace("`", "").replace("'", "")
        text = text.strip()

        while text.upper().startswith("SELECT SELECT"):
            text = text[7:].strip()

        # Vì Prompt builder kết thúc bằng "SELECT ", ta cần chèn lại "SELECT" nếu LLM sinh tiếp
        if not text.upper().startswith("SELECT"):
            text = "SELECT " + text

        return " ".join(text.split())


# ─────────────────────────────────────────────
# KẾT QUẢ & ĐIỀU PHỐI (PIPELINE)
# ─────────────────────────────────────────────
@dataclass
class PredictionResult:
    index: int
    db_id: str
    question: str
    predicted_sql: str
    gold_sql: str

class Text2SQLPipeline:
    def __init__(self, config: PipelineConfig, retriever_alpha: float = 1.0) -> None:
        self.config = config
        self.schema_processor = SchemaProcessor(config.tables_path).load()
        self.retriever = FewShotRetriever(config, alpha=retriever_alpha).load()
        self.prompt_builder = PromptBuilder()
        self.llm = OpenRouterClient(config)

    def run(self) -> List[PredictionResult]:
        test_data = self._load_test_data()
        subset = test_data[: self.config.test_limit] if self.config.test_limit else test_data
        total = len(subset)

        logger.info("Bắt đầu xử lý %d câu hỏi với %d luồng ...", total, self.config.max_workers)
        results: List[PredictionResult] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as pool:
            futures = {
                pool.submit(self._process_single, item, idx + 1, total): idx
                for idx, item in enumerate(subset)
            }
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        results.sort(key=lambda r: r.index)
        logger.info("Hoàn tất xử lý %d câu hỏi.", len(results))
        return results

    def save(self, results: List[PredictionResult]) -> None:
        self._save_json(results)
        self._save_sql(results)

    def _load_test_data(self) -> List[dict]:
        logger.info("Đang đọc tập Test từ %s ...", self.config.test_path)
        with open(self.config.test_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data

    def _process_single(self, item: dict, idx: int, total: int) -> PredictionResult:
        question: str = item["question"]
        db_id: str = item["db_id"]
        gold_sql: str = item["query"]

        logger.info("[%d/%d] db='%s' | question: %s", idx, total, db_id, question[:60])

        schema_text = self.schema_processor.get(db_id)
        few_shot_examples = self.retriever.retrieve(question, top_k=self.config.top_k_examples)
        prompt = self.prompt_builder.build(schema_text, few_shot_examples, question)
        predicted_sql = self.llm.generate(prompt)

        return PredictionResult(idx, db_id, question, predicted_sql, gold_sql)

    def _save_json(self, results: List[PredictionResult]) -> None:
        output = [
            {
                "id": r.index,
                "db_id": r.db_id,
                "question": r.question,
                "predicted_sql": r.predicted_sql,
                "gold_sql": r.gold_sql,
            }
            for r in results
        ]
        with open(self.config.output_json_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, ensure_ascii=False, indent=4)
        logger.info("Đã lưu JSON kết quả -> %s", self.config.output_json_path)

    def _save_sql(self, results: List[PredictionResult]) -> None:
        with open(self.config.output_sql_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(r.predicted_sql for r in results))
        logger.info("Đã lưu file SQL -> %s", self.config.output_sql_path)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def main() -> None:
    # ⚠️ Nhớ export OPENROUTER_API_KEY=sk-or-v1-... trước khi chạy
    config = PipelineConfig()

    # Mặc định dùng alpha=1.0 (Chỉ dùng Question Similarity) để đảm bảo m3e-base tính toán chính xác nhất.
    # Đổi về 0.5 nếu muốn test hybrid search.
    pipeline = Text2SQLPipeline(config, retriever_alpha=1.0)
    
    results = pipeline.run()
    pipeline.save(results)
    logger.info("Pipeline hoàn tất. Tổng %d kết quả.", len(results))

if __name__ == "__main__":
    main()