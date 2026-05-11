"""
Text-to-SQL Pipeline cho bộ dữ liệu Spider Tiếng Việt
======================================================
Pipeline hoàn chỉnh gồm 4 bước:
    1. Load & xử lý schema từ tables.json
    2. Encode tập Train → Cache embedding
    3. Với mỗi câu hỏi test, truy xuất Top-K ví dụ tương đồng (Dynamic Few-shot)
    4. Gọi LLM (qua OpenRouter) sinh SQL, post-process và lưu kết quả

Yêu cầu:
    pip install requests sentence-transformers torch
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
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
    """Toàn bộ tham số cấu hình của pipeline."""

    # ── Đường dẫn dữ liệu ──────────────────────
    base_dir: Path = Path("D:/Uni/DS319/Project/word-level")
    tables_file: str = "tables.json"
    train_file: str = "train.json"
    test_file: str = "test.json"
    output_json_file: str = "test_results.json"
    output_sql_file: str = "predicted.sql"
    embedding_cache_file: str = "train_embeddings_cache.pkl"

    # ── Mô hình & API ──────────────────────────
    embedding_model: str = "moka-ai/m3e-base"
    llm_model: str = "qwen/qwen3-8b"
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_API_KEY",
            "",
        )
    )

    # ── Tham số pipeline ───────────────────────
    top_k_examples: int = 3          # số ví dụ few-shot động
    test_limit: Optional[int] = 20   # None = chạy toàn bộ test set
    max_workers: int = 5             # số luồng song song
    api_retries: int = 3             # số lần retry khi API lỗi
    api_retry_delay: float = 2.0     # giây chờ giữa các retry
    api_timeout: int = 40            # timeout mỗi request (giây)
    llm_temperature: float = 0.0

    # ── Thuộc tính tiện ích ────────────────────
    @property
    def tables_path(self) -> Path:
        return self.base_dir / self.tables_file

    @property
    def train_path(self) -> Path:
        return self.base_dir / self.train_file

    @property
    def test_path(self) -> Path:
        return self.base_dir / self.test_file

    @property
    def output_json_path(self) -> Path:
        return self.base_dir / self.output_json_file

    @property
    def output_sql_path(self) -> Path:
        return self.base_dir / self.output_sql_file

    @property
    def cache_path(self) -> Path:
        return self.base_dir / self.embedding_cache_file


# ─────────────────────────────────────────────
# BƯỚC 1 – XỬ LÝ SCHEMA
# ─────────────────────────────────────────────
class SchemaProcessor:
    """
    Đọc tables.json và tạo schema text theo chuẩn
    OpenAI Demonstration Prompt (ODp) cho mỗi database.

    Output ví dụ:
        # students (id, name, age)
        # courses (id, title, student_id)
    """

    def __init__(self, tables_path: Path) -> None:
        self.tables_path = tables_path
        self._schema_dict: Dict[str, str] = {}

    def load(self) -> "SchemaProcessor":
        logger.info("Đang đọc schema từ %s ...", self.tables_path)
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
        """Chuyển đổi một entry trong tables.json thành schema text."""
        table_names: List[str] = db["table_names_original"]
        columns: List[Tuple[int, str]] = db["column_names_original"]

        # Nhóm cột theo bảng
        col_groups: Dict[int, List[str]] = {i: [] for i in range(len(table_names))}
        for table_idx, col_name in columns:
            if table_idx >= 0:
                col_groups[table_idx].append(col_name)

        schema_lines = [
            f"# {tbl} ({', '.join(col_groups[i])})"
            for i, tbl in enumerate(table_names)
        ]
        return "\n".join(schema_lines)

    def get(self, db_id: str) -> str:
        """Trả về schema text của một database, chuỗi rỗng nếu không tìm thấy."""
        schema = self._schema_dict.get(db_id, "")
        if not schema:
            logger.warning("Không tìm thấy schema cho db_id='%s'", db_id)
        return schema


# ─────────────────────────────────────────────
# BƯỚC 2 – DYNAMIC FEW-SHOT RETRIEVER
# ─────────────────────────────────────────────
class FewShotRetriever:
    """
    Tìm kiếm ngữ nghĩa (semantic search) trên tập Train để lấy
    Top-K ví dụ (câu hỏi, SQL) gần nhất với câu hỏi cần dự đoán.

    Cơ chế:
        - Encode toàn bộ câu hỏi trong train set bằng SentenceTransformer
        - Cache embedding ra file .pkl để tái sử dụng
        - Dùng Cosine Similarity để xếp hạng
    """

    EXAMPLE_TEMPLATE = "# Question: {question}\n# SQL Query: {sql}\n"

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._questions: List[str] = []
        self._sqls: List[str] = []
        self._embeddings: Optional[torch.Tensor] = None
        self._model: Optional[SentenceTransformer] = None

    # ── Public ────────────────────────────────
    def load(self) -> "FewShotRetriever":
        self._load_train_data()
        self._load_model()
        self._load_or_build_embeddings()
        return self

    def retrieve(self, question: str, top_k: int) -> str:
        """
        Trả về chuỗi few-shot gồm top_k ví dụ tương đồng nhất.

        Args:
            question: Câu hỏi tiếng Việt cần tìm ví dụ tương đồng.
            top_k: Số lượng ví dụ muốn lấy.

        Returns:
            Chuỗi văn bản few-shot đã được format sẵn để ghép vào prompt.
        """
        query_emb = self._model.encode(question, convert_to_tensor=True)
        scores = util.cos_sim(query_emb, self._embeddings)[0]
        top_results = torch.topk(scores, k=top_k)

        header = "### Similar examples from training set:\n"
        examples = "".join(
            self.EXAMPLE_TEMPLATE.format(
                question=self._questions[idx],
                sql=self._sqls[idx],
            )
            for _, idx in zip(top_results[0], top_results[1])
        )
        return header + examples

    # ── Private ───────────────────────────────
    def _load_train_data(self) -> None:
        logger.info("Đang đọc tập Train từ %s ...", self.config.train_path)
        with open(self.config.train_path, "r", encoding="utf-8") as fh:
            data: List[dict] = json.load(fh)
        self._questions = [item["question"] for item in data]
        self._sqls = [item["query"] for item in data]
        logger.info("Tập Train: %d mẫu.", len(self._questions))

    def _load_model(self) -> None:
        logger.info("Đang tải embedding model '%s' ...", self.config.embedding_model)
        self._model = SentenceTransformer(self.config.embedding_model)

    def _load_or_build_embeddings(self) -> None:
        cache = self.config.cache_path
        if cache.exists():
            logger.info("Tìm thấy embedding cache → đang load từ %s ...", cache)
            with open(cache, "rb") as fh:
                self._embeddings = pickle.load(fh)
        else:
            logger.info(
                "Chưa có cache → đang encode %d câu hỏi (có thể mất vài phút) ...",
                len(self._questions),
            )
            self._embeddings = self._model.encode(
                self._questions, convert_to_tensor=True, show_progress_bar=True
            )
            with open(cache, "wb") as fh:
                pickle.dump(self._embeddings, fh)
            logger.info("Đã lưu embedding cache tại %s", cache)


# ─────────────────────────────────────────────
# BƯỚC 3 – PROMPT BUILDER
# ─────────────────────────────────────────────
class PromptBuilder:
    """
    Xây dựng prompt hoàn chỉnh gửi cho LLM, bao gồm:
        - System instruction
        - Schema của database
        - Few-shot examples động
        - Câu hỏi cần trả lời
    """

    _SYSTEM_INSTRUCTION = """You are an expert Text-to-SQL system for the Spider benchmark (Vietnamese questions).

Rules:
1. Output ONLY a single valid SQLite SQL query — no explanation, no markdown.
2. Use table aliases (t1, t2, …).
3. Prefer INTERSECT / EXCEPT over complex JOIN + HAVING when appropriate.
4. Always respect foreign-key relationships implied by the schema.
5. Do NOT wrap identifiers in quotes or backticks."""

    def build(
        self,
        schema_text: str,
        few_shot_examples: str,
        question: str,
    ) -> str:
        return (
            f"{self._SYSTEM_INSTRUCTION}\n\n"
            f"### Database Schema:\n{schema_text}\n\n"
            f"{few_shot_examples}\n"
            f"### Question:\n{question}\n\n"
            f"### SQL Query:\n"
        )


# ─────────────────────────────────────────────
# BƯỚC 4 – LLM CLIENT
# ─────────────────────────────────────────────
class OpenRouterClient:
    """
    Gọi API OpenRouter để sinh SQL từ prompt.

    Tính năng:
        - Auto-retry với exponential back-off đơn giản
        - Post-processing để chuẩn hoá SQL output
    """

    _API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._headers = {
            "Authorization": f"Bearer {config.openrouter_api_key}",
            "Content-Type": "application/json",
        }

    def generate(self, prompt: str) -> str:
        """
        Gửi prompt đến LLM và trả về SQL đã được chuẩn hoá.

        Args:
            prompt: Chuỗi prompt hoàn chỉnh.

        Returns:
            SQL query dạng chuỗi một dòng, đã được chuẩn hoá.
            Trả về "SELECT 1" nếu tất cả retry đều thất bại.
        """
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

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "API lỗi (lần %d/%d): %s", attempt, self.config.api_retries, exc
                )
                if attempt < self.config.api_retries:
                    time.sleep(self.config.api_retry_delay)

        logger.error("Tất cả retry thất bại — trả về fallback SQL.")
        return "SELECT 1"

    # ── Post-processing ───────────────────────
    @staticmethod
    def _post_process(text: str) -> str:
        """Chuẩn hoá raw output của LLM thành SQL hợp lệ theo chuẩn Spider."""
        # 1. Bỏ markdown code block nếu có
        if "```" in text:
            parts = text.split("```")
            # lấy nội dung trong block đầu tiên
            text = parts[1] if len(parts) > 1 else parts[0]
            if text.lower().startswith("sql"):
                text = text[3:]

        # 2. Bỏ dấu ngoặc kép, backtick (chuẩn Spider không dùng)
        text = text.replace('"', "").replace("`", "").replace("'", "")

        # 3. Xóa dấu chấm phẩy cuối câu
        text = text.strip().rstrip(";")

        # 4. Sửa lỗi "SELECT SELECT ..."
        while text.upper().startswith("SELECT SELECT"):
            text = text[7:].strip()

        # 5. Đảm bảo bắt đầu bằng SELECT
        if not text.upper().startswith("SELECT"):
            text = "SELECT " + text

        # 6. Chuẩn hoá khoảng trắng
        return " ".join(text.split())


# ─────────────────────────────────────────────
# KẾT QUẢ
# ─────────────────────────────────────────────
@dataclass
class PredictionResult:
    """Kết quả dự đoán cho một câu hỏi."""

    index: int
    db_id: str
    question: str
    predicted_sql: str
    gold_sql: str


# ─────────────────────────────────────────────
# PIPELINE CHÍNH
# ─────────────────────────────────────────────
class Text2SQLPipeline:
    """
    Orchestrator điều phối toàn bộ pipeline:
        SchemaProcessor → FewShotRetriever → PromptBuilder → OpenRouterClient

    Xử lý song song bằng ThreadPoolExecutor.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.schema_processor = SchemaProcessor(config.tables_path).load()
        self.retriever = FewShotRetriever(config).load()
        self.prompt_builder = PromptBuilder()
        self.llm = OpenRouterClient(config)

    # ── Entry point ───────────────────────────
    def run(self) -> List[PredictionResult]:
        """Chạy toàn bộ pipeline và trả về danh sách kết quả đã sắp xếp."""
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
                result = future.result()
                results.append(result)

        # Sắp xếp lại theo thứ tự gốc (đa luồng có thể trả về không tuần tự)
        results.sort(key=lambda r: r.index)
        logger.info("Hoàn tất xử lý %d câu hỏi.", len(results))
        return results

    def save(self, results: List[PredictionResult]) -> None:
        """Lưu kết quả ra file JSON và SQL."""
        self._save_json(results)
        self._save_sql(results)

    # ── Internal helpers ──────────────────────
    def _load_test_data(self) -> List[dict]:
        logger.info("Đang đọc tập Test từ %s ...", self.config.test_path)
        with open(self.config.test_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Tập Test: %d mẫu.", len(data))
        return data

    def _process_single(self, item: dict, idx: int, total: int) -> PredictionResult:
        """
        Xử lý một mẫu dữ liệu qua 4 bước:
            1. Lấy schema
            2. Truy xuất few-shot examples
            3. Xây dựng prompt
            4. Gọi LLM → sinh SQL
        """
        question: str = item["question"]
        db_id: str = item["db_id"]
        gold_sql: str = item["query"]

        logger.info("[%d/%d] db='%s' | question: %s", idx, total, db_id, question[:60])

        # Bước 1: Schema
        schema_text = self.schema_processor.get(db_id)

        # Bước 2: Few-shot retrieval
        few_shot_examples = self.retriever.retrieve(question, top_k=self.config.top_k_examples)

        # Bước 3: Xây dựng prompt
        prompt = self.prompt_builder.build(schema_text, few_shot_examples, question)

        # Bước 4: Gọi LLM
        predicted_sql = self.llm.generate(prompt)

        return PredictionResult(
            index=idx,
            db_id=db_id,
            question=question,
            predicted_sql=predicted_sql,
            gold_sql=gold_sql,
        )

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
        logger.info("Đã lưu JSON kết quả → %s", self.config.output_json_path)

    def _save_sql(self, results: List[PredictionResult]) -> None:
        with open(self.config.output_sql_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(r.predicted_sql for r in results))
        logger.info("Đã lưu file SQL → %s", self.config.output_sql_path)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
def main() -> None:
    config = PipelineConfig(
        # ── Tuỳ chỉnh ở đây nếu cần ─────────────
        # base_dir=Path("path/to/your/data"),
        # test_limit=None,    # None = chạy toàn bộ
        # top_k_examples=3,
        # max_workers=5,
    )

    pipeline = Text2SQLPipeline(config)
    results = pipeline.run()
    pipeline.save(results)

    logger.info("Pipeline hoàn tất. Tổng %d kết quả.", len(results))


if __name__ == "__main__":
    main()