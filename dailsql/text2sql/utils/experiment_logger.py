"""
ExperimentLogger – Ghi log quá trình thực nghiệm ra file JSONL.

Giúp theo dõi chi tiết từng request, prompt, prediction, và lỗi (nếu có)
để thuận tiện cho việc debug và phân tích lỗi sau khi evaluate.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

class ExperimentLogger:
    """
    Ghi từng dòng log dạng JSON vào file .jsonl.
    Phù hợp cho các batch run lớn không thể load toàn bộ vào RAM.
    """
    def __init__(self, log_file: Path | str) -> None:
        self.log_file = Path(log_file)
        # Đảm bảo thư mục tồn tại
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
    def log_sample(
        self,
        index: int,
        db_id: str,
        question: str,
        gold_sql: str,
        predicted_sql: str,
        prompt: str = "",
        is_exact_match: bool = False,
        is_exec_match: bool = False,
        error_msg: str = "",
        additional_info: Dict[str, Any] = None
    ) -> None:
        """Đóng gói và ghi một sample ra file."""
        record = {
            "index": index,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "predicted_sql": predicted_sql,
            "is_exact_match": is_exact_match,
            "is_exec_match": is_exec_match,
            "error_msg": error_msg,
        }
        if prompt:
            record["prompt"] = prompt
        if additional_info:
            record["additional_info"] = additional_info

        self._write(record)

    def log_raw(self, record: Dict[str, Any]) -> None:
        """Ghi log dict tùy ý."""
        self._write(record)

    def _write(self, record: Dict[str, Any]) -> None:
        try:
            with open(self.log_file, "a", encoding="utf-8") as fh:
                # ensure_ascii=False để hiển thị đúng tiếng Việt
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.error("[ExperimentLogger] Không thể ghi log: %s", exc)