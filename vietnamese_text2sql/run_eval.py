"""
run_eval.py – Script thực thi đánh giá benchmark.
"""
import json
import logging
import time
import dataclasses
from tqdm import tqdm
from text2sql.core.config import ConfigManager
from text2sql_pipeline import Text2SQLPipeline
from text2sql.evaluation.evaluator import Evaluator
from text2sql.sql.sql_executor import SQLExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("Eval")


def to_dict(obj) -> dict:
    """Serialize an object to dict, supporting dataclass, __dict__, namedtuple."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Cannot serialize object of type {type(obj)}")


def save_checkpoint(samples: list, total: int, checkpoint_path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True) 
    checkpoint_data = {
        "progress": f"{len(samples)}/{total}",
        "predictions_so_far": [
            {"idx": s[0], "db_id": s[1], "question": s[2],
             "pred_sql": s[3], "gold_sql": s[4]}
            for s in samples
        ]
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)


def main():
    CHECKPOINT_EVERY = 10

    # 1. Load config
    config = ConfigManager.load(yaml_path="configs/spider_qwen.yaml")

    # 2. Khởi tạo Pipeline
    pipeline = Text2SQLPipeline(config)

    # 3. Khởi tạo Executor và Evaluator
    if config.db_dir.exists():
        executor = SQLExecutor(config.db_dir)
    else:
        logger.warning(f"db_dir không tồn tại: {config.db_dir}. Chỉ đánh giá Exact Match.")
        executor = None
    evaluator = Evaluator(executor)

    # 4. Load tập dữ liệu
    with open(config.dev_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if config.test_limit:
        data = data[0:config.test_limit] 

    logger.info(f"Bắt đầu đánh giá trên {len(data)} mẫu...")
    start_time = time.time()

    # 5. Vòng lặp dự đoán
    error_count = 0
    samples_for_eval = []
    checkpoint_path = config.output_json_path.with_suffix(".checkpoint.json")

    for i, item in enumerate(tqdm(data, desc="Predicting")):
        question = item["question"]
        db_id = item["db_id"]
        gold_sql = item["query"]

        try:
            pred_sql = pipeline.run(question, db_id)
        except Exception as e:
            logger.error(f"Lỗi tại mẫu {i} (DB: {db_id}): {e}")
            pred_sql = "SELECT 1"
            error_count += 1

        samples_for_eval.append((i, db_id, question, pred_sql, gold_sql))

        # Auto-save checkpoint sau mỗi CHECKPOINT_EVERY mẫu
        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(samples_for_eval, len(data), checkpoint_path)
            logger.info(f"[Checkpoint] Đã lưu {i + 1}/{len(data)} mẫu → {checkpoint_path}")

    elapsed = time.time() - start_time
    logger.info(
        f"Dự đoán xong: {len(data)} mẫu, {error_count} lỗi, "
        f"thời gian: {elapsed:.1f}s ({elapsed/len(data):.2f}s/mẫu)"
    )

    # 6. Chấm điểm
    scores, agg = evaluator.evaluate(samples_for_eval)

    # Log kết quả ra console
    logger.info("=== KẾT QUẢ ĐÁNH GIÁ ===")
    for k, v in to_dict(agg).items():
        logger.info(f"  {k}: {v}")

    # 7. Lưu kết quả chính thức
    output_data = {
        "metrics": to_dict(agg),
        "predictions": [to_dict(s) for s in scores],
    }

    config.output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Đã lưu kết quả tại: {config.output_json_path}")

    # 8. Xóa checkpoint sau khi hoàn thành
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Đã xóa checkpoint file.")


if __name__ == "__main__":
    main()