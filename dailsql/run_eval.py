"""
run_eval.py – Script thực thi đánh giá benchmark.
"""
import json
import logging
from tqdm import tqdm
from text2sql.core.config import ConfigManager
from text2sql_pipeline import Text2SQLPipeline
from text2sql.evaluation.evaluator import Evaluator
from text2sql.sql.sql_executor import SQLExecutor

logging.basicConfig(level=logging.INFO)

def main():
    # 1. Load config (Có thể truyền override qua CLI ở đây)
    config = ConfigManager.load(yaml_path="configs/spider_qwen.yaml")
    
    # 2. Khởi tạo Pipeline và Evaluator
    pipeline = Text2SQLPipeline(config)
    executor = SQLExecutor(config.db_dir)
    evaluator = Evaluator(executor if config.db_dir.exists() else None)
    
    # 3. Load tập dữ liệu (dev.json)
    with open(config.dev_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if config.test_limit:
        data = data[:config.test_limit]
        
    results = []
    logger = logging.getLogger("Eval")
    logger.info(f"Bắt đầu đánh giá trên {len(data)} mẫu...")

    # 4. Vòng lặp dự đoán
    samples_for_eval = []
    for i, item in enumerate(tqdm(data)):
        question = item["question"]
        db_id = item["db_id"]
        gold_sql = item["query"]
        
        try:
            pred_sql = pipeline.run(question, db_id)
            samples_for_eval.append((i, db_id, question, pred_sql, gold_sql))
        except Exception as e:
            logger.error(f"Lỗi tại mẫu {i} (DB: {db_id}): {e}")
            samples_for_eval.append((i, db_id, question, "SELECT 1", gold_sql))

    # 5. Chấm điểm và xuất báo cáo
    scores, agg = evaluator.evaluate(samples_for_eval)
    
    # 6. Lưu kết quả ra file
    output_data = {
        "metrics": agg.__dict__,
        "predictions": [s.__dict__ for s in scores]
    }
    
    config.output_json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config.output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
if __name__ == "__main__":
    main()