"""
evaluate_translated.py
──────────────────────
Đọc results.json → dịch predicted/gold SQL (VI → EN) dùng schema_matching_map.json
→ chạy lại evaluation qua Evaluator của project → xuất results_translated.json.

Đặt file này tại: dailsql/
Chạy từ thư mục:  dailsql/

    python evaluate_translated.py
    python evaluate_translated.py --config configs/spider_qwen.yaml
    python evaluate_translated.py --config configs/spider_qwen.yaml --results result/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── Thêm dailsql/ vào sys.path ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ════════════════════════════════════════════════════════════════════════════
# Imports từ project
# ════════════════════════════════════════════════════════════════════════════

from text2sql.core.config import ConfigManager, PipelineConfig
from text2sql.evaluation.evaluator import Evaluator
from text2sql.sql.sql_executor import SQLExecutor
from pretrain.schema_matching import translate_sql


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="VI→EN SQL translation + re-evaluation")
    parser.add_argument(
        "--config", default="configs/spider_qwen.yaml",
        help="Path tới YAML config (default: configs/spider_qwen.yaml)",
    )
    parser.add_argument(
        "--results", default=None,
        help="Path tới results.json (default: <output_dir>/results.json)",
    )
    args = parser.parse_args()

    # ── 1. Load config ───────────────────────────────────────────────────────
    print(f"\nLoading config : {args.config}")
    cfg: PipelineConfig = ConfigManager.load(yaml_path=args.config)

    results_path = Path(args.results) if args.results else cfg.output_dir / "results.json"
    out_path     = cfg.output_dir / "results_translated.json"

    print(f"  schema_map    : {cfg.schema_map_path}")
    print(f"  db_dir        : {cfg.db_dir}  (exists={cfg.db_dir.exists()})")
    print(f"  results_path  : {results_path}")

    # ── 2. Load schema matching map ──────────────────────────────────────────
    if not cfg.schema_map_path.exists():
        sys.exit(f"❌ Không tìm thấy schema_map_path: {cfg.schema_map_path}")

    with open(cfg.schema_map_path, encoding="utf-8") as f:
        matching_dict: dict = json.load(f)
    print(f"\n✓ Schema map loaded: {len(matching_dict)} databases")

    # ── 3. Load results.json ─────────────────────────────────────────────────
    if not results_path.exists():
        sys.exit(f"❌ Không tìm thấy results: {results_path}")

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    predictions: list = data["predictions"]
    total = len(predictions)
    print(f"✓ Loaded {total} predictions\n")

    # ── 4. Khởi tạo Evaluator với SQLExecutor thật ───────────────────────────
    executor = SQLExecutor(cfg.db_dir) if cfg.db_dir.exists() else None
    evaluator = Evaluator(executor)

    if executor is None:
        print("⚠️  db_dir không tồn tại → exec_acc sẽ = 0.0")

    # ── 5. Dịch VI→EN + chấm điểm qua Evaluator ─────────────────────────────
    print("-" * 70)
    samples_for_eval: list = []   # (index, db_id, question, pred_en, gold_en)
    translation_map:  dict = {}   # index → (pred_en, gold_en)

    for p in predictions:
        idx      = p["index"]
        db_id    = p["db_id"]
        question = p["question"]
        pred_vi  = p["predicted"]
        gold_vi  = p["gold"]

        pred_en = translate_sql(pred_vi, db_id, matching_dict)
        gold_en = translate_sql(gold_vi, db_id, matching_dict)

        translation_map[idx] = (pred_en, gold_en)
        samples_for_eval.append((idx, db_id, question, pred_en, gold_en))

    scores, agg = evaluator.evaluate(samples_for_eval)

    # ── 6. Log từng sample ───────────────────────────────────────────────────
    for s in scores:
        pred_en, gold_en = translation_map[s.index]
        icon = "✓" if s.exec_match else ("≈" if s.exact_match else "✗")
        print(f"[{icon}] #{s.index:02d} | {s.db_id:<20} em={s.exact_match}  exec={s.exec_match}")
        print(f"      PRED_EN : {pred_en[:80]}")
        print(f"      GOLD_EN : {gold_en[:80]}")
        if s.error_msg:
            print(f"      ERR     : {s.error_msg}")
        print()

    # ── 7. Build output JSON ─────────────────────────────────────────────────
    pred_by_idx = {p["index"]: p for p in predictions}

    new_predictions = []
    for s in scores:
        pred_en, gold_en = translation_map[s.index]
        original = pred_by_idx[s.index]
        new_predictions.append({
            **original,
            "predicted_en":    pred_en,
            "gold_en":         gold_en,
            "exact_match":     s.exact_match,
            "exec_match":      s.exec_match,
            "skeleton_match":  s.skeleton_match,
            "component_scores": s.component_scores,
            "complexity":      s.complexity,
            "error_msg":       s.error_msg,
        })

    output = {
        "metrics":     agg.__dict__,
        "predictions": new_predictions,
    }

    # ── 8. Save ──────────────────────────────────────────────────────────────
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── 9. Summary ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("  RESULTS AFTER VI → EN TRANSLATION")
    print("=" * 70)
    print(agg.summary())
    print(f"\n✓ Saved → {out_path}")


if __name__ == "__main__":
    main()
