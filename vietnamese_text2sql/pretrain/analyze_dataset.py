"""
analyze_dataset.py
------------------
Phân tích cấu trúc toàn bộ dataset Spider (EN) và ViSpider (VI word-level).
Chạy: python analyze_dataset.py
"""
import json
from pathlib import Path
from collections import Counter

# ── Đường dẫn — chỉnh lại cho đúng project ──────────────────────────────────
BASE = Path("C:/Users/Admin/Documents/Uni/DS319/Text-to-SQL/data")

FILES = {
    # English Spider
    "EN train"       : BASE / "spider_data/train_spider.json",
    "EN dev"         : BASE / "spider_data/dev_spider.json",
    "EN test"        : BASE / "spider_data/test_spider.json",
    "EN tables"      : BASE / "spider_data/tables_spider.json",
    "EN dev_gold"    : BASE / "spider_data/dev_gold_spider.sql",
    "EN test_gold"   : BASE / "spider_data/test_gold_spider.sql",
    "EN train_gold"  : BASE / "spider_data/train_gold_spider.sql",
    # Vietnamese ViSpider word-level
    "VI train"       : BASE / "word-level/train.json",
    "VI dev"         : BASE / "word-level/dev.json",
    "VI test"        : BASE / "word-level/test.json",
    "VI tables"      : BASE / "word-level/tables.json",
    "VI test_gold"   : BASE / "word-level/test_gold.sql",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def count_sql_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for l in f if l.strip())

def analyze_qa_file(name: str, path: Path):
    """Phân tích file JSON chứa Q&A (train/dev/test)."""
    data = load_json(path)
    db_ids = [r["db_id"] for r in data]
    db_counter = Counter(db_ids)

    print(f"\n{'─'*55}")
    print(f"  📄 {name}  ({path.name})")
    print(f"{'─'*55}")
    print(f"  Tổng số câu hỏi   : {len(data):,}")
    print(f"  Số database dùng  : {len(db_counter)}")
    print(f"  Fields            : {list(data[0].keys())}")

    print(f"\n  Top 10 database (số câu hỏi):")
    for db, cnt in db_counter.most_common(10):
        bar = "█" * (cnt // 5)
        print(f"    {db:<35} {cnt:>4}  {bar}")

    if len(db_counter) > 10:
        print(f"    ... và {len(db_counter)-10} database khác")

    all_dbs = sorted(db_counter.keys())
    print(f"\n  Tất cả databases ({len(all_dbs)}):")
    for i in range(0, len(all_dbs), 5):
        print("    " + ",  ".join(all_dbs[i:i+5]))

    return {
        "total"    : len(data),
        "num_dbs"  : len(db_counter),
        "db_list"  : all_dbs,
        "db_counts": dict(db_counter),
    }


def analyze_tables_file(name: str, path: Path):
    """Phân tích file tables.json — schema thông tin."""
    data = load_json(path)
    
    total_tables  = sum(len(db["table_names"]) for db in data)
    total_cols    = sum(len(db["column_names"]) - 1 for db in data)  # -1 bỏ *
    total_fk      = sum(len(db.get("foreign_keys", [])) for db in data)

    avg_tables = total_tables / len(data)
    avg_cols   = total_cols   / len(data)

    print(f"\n{'─'*55}")
    print(f"  📋 {name}  ({path.name})")
    print(f"{'─'*55}")
    print(f"  Số database       : {len(data)}")
    print(f"  Tổng số bảng      : {total_tables}  (avg {avg_tables:.1f}/DB)")
    print(f"  Tổng số cột       : {total_cols}   (avg {avg_cols:.1f}/DB)")
    print(f"  Tổng Foreign Keys : {total_fk}")

    # DB phức tạp nhất (nhiều bảng nhất)
    sorted_by_tables = sorted(data, key=lambda d: len(d["table_names"]), reverse=True)
    print(f"\n  Top 5 DB phức tạp nhất (nhiều bảng):")
    for db in sorted_by_tables[:5]:
        print(f"    {db['db_id']:<35} {len(db['table_names'])} bảng, {len(db['column_names'])-1} cột")

    return {
        "num_dbs"      : len(data),
        "total_tables" : total_tables,
        "total_cols"   : total_cols,
        "avg_tables"   : avg_tables,
        "avg_cols"     : avg_cols,
    }


def analyze_gold_sql(name: str, path: Path):
    """Phân tích file gold SQL."""
    total = count_sql_lines(path)
    db_counter = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit("\t", 1)
            if len(parts) == 2:
                db_counter[parts[1]] += 1

    print(f"\n{'─'*55}")
    print(f"  🗄️  {name}  ({path.name})")
    print(f"{'─'*55}")
    print(f"  Tổng số SQL       : {total:,}")
    if db_counter:
        print(f"  Số database       : {len(db_counter)}")

    return {"total": total, "num_dbs": len(db_counter)}


def compare_vi_en(vi_stats: dict, en_stats: dict, split: str):
    """So sánh VI vs EN cho cùng một split."""
    print(f"\n{'═'*55}")
    print(f"  📊 So sánh VI vs EN — {split}")
    print(f"{'═'*55}")
    vi_n  = vi_stats["total"]
    en_n  = en_stats["total"]
    ratio = vi_n / en_n * 100 if en_n else 0
    print(f"  VI records : {vi_n:,}")
    print(f"  EN records : {en_n:,}")
    print(f"  VI/EN ratio: {ratio:.1f}%  ({en_n - vi_n:,} records bị thiếu trong VI)")
    print(f"  VI DBs     : {vi_stats['num_dbs']}")
    print(f"  EN DBs     : {en_stats['num_dbs']}")

    vi_dbs = set(vi_stats["db_list"])
    en_dbs = set(en_stats["db_list"])
    only_vi = vi_dbs - en_dbs
    only_en = en_dbs - vi_dbs
    common  = vi_dbs & en_dbs
    print(f"  DB chung   : {len(common)}")
    if only_vi:
        print(f"  Chỉ có VI  : {sorted(only_vi)}")
    if only_en:
        print(f"  Chỉ có EN  : {sorted(only_en)}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  SPIDER & VISPIDER DATASET ANALYSIS")
    print("=" * 55)

    stats = {}

    # --- JSON Q&A files ---
    qa_files = {
        "EN train" : FILES["EN train"],
        "EN dev"   : FILES["EN dev"],
        "EN test"  : FILES["EN test"],
        "VI train" : FILES["VI train"],
        "VI dev"   : FILES["VI dev"],
        "VI test"  : FILES["VI test"],
    }
    for name, path in qa_files.items():
        if path.exists():
            stats[name] = analyze_qa_file(name, path)
        else:
            print(f"\n  ⚠️  {name}: File không tồn tại — {path}")

    # --- Tables files ---
    for name in ["EN tables", "VI tables"]:
        path = FILES[name]
        if path.exists():
            stats[name] = analyze_tables_file(name, path)
        else:
            print(f"\n  ⚠️  {name}: File không tồn tại — {path}")

    # --- Gold SQL files ---
    for name in ["EN dev_gold", "EN test_gold", "EN train_gold", "VI test_gold"]:
        path = FILES[name]
        if path.exists():
            stats[name] = analyze_gold_sql(name, path)
        else:
            print(f"\n  ⚠️  {name}: File không tồn tại — {path}")

    # --- So sánh VI vs EN ---
    for split in ["train", "dev", "test"]:
        vi_key = f"VI {split}"
        en_key = f"EN {split}"
        if vi_key in stats and en_key in stats:
            compare_vi_en(stats[vi_key], stats[en_key], split.upper())

    # --- Tổng kết ---
    print(f"\n{'═'*55}")
    print(f"  📈 TỔNG KẾT")
    print(f"{'═'*55}")
    for name, s in stats.items():
        if "total" in s:
            print(f"  {name:<20}: {s['total']:>6,} records")
    print()


if __name__ == "__main__":
    main()